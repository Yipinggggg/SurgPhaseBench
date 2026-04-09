import math
import copy

import torch
import torch.nn as nn

from einops import repeat, rearrange
from einops.layers.torch import Rearrange

from .layer import MyActivation, MyDropout1d, MyLayerNorm, MyLinear, UpsamplingConv1d, DownsamplingConv1d, \
    MyAttention, create_causal_mask, create_relative_position_index, create_local_mask, PositionalEncoding, \
    ConvBlock, ResBlock
from .token_aggregator import TokenAggregator, SimpleTokenAggregator
from .template import ModelTemplate, TrainerTemplate
from DinoTemporal.TemporalModel.plot import visualize_predictions, visualize_bottleneck_outputs, visualize_masked_bottleneck_outputs, \
    visualize_attention_weights
from DinoTemporal.utils import Cholec80, calculate_metrics
from DinoTemporal.utils import LOGITS, FEATURES, ATTENTION_WEIGHTS, BOTTLENECK_LOGITS, \
    FEATURE_SEQ, LABEL_SEQ, LABEL_SEQ_MULTISCALE, PADDING_MASK, LABELS_BOTTLENECK, TOKEN_MASK, MASKED_RATIO


class ConvolutionalDownsampling(nn.Module):
    def __init__(self, dim, downscale=2, expansion=2, depthwise=False, causal=True,
                 init_method='trunc_normal', normalize=True):
        super().__init__()
        # spatial downsampling by factor 'downscale' + feature map width will increase by factor 'expansion'

        self.scale_factor = downscale
        self.layer = nn.Sequential(
            MyLayerNorm(dim) if normalize else nn.Identity(),
            DownsamplingConv1d(dim, int(dim * expansion), causal, factor=downscale, depthwise=depthwise,
                               init_method=init_method, followed_by_relu=False),
        )

    def forward(self, t):  # t = (x, mask); x is of shape N x S x C, mask is N x S (1 --> keep, 0 --> mask)
        x, mask = self.layer(t)
        mask = mask[:, ::self.scale_factor]  # downsample mask as well

        return x, mask


class ConvolutionalUpsampling(nn.Module):
    def __init__(self, dim, upscale=2, shrinking=2, depthwise=False, init_method='trunc_normal', normalize=True):
        super().__init__()
        #  spatial upsampling by factor 'upscale' + feature map width will decrease by factor 'shrinking'

        self.layer = nn.Sequential(
            MyLayerNorm(dim) if normalize else nn.Identity(),
            UpsamplingConv1d(dim, int(dim / shrinking), factor=upscale, depthwise=depthwise,
                             init_method=init_method, followed_by_relu=False)
        )

    def forward(self, t):  # t = (x, mask); x is of shape N x S x C, mask is N x S (1 --> keep, 0 --> mask)
        x, mask = self.layer(t)

        return x, None  # low-resolution mask will be invalid


class ConvStageDown(nn.Module):  # stack of convolutional blocks --> downsampling
    def __init__(self, dim, block_count, downscale, expansion, causal, block_cfg, down_cfg,
                 add_attention_cfg=None):
        super().__init__()

        if add_attention_cfg is None:
            block_cfg['is_causal'] = causal
            block_cfg['dim'] = dim
            blocks = []
            for i in range(block_count):
                blocks.append(ResBlock(dim, block_cfg=block_cfg))
            self.blocks = nn.Sequential(*blocks)
            # block_cfg:
            # dilation, kernel_size=3, activation="relu", dropout=0.5, depthwise=False, init_method='trunc_normal'
        else:
            self.blocks = ConvAttentionStack(dim, block_cfg, **add_attention_cfg, causal=causal, num_class=None)
            # add_attention_cfg:
            # attn_cfg, mlp_cfg, conv_attn_block_cfg, max_seq_len
            # **transformer_cfg

        if downscale == 1 and expansion == 1:
            self.down = nn.Identity()
        else:
            down_cfg['causal'] = causal
            down_cfg['init_method'] = block_cfg['init_method']
            self.down = ConvolutionalDownsampling(dim, downscale, expansion, **down_cfg)
            # down_cfg:
            # depthwise=False, normalize=True

    def forward(self, x, mask):  # x is of shape N x S x C, mask is N x S (1 --> keep, 0 --> mask)
        if isinstance(self.blocks, ConvAttentionStack):
            out = self.blocks(x, mask)
            interm = out[FEATURES]
        else:
            interm, mask = self.blocks((x, mask))
        x, mask_small = self.down((interm, mask))

        return (x, mask_small), (interm, mask)


class ConvStageUp(nn.Module):  # upsampling --> fusion (skip connection) --> convolutional blocks --> classifier
    def __init__(self, dim, block_count, upscale, shrinking, causal, block_cfg, up_cfg, num_class=-1,
                 weighted_fusion=False, fusion_weight_initial_value=1.0, add_attention_cfg=None):
        super().__init__()
        # dim: *after* upsampling

        if upscale == 1 and shrinking == 1:
            self.up = nn.Identity()
        else:
            up_cfg['init_method'] = block_cfg['init_method']
            self.up = ConvolutionalUpsampling(int(dim * shrinking), upscale, shrinking, **up_cfg)
            # up_cfg:
            # depthwise=False, normalize=True

        if add_attention_cfg is None:
            block_cfg['is_causal'] = causal
            block_cfg['dim'] = dim
            blocks = []
            for i in range(block_count):
                blocks.append(ResBlock(dim, block_cfg=block_cfg))
            self.blocks = nn.Sequential(*blocks)
            # block_cfg:
            # dilation, kernel_size=3, activation="relu", dropout=0.5, depthwise=False, init_method='trunc_normal'

            if num_class > 0:
                self.head = nn.Sequential(
                    MyLayerNorm(dim) if up_cfg['normalize'] is True else nn.Identity(),  # TODO: configure properly
                    MyLinear(dim, dim_out=num_class, init_method=block_cfg['init_method'], followed_by_relu=False)
                )
            else:
                self.head = None
        else:
            self.blocks = ConvAttentionStack(dim, block_cfg, **add_attention_cfg, causal=causal,
                                             num_class=None if num_class <= 0 else num_class)
            # add_attention_cfg:
            # attn_cfg, mlp_cfg, conv_attn_block_cfg, max_seq_len
            # **transformer_cfg

        if weighted_fusion is True:
            self.t = nn.Parameter(torch.ones(1) * fusion_weight_initial_value, requires_grad=True)
        else:
            self.t = nn.Parameter(torch.tensor([0.5]), requires_grad=False)
        self.weighted_fusion = weighted_fusion

    def extra_repr(self) -> str:
        return 'learnable_fusion_weight={}'.format(self.t.requires_grad)

    def forward(self, x, mask, mask_high_res, inp=None):
        result = dict()

        x, _ = self.up((x, mask))
        if inp is not None:  # skip connection
            if self.weighted_fusion is True:
                t = torch.clamp(self.t, min=0, max=2)  # -1 <= (1 - t) <= 1
            else:
                t = self.t  # == 0.5
            x = t * x + (1 - t) * inp

        if isinstance(self.blocks, ConvAttentionStack):
            out = self.blocks(x, mask_high_res)
            result[FEATURES] = (out[FEATURES], mask_high_res)
            result[LOGITS] = out[LOGITS]
        else:
            out, mask = self.blocks((x, mask_high_res))
            result[FEATURES] = (out, mask)

            if self.head is not None:
                logits, _ = self.head((out, mask))
                result[LOGITS] = logits

        return result


class ConvAttentionBlock(nn.Module):  # convolution --> self-attention --> mlp
    def __init__(self, dim, conv_block_cfg, attn_cfg, mlp_cfg, max_seq_len, causal=True, skip_conv=False,
                 skip_attn=False, attn_dilated=True, dilation_factor=1, normalize=True, residual_dropout=0.,
                 sinusoidal_pe=None):
        super().__init__()

        self.pe = sinusoidal_pe  # if not None, use sinusoidal positional encoding
        if sinusoidal_pe is None:
            # temporal convolution
            if skip_conv is True:
                self.conv = nn.Identity()
            else:
                conv_block_cfg['dim'] = dim
                conv_block_cfg['dilation'] = dilation_factor
                conv_block_cfg['is_causal'] = causal
                conv_block = ConvBlock(**conv_block_cfg)
                self.conv = ResBlock(dim, block=conv_block, dropout=residual_dropout)
                # conv_block_cfg:
                # kernel_size=3, activation="relu", dropout=0.5, depthwise=False, init_method='trunc_normal'

        if skip_attn is True:
            self.attn, self.ff = None, None
        else:
            if attn_dilated is True:
                # Code based on https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/max_vit.py
                self.attn_dilation = (conv_block_cfg['kernel_size'] - 1) * dilation_factor + 1  # effective kernel size
                self.grid_fn = Rearrange('n (t d) c -> (n d) t c', d=self.attn_dilation)
                self.grid_mask_fn = Rearrange('n (t d) -> (n d) t', d=self.attn_dilation)
                self.ungrid_fn = Rearrange('(n d) t c -> n (t d) c', d=self.attn_dilation)
            else:
                self.attn_dilation = 1

            # self-attention
            attn_cfg['in_dim'] = dim
            if 'attn_dim' not in attn_cfg:
                if 'dim_expansion' in attn_cfg:
                    attn_cfg['attn_dim'] = int(dim * attn_cfg['dim_expansion'])
                    del attn_cfg['dim_expansion']
                else:
                    attn_cfg['attn_dim'] = None
            attn_cfg['causal'] = causal
            attn_cfg['max_len'] = max_seq_len
            attn_layer = MyAttention(**attn_cfg)
            # attn_cfg:
            # attn_dim (dim_expansion), nheads=4, relative_position_bias=False, attn_dropout=0., proj_bias=False,
            # init_method='trunc_normal'
            attn_block = nn.Sequential(
                MyLayerNorm(dim) if normalize is True else nn.Identity(),
                attn_layer
            )
            self.attn = ResBlock(dim, block=attn_block, dropout=residual_dropout, has_attn=True)

            # pointwise feed-forward network
            ff_hidden_dim = int(dim * mlp_cfg['dim_expansion'])
            ff_activation = mlp_cfg['activation']
            ff_init = attn_cfg['init_method']
            ff_block = nn.Sequential(
                MyLayerNorm(dim) if normalize is True else nn.Identity(),
                MyLinear(dim, dim_out=ff_hidden_dim, init_method=ff_init, followed_by_relu=ff_activation in {'relu', 'gelu'}),
                MyActivation(ff_activation),
                MyDropout1d(p=mlp_cfg['dropout']),
                MyLinear(ff_hidden_dim, dim_out=dim, init_method=ff_init, followed_by_relu=False),
            )
            # mlp_cfg:
            # dim_expansion, activation, dropout
            self.ff = ResBlock(dim, block=ff_block, dropout=residual_dropout)

    def extra_repr(self) -> str:
        repr = ""
        if self.attn is not None:
            repr = 'attn_dilation={}'.format(self.attn_dilation)
        if self.pe is not None:
            pe_ = "sinusoidal"
        else:
            if isinstance(self.conv, nn.Identity):
                pe_ = "none"
            else:
                pe_ = "convolutional (CoPE)"
        repr += '; pe={}'.format(pe_)
        return repr

    def set_rel_pos_indices(self, rel_pos_indices):
        if self.attn is not None:
            self.attn.block[1].set_rel_pos_indices(rel_pos_indices)

    def set_valid_mask(self, valid_mask):
        if self.attn is not None:
            attn_block = self.attn.block[1]
            if isinstance(attn_block, MyAttention):
                self.attn.block[1].set_valid_mask(valid_mask)
            else:
                print("Cannot set valid mask on instance of {}".format(type(attn_block)))

    def forward(self, x, mask):  # x is of shape N x S x C, mask is N x S (1 --> keep, 0 --> mask)
        N, S, C = x.shape

        if self.pe is not None:
            x = self.pe(x)
        else:
            x, _ = self.conv((x, mask))
        if self.attn is None:
            attn_weights = None
        else:
            if self.attn_dilation == 1:
                attn_out = self.attn((x, mask))
                x, attn_weights = attn_out[FEATURES], attn_out[ATTENTION_WEIGHTS]
                x, _ = self.ff((x, mask))
            else:
                # compute dilated attention with dilation factor (--> self.attn_dilation) > 1
                to_pad = 0
                if (S % self.attn_dilation) != 0:  # padding required
                    target_len = math.ceil(S / self.attn_dilation) * self.attn_dilation
                    to_pad = target_len - S
                    assert (0 < to_pad < self.attn_dilation)
                    x = torch.cat([x, torch.zeros((N, to_pad, C), device=x.device, dtype=x.dtype)], dim=1)
                    mask_ = torch.cat([mask, torch.zeros((N, to_pad), device=mask.device, dtype=mask.dtype)], dim=1)
                else:
                    mask_ = mask

                x = self.grid_fn(x)  # (N * d) x S' x C
                mask_ = self.grid_mask_fn(mask_)  # (N x d) x S'

                attn_out = self.attn((x, mask_))
                x, attn_weights = attn_out[FEATURES], attn_out[ATTENTION_WEIGHTS]
                x = self.ungrid_fn(x)  # N x (S' * d) x C
                if to_pad > 0:
                    x = x[:, :-to_pad, :]
                x, _ = self.ff((x, mask))

                with torch.no_grad():  # inflate computed attention weights to original dimensions
                    nheads = self.attn.block[1].nheads
                    attn_weights = rearrange(attn_weights, '(n d) h s t -> n h d s t', d=self.attn_dilation)  # N x nheads x d x S' x S'
                    reconstructed_attn_weights = []
                    for n in range(N):
                        for h in range(nheads):
                            attn_weights_ = []
                            for d in range(self.attn_dilation):
                                mat_ = torch.zeros((self.attn_dilation, self.attn_dilation),
                                                   dtype=attn_weights.dtype, device=attn_weights.device)
                                mat_[d, d] = 1
                                attn_weights_.append(torch.kron(attn_weights[n, h, d], mat_).unsqueeze(0))  # (S' * d) x (S' * d)
                            reconstructed_attn_weights.append(torch.cat(attn_weights_, dim=0).sum(dim=0).unsqueeze(0))
                    del attn_weights
                    attn_weights = torch.cat(reconstructed_attn_weights, dim=0)  # (N * nheads) x S x S
                    attn_weights = rearrange(attn_weights, '(n h) s t -> n h s t', h=nheads)

        return {
            FEATURES: x,
            ATTENTION_WEIGHTS: attn_weights,
        }


class ConvAttentionStack(nn.Module):  # blocks with convolutional attention --> classifier
    def __init__(self, dim, conv_block_cfg, attn_cfg, mlp_cfg, conv_attn_block_cfg, max_seq_len, num_class=None,
                 nlayers=6, dilation_mode='exponential', dilation_factors=None,
                 causal=True, num_cls_heads=1, classifier_dropout=0., normalize=True,
                 attention_masks_cfg=None, local_attn_window_size=7, use_sinusoidal_pe=False):
        super().__init__()
        self.num_class = num_class

        self.causal = causal
        if causal is True:
            assert (attention_masks_cfg is None)
            self.attention_masks_cfg = None
            causal_mask = create_causal_mask(max_seq_len)
            self.register_buffer('causal_mask', causal_mask, persistent=False)
        else:
            if attention_masks_cfg is not None:
                assert (len(attention_masks_cfg) == nlayers)
                self.attention_masks_cfg = attention_masks_cfg
            else:
                self.attention_masks_cfg = ["mask_none"] * nlayers

            if "mask_future" in self.attention_masks_cfg:
                future_mask = create_causal_mask(max_seq_len)  # allowed to attend to past positions only
                self.register_buffer('future_mask', future_mask, persistent=False)
            if "mask_past" in self.attention_masks_cfg:
                past_mask = torch.t(create_causal_mask(max_seq_len))  # allowed to attend to future positions only
                self.register_buffer('past_mask', past_mask, persistent=False)
            if "mask_local" in self.attention_masks_cfg:
                local_mask = create_local_mask(max_seq_len, local_attn_window_size)
                self.register_buffer('local_mask', local_mask, persistent=False)

        self.attn_relative = attn_cfg['relative_position_bias']
        if self.attn_relative is True:
            rel_pos_indices = create_relative_position_index(max_seq_len, causal=causal)
            self.register_buffer('rel_pos_indices', rel_pos_indices, persistent=False)

        if use_sinusoidal_pe is True:
            self.pe = PositionalEncoding(dim, max_len=max_seq_len)

        if dilation_factors is None:
            assert (dilation_mode is not None)
            if dilation_mode == 'exponential':
                dilation_factors = [2 ** i for i in range(nlayers)]
            elif dilation_mode == 'none':
                dilation_factors = [1 for i in range(nlayers)]
            else:
                assert (dilation_mode == 'linear')
                dilation_factors = [i + 1 for i in range(nlayers)]
        else:
            assert (len(dilation_factors) == nlayers)

        conv_attn_block_cfg['normalize'] = normalize
        layers = []
        for i in range(nlayers):
            layers.append(ConvAttentionBlock(
                dim, copy.deepcopy(conv_block_cfg), attn_cfg, mlp_cfg, max_seq_len, causal=causal,
                dilation_factor=dilation_factors[i], **conv_attn_block_cfg,
                sinusoidal_pe=(self.pe if use_sinusoidal_pe is True else None)
            ))
        self.layers = nn.ModuleList(layers)
        # conv_attn_block_cfg:
        # skip_conv=False, skip_attn=False, attn_dilated=True, residual_dropout=0.

        if num_class is not None:
            assert (num_cls_heads > 0)
            self.multi_task = num_cls_heads > 1
            self.head = nn.Sequential(
                MyLayerNorm(dim) if normalize is True else nn.Identity(),
                MyDropout1d(p=classifier_dropout),
                MyLinear(dim, dim_out=num_class * num_cls_heads, init_method=attn_cfg['init_method'],
                         followed_by_relu=False),
            )
        else:
            self.head = None

    def extra_repr(self) -> str:
        if self.attention_masks_cfg is not None:
            return 'attention_masking={}'.format(self.attention_masks_cfg)
        else:
            return ""

    def setup(self):  # call this after model has been sent to GPU
        for layer in self.layers:
            if self.attn_relative:
                layer.set_rel_pos_indices(self.rel_pos_indices)

        if self.causal:
            for layer in self.layers:
                layer.set_valid_mask(self.causal_mask)
        else:
            for i in range(len(self.layers)):
                attn_mask_cfg = self.attention_masks_cfg[i]
                if attn_mask_cfg == "mask_none":
                    pass
                elif attn_mask_cfg == "mask_future":
                    self.layers[i].set_valid_mask(self.future_mask)
                elif attn_mask_cfg == "mask_past":
                    self.layers[i].set_valid_mask(self.past_mask)
                elif attn_mask_cfg == "mask_local":
                    self.layers[i].set_valid_mask(self.local_mask)
                else:
                    raise ValueError("Unknown attention mask specification '{}'".format(attn_mask_cfg))

    def show_masks(self):
        import matplotlib.pyplot as plt

        nlayers = len(self.layers)
        fig, axes = plt.subplots(nrows=1, ncols=nlayers, sharey=True)

        for i in range(nlayers):
            layer = self.layers[i]
            mask = layer.attn.block[1].valid_mask
            if mask is not None:
                mask = mask.to("cpu").float().numpy()
                ax = axes[i]
                ax.imshow(mask, cmap='Blues', interpolation='none', aspect='equal', vmin=0, vmax=1)

        plt.tight_layout()

        return fig

    def forward(self, x, mask):  # x is of shape N x S x C, mask is N x S (1 --> keep, 0 --> mask)
        attn_weights = tuple()
        for layer in self.layers:
            out = layer(x, mask)
            x = out[FEATURES]
            attn_weights += (out[ATTENTION_WEIGHTS], )

        logits = None
        if self.head is not None:
            logits, _ = self.head((x, mask))
            if self.multi_task is True:
                logits = torch.split(logits, self.num_class, dim=-1)  # --> tuple of predictions

        return {
            FEATURES: x,
            LOGITS: logits,
            ATTENTION_WEIGHTS: attn_weights
        }


class MyModel(nn.Module, ModelTemplate):  # TUNeS
    def __init__(self, d_in, num_class, causal_model,
                 down_up_cfg, conv_block_cfg, attn_cfg, mlp_cfg, conv_attn_block_cfg, transformer_cfg, max_seq_len,
                 d_model=64, transformer_add_tokens=True,
                 down_blocks=[2, 2, 2], up_blocks=[2, 2, 2], temporal_scales=[2, 2, 2], channel_scales=[2, 2, 2],
                 up_kernels=[], up_dilations=[], skip_connections=True, weighted_fusion=False, fusion_weight_init=1.0,
                 attention_idx=-1, 
                 # Multi-token input support
                 input_format="2d", num_tokens_per_timestep=1, token_aggregation_type="linear", 
                 token_aggregation_hidden_dim=None):
        super().__init__()
        assert (len(down_blocks) == len(up_blocks))
        assert (len(temporal_scales) == len(down_blocks) and len(channel_scales) == len(down_blocks))
        for list_ in [up_kernels, up_dilations]:
            assert (len(list_) == 0 or len(list_) == len(up_blocks))

        # Multi-token input configuration
        self.input_format = input_format
        self.num_tokens_per_timestep = num_tokens_per_timestep
        self.token_aggregation_type = token_aggregation_type
        
        # Initialize token aggregator for 3D input
        if input_format == "3d":
            assert num_tokens_per_timestep > 1, "For 3D input, num_tokens_per_timestep must be > 1"
            
            if token_aggregation_hidden_dim is None:
                token_aggregation_hidden_dim = d_model
                
            if token_aggregation_type in ["linear", "mlp", "attention"]:
                self.token_aggregator = TokenAggregator(
                    input_dim=d_in,
                    output_dim=d_in,  # Keep same dimension, will be projected by in_proj later
                    num_tokens=num_tokens_per_timestep,
                    aggregation_type=token_aggregation_type,
                    hidden_dim=token_aggregation_hidden_dim,
                    init_method=conv_block_cfg['init_method'],
                    dropout=conv_block_cfg.get('dropout', 0.0)
                )
            else:
                # Use simple aggregation (cls, mean, max)
                self.token_aggregator = SimpleTokenAggregator(
                    aggregation_type=token_aggregation_type
                )
        else:
            self.token_aggregator = None

        bottleneck_idx = len(down_blocks)
        if attention_idx < 0:
            attention_idx = bottleneck_idx
        # attention_idx specifies in which stage we introduce attention blocks
        # we suggest using attention_idx = 3 --> use attention at bottleneck
        if attention_idx != bottleneck_idx:
            assert (0 <= attention_idx <= 2 * bottleneck_idx)
            assert (conv_attn_block_cfg["skip_attn"] is False)
            attention_cfg = {
                'attn_cfg': copy.deepcopy(attn_cfg),
                'mlp_cfg': copy.deepcopy(mlp_cfg),
                'conv_attn_block_cfg': copy.deepcopy(conv_attn_block_cfg),
            }
            for key in transformer_cfg:
                attention_cfg[key] = transformer_cfg[key]

        self.causal = causal_model
        self.use_skip_connections = skip_connections

        self.channels_last = True
        self.temporal_scales = [1]
        scale = 1
        if len(temporal_scales) > 0:
            for i in range(len(temporal_scales)):
                scale *= temporal_scales[i]
                self.temporal_scales.append(scale)

        if d_in < 0:
            self.in_proj = nn.Identity()  # skip initial projection
        else:
            self.in_proj = MyLinear(d_in, dim_out=d_model, init_method=conv_block_cfg['init_method'], followed_by_relu=False)

        stage_ctr = 0
        dim_ = d_model
        dims_ = [dim_]
        self.down_path = nn.ModuleList()
        i = 0
        while i < len(down_blocks):
            if stage_ctr == attention_idx:  # use convolutional attention stack at this stage
                attention_cfg['max_seq_len'] = int(math.ceil(max_seq_len / self.temporal_scales[i]) * 2)
                self.down_path.append(ConvStageDown(
                    dim_, down_blocks[i], temporal_scales[i], channel_scales[i], self.causal,
                    copy.deepcopy(conv_block_cfg), copy.deepcopy(down_up_cfg), add_attention_cfg=attention_cfg
                ))
            else:
                self.down_path.append(ConvStageDown(
                    dim_, down_blocks[i], temporal_scales[i], channel_scales[i], self.causal,
                    copy.deepcopy(conv_block_cfg), copy.deepcopy(down_up_cfg)
                ))
            dim_ = dim_ * channel_scales[i]
            dims_.append(dim_)
            i += 1
            stage_ctr += 1

        if attention_idx != bottleneck_idx:
            conv_attn_block_cfg_ = copy.deepcopy(conv_attn_block_cfg)
            assert (conv_attn_block_cfg_["skip_conv"] is False)
            conv_attn_block_cfg_["skip_attn"] = True  # skip attention in bottleneck
            transformer_cfg_ = copy.deepcopy(transformer_cfg)
            transformer_cfg_["attention_masks_cfg"] = None
            transformer_cfg_["nlayers"] = down_blocks[-1]  # use same number of convolutional blocks as in other stages
        else:
            conv_attn_block_cfg_ = conv_attn_block_cfg
            transformer_cfg_ = transformer_cfg
        self.bottleneck = ConvAttentionStack(
            dim_, conv_block_cfg, attn_cfg, mlp_cfg, conv_attn_block_cfg_,
            max_seq_len=int(math.ceil(max_seq_len / self.temporal_scales[-1]) * 2),
            num_class=num_class, causal=self.causal, **transformer_cfg_)
        # transformer_cfg:
        # nlayers=6, dilation_mode='exponential', dilation_factors=None,
        # num_cls_heads=1, classifier_dropout=0., normalize=True, attention_masks_cfg=None, local_attn_window_size=7
        # use_sinusoidal_pe=False
        stage_ctr += 1

        token_embeddings = dict()
        token_embeddings['mask'] = nn.Parameter(torch.empty(dim_, requires_grad=True))
        token_embeddings['end'] = nn.Parameter(torch.empty(dim_, requires_grad=True))
        if transformer_add_tokens:
            token_embeddings['start'] = nn.Parameter(torch.empty(dim_, requires_grad=True))
        self.token_embeddings = nn.ParameterDict(token_embeddings)
        for token in self.token_embeddings:
            nn.init.normal_(self.token_embeddings[token], mean=0.0, std=0.02)

        self.up_path = nn.ModuleList()
        for i in range(len(up_blocks)):
            if len(up_kernels) > 0:
                conv_block_cfg['kernel_size'] = up_kernels[i]
            if len(up_dilations) > 0:
                conv_block_cfg['dilation'] = up_dilations[i]
            if stage_ctr == attention_idx:  # use convolutional attention stack at this stage
                attention_cfg['max_seq_len'] = int(math.ceil(max_seq_len / self.temporal_scales[-(i + 2)]) * 2)
                self.up_path.append(ConvStageUp(
                    dim_ // channel_scales[-(i + 1)], up_blocks[i], temporal_scales[-(i + 1)], channel_scales[-(i + 1)],
                    self.causal, copy.deepcopy(conv_block_cfg), copy.deepcopy(down_up_cfg),
                    num_class=num_class, weighted_fusion=weighted_fusion, fusion_weight_initial_value=fusion_weight_init,
                    add_attention_cfg=attention_cfg)
                )
            else:
                self.up_path.append(ConvStageUp(
                    dim_ // channel_scales[-(i + 1)], up_blocks[i], temporal_scales[-(i + 1)], channel_scales[-(i + 1)],
                    self.causal, copy.deepcopy(conv_block_cfg), copy.deepcopy(down_up_cfg),
                    num_class=num_class, weighted_fusion=weighted_fusion, fusion_weight_initial_value=fusion_weight_init)
                )
            dim_ = dim_ // channel_scales[-(i + 1)]
            stage_ctr += 1

    def extra_repr(self) -> str:
        return 'skip_connections={}'.format(self.use_skip_connections)

    def get_temporal_scales(self):
        return self.temporal_scales

    def setup(self):  # call this after model has been sent to GPU
        for name, module in self.named_modules():
            if isinstance(module, ConvAttentionStack):
                module.setup()

    def forward(self, x, mask, feature_mask=None):
        # Handle multi-token input format
        if self.input_format == "3d":
            # x is of shape N x S x num_tokens x C, mask is N x S (1 --> keep, 0 --> mask)
            N, S, num_tokens, C = x.shape
            assert num_tokens == self.num_tokens_per_timestep, \
                f"Expected {self.num_tokens_per_timestep} tokens per timestep, got {num_tokens}"
            
            # Apply token aggregation to convert (N, S, num_tokens, C) -> (N, S, C)
            x, mask = self.token_aggregator(x, mask)
            N, S, C = x.shape
        else:
            # x is of shape N x S x C, mask is N x S (1 --> keep, 0 --> mask), feature_mask is N x S (1 --> mask)
            N, S, C = x.shape
            
        x, mask = self.in_proj((x, mask))

        intermediates = []
        i = 0
        for down in self.down_path:
            (x, mask), interm = down(x, mask)
            intermediates.append(interm)
            i += 1

        # preparations for bottleneck stage
        if feature_mask is not None:
            x[feature_mask] = self.token_embeddings["mask"]
        x[torch.logical_not(mask)] = self.token_embeddings['end']  # padded tokens
        if 'start' in self.token_embeddings:  # add start/ end tokens
            x = torch.cat([
                self.token_embeddings['start'].expand(N, 1, -1), x, self.token_embeddings['end'].expand(N, 1, -1)
            ], dim=1)
            mask_ = torch.cat([torch.ones((N, 2), dtype=mask.dtype, device=mask.device), mask], dim=1)
        else:
            mask_ = mask

        bottleneck_out = self.bottleneck(x, mask_)
        x_ = bottleneck_out[FEATURES]
        if self.token_embeddings is not None and 'start' in self.token_embeddings:  # remove start/ end tokens
            x = x_[:, 1:-1, :]

        outputs = tuple()
        for up in self.up_path:
            interm, mask_high_res = intermediates.pop()
            up_out = up(x, mask, mask_high_res, inp=interm if self.use_skip_connections is True else None)
            x, mask = up_out[FEATURES]
            outputs += (up_out[LOGITS], )
        assert (len(intermediates) == 0)

        result = {
            LOGITS: outputs,
            FEATURES: x,
            BOTTLENECK_LOGITS: bottleneck_out[LOGITS],  # --> start/ end tokens not removed yet
            ATTENTION_WEIGHTS: bottleneck_out[ATTENTION_WEIGHTS]
        }

        return result

    def parse_batch(self, batch, device_gpu, train=True, get_target=True, deep_supervision=True):
        results = {}

        input = batch[FEATURE_SEQ]
        N, S, C = input.shape
        results["batch_size"] = N
        input = input.to(device_gpu)

        in_valid_mask = torch.logical_not(batch[PADDING_MASK])
        if train is False:
            results["in_valid_mask_cpu"] = in_valid_mask.detach()
        nelems = in_valid_mask.sum().item()
        in_valid_mask = in_valid_mask.to(device_gpu)

        bottleneck_scale = self.temporal_scales[-1]
        token_mask_in, token_mask_out = None, None
        if train is True and TOKEN_MASK in batch:
            token_mask_in = batch[TOKEN_MASK][:, ::bottleneck_scale]
            if 'start' in self.token_embeddings:
                token_mask_out = torch.cat([torch.zeros((N, 1)), token_mask_in, torch.zeros((N, 1))], dim=1)
            else:
                token_mask_out = token_mask_in
            results["bottleneck_mask_cpu"] = token_mask_out.detach()
            token_mask_out = token_mask_out.to(device_gpu)
            token_mask_in = token_mask_in.to(device_gpu)

            masked_ratios = batch[MASKED_RATIO]
            nonzero_masked_ratios = masked_ratios[masked_ratios > 0]
            results[MASKED_RATIO] = torch.sum(nonzero_masked_ratios), torch.numel(nonzero_masked_ratios)

        if get_target is True:
            target = batch[LABEL_SEQ]
            if train is False:
                results["target_cpu"] = target.detach()
            target = target.to(device_gpu)

            if deep_supervision is True:
                multi_scale_targets = dict()
                multi_scale_valid_masks = dict()
                for scale in batch[LABEL_SEQ_MULTISCALE]:
                    if scale != bottleneck_scale:
                        multi_scale_targets[scale] = batch[LABEL_SEQ_MULTISCALE][scale][0].to(device_gpu)
                        multi_scale_valid_masks[scale] = \
                            torch.logical_not(batch[LABEL_SEQ_MULTISCALE][scale][1].to(device_gpu))
            else:
                multi_scale_targets, multi_scale_valid_masks = None, None

            bottleneck_target = batch[LABELS_BOTTLENECK][0]
            bottleneck_len = S // bottleneck_scale
            if 'start' in self.token_embeddings:
                bottleneck_len = bottleneck_len + 2
            bottleneck_valid_mask = torch.logical_not(batch[LABELS_BOTTLENECK][1])
            assert (bottleneck_target.shape[1] == bottleneck_len)
            assert (bottleneck_valid_mask.shape[1] == bottleneck_len)
            results["bottleneck_target_cpu"] = bottleneck_target.detach() * bottleneck_valid_mask.detach().unsqueeze(-1)
            bottleneck_target = bottleneck_target.to(device_gpu)
            bottleneck_valid_mask = bottleneck_valid_mask.to(device_gpu)

        else:
            target = None
            multi_scale_targets, multi_scale_valid_masks = None, None
            bottleneck_target, bottleneck_valid_mask = None, None

        return input, in_valid_mask, token_mask_in, token_mask_out, \
               target, multi_scale_targets, multi_scale_valid_masks, bottleneck_target, bottleneck_valid_mask, \
               nelems, N, S, results

    def infer_batch(self, batch, device_gpu, return_logits=False):
        input, in_valid_mask, _, _, _, _, _, _, _, _, _, _, results = \
            self.parse_batch(batch, device_gpu, train=False, get_target=False, deep_supervision=False)

        out = self.forward(input, in_valid_mask, None)

        logits = out[LOGITS][-1]
        if return_logits:
            results["logits_gpu"] = logits.detach()
        else:
            _, predicted = torch.max(torch.nn.Softmax(dim=-1)(logits), dim=-1)
            results["predicted_gpu"] = predicted.detach()

        return results


class MyTrainer(TrainerTemplate):
    def __init__(self, deep_supervision, phase_recognition_loss, smooth_logits_loss, multilabel_loss,
                 phase_recognition_factor, smooth_logits_factor, smoothing_loss_max_scale,
                 multi_label_factor, bottleneck_unmasked_factor, bottleneck_masked_factor, model_scales, mask_prob,
                 device_cpu, device_gpu, nplot=4, dataset=Cholec80):
        super().__init__(deep_supervision, phase_recognition_loss, smooth_logits_loss,
                         phase_recognition_factor, smooth_logits_factor, device_cpu, device_gpu, nplot, dataset)

        self.loss_functions["multi_label"] = multilabel_loss

        self.loss_factors["multi_label"] = multi_label_factor
        self.loss_factors["bottleneck_unmasked"] = bottleneck_unmasked_factor
        self.loss_factors["bottleneck_masked"] = bottleneck_masked_factor

        for j in range(len(model_scales) - 1):
            self.loss_keys.append("loss_multi_label_s-{}".format(model_scales[j]))
        self.loss_keys.append("loss_bottleneck_unmasked")
        if mask_prob > 0:
            self.loss_keys.append("loss_bottleneck_masked")

        self.smoothing_loss_max_scale = smoothing_loss_max_scale

    def process_batch(self, model, batch, train=True):
        input, in_valid_mask, token_mask_in, token_mask_out, \
        target, multi_scale_targets, multi_scale_valid_masks, bottleneck_target, bottleneck_valid_mask, \
        nelems, N, S, results = model.parse_batch(
            batch, self.device_gpu, train, get_target=True, deep_supervision=self.deep_supervision
        )

        out = model(input, in_valid_mask, token_mask_in)

        self.process_(
            out, in_valid_mask, token_mask_in, token_mask_out,
            target, multi_scale_targets, multi_scale_valid_masks, bottleneck_target, bottleneck_valid_mask,
            nelems, N, results, train
        )

        return results

    def process_(self, out, in_valid_mask, token_mask_in, token_mask_out,
                 target, multi_scale_targets, multi_scale_valid_masks, bottleneck_target, bottleneck_valid_mask,
                 nelems, N, results, train):
        if ATTENTION_WEIGHTS in out:
            results["attn_weights_gpu"] = [
                None if attn_weights is None else attn_weights.detach() for attn_weights in out[ATTENTION_WEIGHTS]
            ]

        total_loss, loss_factor_sum = self.compute_multiscale_phase_recognition_loss(
            out[LOGITS], in_valid_mask, target, multi_scale_targets, multi_scale_valid_masks, nelems, N, train, results
        )

        b_loss, b_loss_factor_sum = self.compute_bottleneck_reconstruction_loss(
            out[BOTTLENECK_LOGITS], bottleneck_valid_mask, bottleneck_target, token_mask_in, token_mask_out, results
        )
        total_loss = total_loss + b_loss
        loss_factor_sum = loss_factor_sum + b_loss_factor_sum

        total_loss = total_loss / loss_factor_sum  # normalize loss factors

        if train is True:
            results["total_loss"] = total_loss

    def compute_multiscale_phase_recognition_loss(
            self, ms_logits, valid_mask, target, multi_scale_targets, multi_scale_valid_masks, nelems, N, train, results
    ):
        total_loss = 0
        loss_factor_sum = 0

        nlogits = len(ms_logits)
        for i in range(0 if self.deep_supervision else (nlogits - 1), nlogits):
            logits = ms_logits[i]

            logits_chF = logits.permute(0, 2, 1)  # N x S x C --> N x C x S; channels first format
            logits_chL = logits
            del logits

            scale = int(target.shape[1] // logits_chF.shape[-1])
            if scale == 1:
                loss_ = self.loss_functions["phase_recognition"](logits_chF, target) \
                        * self.loss_factors["phase_recognition"]
                total_loss = total_loss + loss_
                loss_factor_sum += self.loss_factors["phase_recognition"]
            else:
                target_scaled = multi_scale_targets[scale]
                valid_mask_scaled = multi_scale_valid_masks[scale]

                loss_ = self.loss_functions["multi_label"](logits_chL, target_scaled.float(), valid_mask_scaled) \
                        * self.loss_factors["multi_label"]
                total_loss = total_loss + loss_
                loss_factor_sum += self.loss_factors["multi_label"]
                results["loss_multi_label_s-{}".format(scale)] = (loss_.item(), valid_mask_scaled.sum().item())

            if i == (nlogits - 1):
                assert (scale == 1)
                results["loss_phase_recognition"] = (loss_.item(), nelems)

            if self.loss_factors["smooth_logits"] > 0 and \
                    (self.smoothing_loss_max_scale < 0 or scale <= self.smoothing_loss_max_scale):
                if scale == 1:
                    valid_mask_scaled = valid_mask
                else:
                    valid_mask_scaled = multi_scale_valid_masks[scale]
                loss_ = self.loss_functions["smooth_logits"](logits_chL, valid_mask_scaled) \
                        * self.loss_factors["smooth_logits"]
                total_loss = total_loss + loss_
                loss_factor_sum += self.loss_factors["smooth_logits"]
                if i == (nlogits - 1):
                    results["loss_smooth"] = (loss_.item(), nelems - N)

        # logits_chF = out[LOGITS][-1].permute(0, 2, 1)
        self.get_prediction(logits_chF, valid_mask, target, results, nelems, train)

        return total_loss, loss_factor_sum

    def compute_bottleneck_reconstruction_loss(
            self, logits, valid_mask, target, token_mask_in, token_mask_out, results
    ):
        # logits shape: # N x bottleneck_len x C

        total_loss = 0
        loss_factor_sum = 0

        # loss on unmasked positions
        if token_mask_in is not None:
            loss_mask = torch.logical_and(torch.logical_not(token_mask_out), valid_mask)  # unmasked & valid positions
        else:
            loss_mask = valid_mask
        loss_ = self.loss_functions["multi_label"](logits, target.float(), loss_mask) \
                * self.loss_factors["bottleneck_unmasked"]
        total_loss = total_loss + loss_
        loss_factor_sum += self.loss_factors["bottleneck_unmasked"]
        results["loss_bottleneck_unmasked"] = (loss_.item(), loss_mask.sum().item())

        # loss on masked positions
        if token_mask_in is not None:
            loss_mask = torch.logical_and(token_mask_out, valid_mask)  # masked & valid positions
            if loss_mask.sum() > 0:
                loss_ = self.loss_functions["multi_label"](logits, target.float(), loss_mask) \
                        * self.loss_factors["bottleneck_masked"]
                total_loss = total_loss + loss_
                loss_factor_sum += self.loss_factors["bottleneck_masked"]
                results["loss_bottleneck_masked"] = (loss_.item(), loss_mask.sum().item())

        with torch.no_grad():
            probs = torch.sigmoid(logits) * valid_mask.unsqueeze(-1)
            results["bottleneck_probs_gpu"] = probs.detach()

        return total_loss, loss_factor_sum

    def reset(self, train=False):
        if train is True:
            self.predictions = []
        else:
            self.predictions = {}

    def update_predictions(self, results, train=True):
        if train is True:
            if len(self.predictions) < self.nplot:
                bottleneck_out = results["bottleneck_probs_gpu"].to(self.device_cpu).numpy()
                bottleneck_target = results["bottleneck_target_cpu"].numpy()
                if "bottleneck_mask_cpu" in results:
                    bottleneck_mask = results["bottleneck_mask_cpu"].float().numpy()
                else:
                    bottleneck_mask = None
                attn_weights = [None if attn is None else attn.to(self.device_cpu).numpy() for attn in results["attn_weights_gpu"]]

                for i in range(results["batch_size"]):
                    if len(self.predictions) < self.nplot:
                        self.predictions.append({
                            "bottleneck_probs": bottleneck_out[i],
                            "bottleneck_target": bottleneck_target[i],
                            "bottleneck_mask": None if bottleneck_mask is None else bottleneck_mask[i],
                            "attn_weights": [None if attn is None else attn[i] for attn in attn_weights]
                        })
        else:
            predicted = results["predicted_gpu"].to(self.device_cpu).numpy()
            target = results["target_cpu"].numpy()
            valid_mask = results["in_valid_mask_cpu"].numpy()
            bottleneck_out = results["bottleneck_probs_gpu"].to(self.device_cpu).numpy()
            attn_weights = [None if attn is None else attn.to(self.device_cpu).numpy() for attn in results["attn_weights_gpu"]]

            for i in range(results["batch_size"]):
                P = predicted[i, valid_mask[i, :]]
                Y = target[i, valid_mask[i, :]]

                metrics = calculate_metrics(Y, P, self.dataset.phase_labels)
                key_ = ("{:.4f}".format(metrics["accuracy"]), "{:.4f}".format(metrics["macro_jaccard"]))
                self.predictions[key_] = {
                    'predicted': P,
                    'target': Y,
                    'bottleneck_probs': bottleneck_out[i],
                    'attn_weights': [None if attn is None else attn[i] for attn in attn_weights]
                }

    def visualize_outputs(self, epoch, logger, logger_prefix=None, train=True):
        if train is True:
            logger_prefix = "train"
            logger.add_figure("{}/bottleneck_probs".format(logger_prefix),
                              visualize_masked_bottleneck_outputs(self.predictions, self.dataset.num_phases), epoch)
            logger.add_figure("{}/attention".format(logger_prefix),
                              visualize_attention_weights(self.predictions), epoch)
        else:
            assert (logger_prefix is not None)
            achieved_metrics = sorted(sorted(list(self.predictions.keys()), key=lambda t: t[0]), key=lambda t: t[1])
            if len(achieved_metrics) > 0:
                # show results with lowest performance
                to_plot = [self.predictions[key_] for key_ in achieved_metrics[:self.nplot]]
                logger.add_figure("{}/predictions".format(logger_prefix), visualize_predictions(to_plot, self.dataset.num_phases), epoch)
                logger.add_figure("{}/bottleneck_probs".format(logger_prefix),
                                  visualize_bottleneck_outputs(to_plot, self.dataset.num_phases), epoch)
                logger.add_figure("{}/attention".format(logger_prefix), visualize_attention_weights(to_plot), epoch)
