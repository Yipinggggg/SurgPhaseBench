import math
import torch
from torch import nn

from .modeling.backbones import ConvTransformerBackbone
from .modeling.necks import FPN1D
from .modeling.blocks import MaskedConv1D, LayerNorm

from ...template import ModelTemplate, TrainerTemplate
from .....utils import LOGITS, FEATURE_SEQ, LABEL_SEQ, LABEL_SEQ_MULTISCALE, PADDING_MASK, Cholec80


class ClsHead(nn.Module):
    """
    1D Conv heads for classification
    """
    def __init__(
        self,
        input_dim,
        feat_dim,
        num_classes,
        prior_prob=0.01,
        num_layers=3,
        kernel_size=3,
        act_layer=nn.ReLU,
        with_ln=False,
        empty_cls = []
    ):
        super().__init__()
        self.act = act_layer()

        # build the head
        self.head = nn.ModuleList()
        self.norm = nn.ModuleList()
        for idx in range(num_layers-1):
            if idx == 0:
                in_dim = input_dim
                out_dim = feat_dim
            else:
                in_dim = feat_dim
                out_dim = feat_dim
            self.head.append(
                MaskedConv1D(
                    in_dim, out_dim, kernel_size,
                    stride=1,
                    padding=kernel_size//2,
                    bias=(not with_ln)
                )
            )
            if with_ln:
                self.norm.append(LayerNorm(out_dim))
            else:
                self.norm.append(nn.Identity())

        # classifier
        self.cls_head = MaskedConv1D(
                feat_dim, num_classes, kernel_size,
                stride=1, padding=kernel_size//2
            )

        # use prior in model initialization to improve stability
        # this will overwrite other weight init
        if prior_prob > 0:
            bias_value = -(math.log((1 - prior_prob) / prior_prob))
            torch.nn.init.constant_(self.cls_head.conv.bias, bias_value)

        # a quick fix to empty categories:
        # the weights assocaited with these categories will remain unchanged
        # we set their bias to a large negative value to prevent their outputs
        if len(empty_cls) > 0:
            bias_value = -(math.log((1 - 1e-6) / 1e-6))
            for idx in empty_cls:
                torch.nn.init.constant_(self.cls_head.conv.bias[idx], bias_value)

    def forward(self, fpn_feats, fpn_masks):
        assert len(fpn_feats) == len(fpn_masks)

        # apply the classifier for each pyramid level
        out_logits = tuple()
        for _, (cur_feat, cur_mask) in enumerate(zip(fpn_feats, fpn_masks)):
            cur_out = cur_feat
            for idx in range(len(self.head)):
                cur_out, _ = self.head[idx](cur_out, cur_mask)
                cur_out = self.act(self.norm[idx](cur_out))
            cur_logits, _ = self.cls_head(cur_out, cur_mask)
            out_logits += (cur_logits, )

        # fpn_masks remains the same
        return out_logits


class ActionFormer(nn.Module, ModelTemplate):
    """
        Transformer based model for single stage action localization
    """
    def __init__(
        self,
        input_dim,                # input feat dim
        max_seq_len,              # max sequence length (used for training)
        num_classes,              # number of action classes
        backbone_arch=(2, 2, 5),  # a tuple defines #layers in embed / stem / branch
        scale_factor=2,           # scale factor between branch layers
        n_head=4,                 # number of heads for self-attention in transformer
        n_mha_win_size=9,         # window size for self attention; -1 to use full seq
        embd_kernel_size=3,       # kernel size of the embedding network
        d_model=64,
        embd_with_ln=True,        # attach layernorm to embedding network
        fpn_with_ln=True,         # if to apply layer norm at the end of fpn
        fpn_start_level=0,        # start level of fpn
        use_abs_pe=False,         # if to use abs position encoding
        use_rel_pe=False,         # if to use rel position encoding
        train_dropout=0.0,
        train_droppath=0.1,
        drop_layer_norm=False,
        use_linear_in_projection=False,
        conv_classifier=False
    ):
        super().__init__()
         # re-distribute params to backbone / neck / head
        self.fpn_strides = [scale_factor**i for i in range(
            fpn_start_level, backbone_arch[-1]+1
        )]
        self.scale_factor = scale_factor
        # #classes = num_classes + 1 (background) with last category as background
        # e.g., num_classes = 10 -> 0, 1, ..., 9 as actions, 10 as background
        self.num_classes = num_classes

        # check the feature pyramid and local attention window size
        self.max_seq_len = max_seq_len
        if isinstance(n_mha_win_size, int):
            self.mha_win_size = [n_mha_win_size]*(1 + backbone_arch[-1])
        else:
            assert len(n_mha_win_size) == (1 + backbone_arch[-1])
            self.mha_win_size = n_mha_win_size
        max_div_factor = 1
        for l, (s, w) in enumerate(zip(self.fpn_strides, self.mha_win_size)):
            stride = s * (w // 2) * 2 if w > 1 else s
            assert max_seq_len % stride == 0, "max_seq_len must be divisible by fpn stride and window size"
            if max_div_factor < stride:
                max_div_factor = stride
        self.max_div_factor = max_div_factor

        # we will need a better way to dispatch the params to backbones / necks
        # backbone network: conv + transformer
        self.backbone = ConvTransformerBackbone(
            **{
                'n_in' : input_dim,
                'n_embd' : d_model,  # output feat channel of the embedding network
                'n_head': n_head,
                'n_embd_ks': embd_kernel_size,
                'max_len': max_seq_len,
                'arch' : backbone_arch,
                'mha_win_size': self.mha_win_size,
                'scale_factor' : scale_factor,
                'with_ln' : embd_with_ln,
                'attn_pdrop' : 0.0,
                'proj_pdrop' : train_dropout,
                'path_pdrop' : train_droppath,
                'use_abs_pe' : use_abs_pe,
                'use_rel_pe' : use_rel_pe,
                'use_linear_in_projection' : use_linear_in_projection,
                'drop_ln' : drop_layer_norm
            }
        )

        # fpn network: convs
        self.neck = FPN1D(
            **{
                'in_channels' : [d_model] * (backbone_arch[-1] + 1),
                'out_channel' : d_model,  # feature dim on FPN
                'scale_factor' : scale_factor,
                'start_level' : fpn_start_level,
                'with_ln' : fpn_with_ln
            }
        )

        self.cls_head = ClsHead(
            d_model, d_model, self.num_classes,
            kernel_size=3 if conv_classifier else 1,
            prior_prob=-1,
            with_ln=(not drop_layer_norm),
            num_layers=3 if conv_classifier else 1,
            empty_cls=[]
        )

        self.causal_model = False
        self.channels_last = False

    def get_temporal_scales(self):
        return self.fpn_strides

    def forward(self, x, mask):  # x is of shape N x C x S, mask is N x S (1 --> keep, 0 --> mask)

        # forward the network (backbone -> neck -> heads)
        feats, masks = self.backbone(x, mask.unsqueeze(1))
        fpn_feats, fpn_masks = self.neck(feats, masks)  # feature pyramid: fine --> coarse

        out_cls_logits = self.cls_head(fpn_feats, fpn_masks)

        return {
            LOGITS: tuple(reversed(out_cls_logits))  # updated order: coarse --> fine
        }

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

        if get_target is True:
            target = batch[LABEL_SEQ]
            if train is False:
                results["target_cpu"] = target.detach()
            target = target.to(device_gpu)

            if deep_supervision is True:
                multi_scale_targets = dict()
                multi_scale_valid_masks = dict()
                for scale in batch[LABEL_SEQ_MULTISCALE]:
                    multi_scale_targets[scale] = batch[LABEL_SEQ_MULTISCALE][scale][0].to(device_gpu)
                    multi_scale_valid_masks[scale] = \
                        torch.logical_not(batch[LABEL_SEQ_MULTISCALE][scale][1].to(device_gpu))
            else:
                multi_scale_targets, multi_scale_valid_masks = None, None
        else:
            target = None
            multi_scale_targets, multi_scale_valid_masks = None, None

        return input, in_valid_mask, target, multi_scale_targets, multi_scale_valid_masks, \
               nelems, N, results

    def infer_batch(self, batch, device_gpu, return_logits=False):
        input, in_valid_mask, _, _, _, _, _, results = \
            self.parse_batch(batch, device_gpu, train=False, get_target=False, deep_supervision=False)

        input = input.permute(0, 2, 1)  # N x S x C --> N x C x S
        out = self.forward(input, in_valid_mask)

        logits = out[LOGITS][-1]
        if return_logits:
            results["logits_gpu"] = logits.permute(0, 2, 1).detach()
        else:
            _, predicted = torch.max(torch.nn.Softmax(dim=1)(logits), dim=1)
            results["predicted_gpu"] = predicted.detach()

        return results


class TrainerActionFormer(TrainerTemplate):
    def __init__(self, deep_supervision, phase_recognition_loss, smooth_logits_loss, multilabel_loss,
                 phase_recognition_factor, smooth_logits_factor, smoothing_loss_max_scale,
                 multi_label_factor, model_scales, device_cpu, device_gpu, nplot=4, dataset=Cholec80):
        super().__init__(deep_supervision, phase_recognition_loss, smooth_logits_loss,
                         phase_recognition_factor, smooth_logits_factor, device_cpu, device_gpu, nplot, dataset)

        self.loss_functions["multi_label"] = multilabel_loss
        self.loss_factors["multi_label"] = multi_label_factor
        for j in range(len(model_scales)):
            self.loss_keys.append("loss_multi_label_s-{}".format(model_scales[j]))

        self.smoothing_loss_max_scale = smoothing_loss_max_scale

    def process_batch(self, model, batch, train=True):
        input, in_valid_mask, target, multi_scale_targets, multi_scale_valid_masks, nelems, N, results = \
            model.parse_batch(batch, self.device_gpu, train, get_target=True, deep_supervision=self.deep_supervision)
        channels_last = model.channels_last

        input = input.permute(0, 2, 1)  # N x S x C --> N x C x S
        out = model(input, in_valid_mask)

        total_loss, loss_factor_sum = self.compute_multiscale_phase_recognition_loss(
            out[LOGITS], in_valid_mask, target, multi_scale_targets, multi_scale_valid_masks, nelems, N, train, results,
            channels_last=channels_last
        )
        total_loss = total_loss / loss_factor_sum  # normalize loss factors

        if train is True:
            results["total_loss"] = total_loss
        return results

    # same as function in MyTrainer
    def compute_multiscale_phase_recognition_loss(
            self, ms_logits, valid_mask, target, multi_scale_targets, multi_scale_valid_masks, nelems, N, train, results,
            channels_last=True
    ):
        total_loss = 0
        loss_factor_sum = 0

        nlogits = len(ms_logits)
        for i in range(0 if self.deep_supervision else (nlogits - 1), nlogits):
            logits = ms_logits[i]

            if channels_last:
                logits_chF = logits.permute(0, 2, 1)  # N x S x C --> N x C x S; channels first format
                logits_chL = logits
            else:
                logits_chF = logits
                logits_chL = logits.permute(0, 2, 1)  # N x C x S --> N x S x C; channels last format
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

        self.get_prediction(logits_chF, valid_mask, target, results, nelems, train)

        return total_loss, loss_factor_sum
