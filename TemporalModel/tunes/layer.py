import math

import torch
from torch import nn as nn
from torch.nn import functional as F

from einops import rearrange
from timm.layers import trunc_normal_

from .activations import get_activation
from ..utils import FEATURES, ATTENTION_WEIGHTS


class MyActivation(nn.Module):
    def __init__(self, activation):
        super().__init__()

        self.activation_repr = activation
        self.activation = get_activation(activation)

    def forward(self, t):  # t = (x, mask); x is of shape N x S x C, mask is N x S (1 --> keep, 0 --> mask)
        x, mask = t
        x = self.activation(x)

        return x, mask

    def extra_repr(self) -> str:
        return 'activation={}'.format(self.activation_repr)


class MyLinear(nn.Module):
    def __init__(self, dim, dim_out=None, init_method='trunc_normal', followed_by_relu=False):
        super().__init__()

        if dim_out is None:
            dim_out = dim
        self.ff = nn.Linear(dim, dim_out)

        self._init_weights(init_method, followed_by_relu)
        self.init_method = init_method

    def _init_weights(self, init_method, followed_by_relu):
        if init_method == "trunc_normal":
            trunc_normal_(self.ff.weight, std=.02)
            nn.init.constant_(self.ff.bias, 0)
        elif init_method == "he_glorot":
            if followed_by_relu:
                nn.init.kaiming_normal_(self.ff.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(self.ff.bias, 0.)
            else:
                nn.init.xavier_uniform_(self.ff.weight)
                nn.init.constant_(self.ff.bias, 0.)
        elif init_method == "default":
            pass  # use pytorch standard initialization
        else:
            raise ValueError("Unknown init_method {}".format(init_method))

    def forward(self, t):  # t = (x, mask); x is of shape N x S x C, mask is N x S (1 --> keep, 0 --> mask)
        x, mask = t

        x = self.ff(x)

        return x, mask

    def extra_repr(self) -> str:
        return 'init={}'.format(self.init_method)


class MyConv1d(nn.Module):
    def __init__(self, dim_in, dim_out, causal=True, kernel_size=5, dilation=1, depthwise=False,
                 init_method='trunc_normal', followed_by_relu=False):
        super().__init__()
        self.causal = causal
        self.padding = dilation * ((kernel_size - 1) if self.causal else (kernel_size // 2))
        self.conv = nn.Conv1d(dim_in, dim_out, kernel_size, padding=self.padding, dilation=dilation,
                              groups=(dim_in if depthwise else 1), padding_mode='zeros')

        self._init_weights(init_method, followed_by_relu)
        self.init_method = init_method

    def _init_weights(self, init_method, followed_by_relu):
        if init_method == "trunc_normal":
            trunc_normal_(self.conv.weight, std=.02)
            nn.init.constant_(self.conv.bias, 0)
        elif init_method == "he_glorot":
            if followed_by_relu:
                nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(self.conv.bias, 0.)
            else:
                nn.init.xavier_uniform_(self.conv.weight)
                nn.init.constant_(self.conv.bias, 0.)
        elif init_method == "default":
            pass  # use pytorch standard initialization
        else:
            raise ValueError("Unknown init_method {}".format(init_method))

    def forward(self, t):  # t = (x, mask); x is of shape N x S x C, mask is N x S (1 --> keep, 0 --> mask)
        x, mask = t

        x = x * mask.unsqueeze(-1)
        x = x.permute(0, 2, 1)  # (N, S, C) -> (N, C, S)
        x = self.conv(x)
        if self.causal and self.padding > 0:
            x = x[:, :, :-self.padding]
        x = x.permute(0, 2, 1)  # (N, C, S) -> (N, S, C)

        return x, mask

    def extra_repr(self) -> str:
        return 'init={}, causal={}'.format(self.init_method, self.causal)


class MyDropout1d(nn.Module):
    """
    "unstructured" mode: randomly drop elements from tensor
    "channels" mode: randomly drop channels (set all elements in feature map to zero)
    "positions" mode: randomly mask positions in sequence (set corresponding feature vector to zero)
    """
    def __init__(self, p=0.5, mode="unstructured"):
        super().__init__()

        assert (mode in ["unstructured", "channels", "positions"])
        self.mode = mode
        if mode == "unstructured":
            self.dropout = nn.Dropout(p=p)
        else:
            self.dropout = nn.Dropout2d(p=p)

    def forward(self, t):  # t = (x, mask); x is of shape N x S x C, mask is N x S (1 --> keep, 0 --> mask)
        x, mask = t
        if self.mode == "unstructured":
            x = self.dropout(x)
        elif self.mode == "channels":
            x = self.dropout(x.permute(0, 2, 1).unsqueeze(-1)).squeeze(-1).permute(0, 2, 1)
        elif self.mode == "positions":
            x = self.dropout(x.unsqueeze(-1)).squeeze(-1)

        return x, mask

    def extra_repr(self) -> str:
        return 'mode={}'.format(self.mode)


class MyLayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.norm = nn.LayerNorm(dim)

    def forward(self, t):  # t = (x, mask); x is of shape N x S x C, mask is N x S (1 --> keep, 0 --> mask)
        x, mask = t

        x = x * mask.unsqueeze(-1)
        x = self.norm(x)

        return x, mask


class DownsamplingConv1d(nn.Module):
    def __init__(self, dim_in, dim_out, causal=True, factor=2, depthwise=False,
                 init_method='trunc_normal', followed_by_relu=False):
        super().__init__()
        if depthwise is True:
            assert (dim_out >= dim_in and dim_out % dim_in == 0)

        self.causal = causal
        self.padding = (factor - 1) if self.causal else 0
        self.conv = nn.Conv1d(dim_in, dim_out, factor, stride=factor, padding=self.padding,
                              groups=(dim_in if depthwise else 1), padding_mode='replicate')

        self._init_weights(init_method, followed_by_relu)
        self.init_method = init_method

    def _init_weights(self, init_method, followed_by_relu):
        if init_method == "trunc_normal":
            trunc_normal_(self.conv.weight, std=.02)
            nn.init.constant_(self.conv.bias, 0)
        elif init_method == "he_glorot":
            if followed_by_relu:
                nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(self.conv.bias, 0.)
            else:
                nn.init.xavier_uniform_(self.conv.weight)
                nn.init.constant_(self.conv.bias, 0.)
        elif init_method == "default":
            pass  # use pytorch standard initialization
        else:
            raise ValueError("Unknown init_method {}".format(init_method))

    def forward(self, t):  # t = (x, mask); x is of shape N x S x C, mask is N x S (1 --> keep, 0 --> mask)
        x, mask = t

        x = x * mask.unsqueeze(-1)
        x = x.permute(0, 2, 1)  # (N, S, C) -> (N, C, S)
        x = self.conv(x)
        if self.causal:
            x = x[:, :, :-1]
        x = x.permute(0, 2, 1)  # (N, C, S) -> (N, S, C)

        return x, mask

    def extra_repr(self) -> str:
        return 'init={}, causal={}'.format(self.init_method, self.causal)


class UpsamplingConv1d(nn.Module):
    def __init__(self, dim_in, dim_out, factor=2, depthwise=False, init_method='trunc_normal', followed_by_relu=False):
        super().__init__()
        if depthwise is True:
            assert (dim_out >= dim_in and dim_out % dim_in == 0)

        self.conv = nn.ConvTranspose1d(dim_in, dim_out, factor, stride=factor, padding=0,
                                       groups=(dim_in if depthwise else 1))

        self._init_weights(init_method, followed_by_relu)
        self.init_method = init_method

    def _init_weights(self, init_method, followed_by_relu):
        if init_method == "trunc_normal":
            trunc_normal_(self.conv.weight, std=.02)
            nn.init.constant_(self.conv.bias, 0)
        elif init_method == "he_glorot":
            if followed_by_relu:
                nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(self.conv.bias, 0.)
            else:
                nn.init.xavier_uniform_(self.conv.weight)
                nn.init.constant_(self.conv.bias, 0.)
        elif init_method == "default":
            pass  # use pytorch standard initialization
        else:
            raise ValueError("Unknown init_method {}".format(init_method))

    def forward(self, t):  # t = (x, mask); x is of shape N x S x C, mask is N x S (1 --> keep, 0 --> mask)
        x, mask = t

        x = x * mask.unsqueeze(-1)
        x = x.permute(0, 2, 1)  # (N, S, C) -> (N, C, S)
        x = self.conv(x)
        x = x.permute(0, 2, 1)  # (N, C, S) -> (N, S, C)

        return x, mask

    def extra_repr(self) -> str:
        return 'init={}'.format(self.init_method)


class ConvBlock(nn.Module):  # corresponds to convolutional block (w/o residual connection) in MS-TCN
    def __init__(self, dim, dilation, is_causal=True, kernel_size=3, activation="relu", dropout=0.5, depthwise=False,
                 init_method='trunc_normal', **unused):
        super().__init__()

        self.block = nn.Sequential(
            MyConv1d(dim, dim, causal=is_causal, kernel_size=kernel_size, dilation=dilation, depthwise=depthwise,
                     init_method=init_method, followed_by_relu=(activation in ["relu", "gelu"])),
            MyActivation(activation),
            MyLinear(dim, init_method=init_method, followed_by_relu=False),
            MyDropout1d(p=dropout, mode="unstructured")
        )

    def forward(self, t):
        return self.block(t)


class ResBlock(nn.Module):  # residual connection
    def __init__(self, dim, block=None, block_cfg=None, dropout=0., has_attn=False):
        super().__init__()

        if block is not None and isinstance(block, nn.Module):
            self.block = block
        else:
            if 'dim' in block_cfg:
                assert (block_cfg['dim'] == dim)
            self.block = ConvBlock(**block_cfg)

        self.dropout = MyDropout1d(p=dropout)
        self.has_attn = has_attn

    def forward(self, t):  # t = (x, mask); x is of shape N x S x C, mask is N x S (1 --> keep, 0 --> mask)
        x, mask = t
        residual = x

        if self.has_attn:
            out = self.block((x, mask))
            x, attn = out[FEATURES], out[ATTENTION_WEIGHTS]
        else:
            x, _ = self.block((x, mask))
        x, _ = self.dropout((x, mask))
        x = residual + x

        if self.has_attn:
            return {
                FEATURES: x,
                ATTENTION_WEIGHTS: attn
            }
        else:
            return x, mask


# see https://nlp.seas.harvard.edu/2018/04/03/attention.html
class PositionalEncoding(nn.Module):
    """Implement the PE function."""

    def __init__(self, d_model, dropout=0, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # 1 x S x C
        self.register_buffer('pe', pe, persistent=False)

    def forward(self, x):  # x of shape N x S x C
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


# ***** ***** Attention operations ***** *****
# Code based on
# https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/max_vit.py
# https://github.com/karpathy/minGPT/blob/master/mingpt/model.py
class MyAttention(nn.Module):
    def __init__(self, in_dim, attn_dim=None, nheads=4, relative_position_bias=False, causal=True, attn_dropout=0.,
                 max_len=100, proj_bias=False, init_method='trunc_normal'):
        super().__init__()

        if attn_dim is None:
            attn_dim = in_dim
        assert (attn_dim % nheads == 0)
        dim_head = attn_dim // nheads
        self.nheads = nheads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(in_dim, attn_dim * 3, bias=proj_bias)
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.out_proj = nn.Linear(attn_dim, in_dim, bias=proj_bias)

        if relative_position_bias is True:
            self.rel_pos_bias = nn.Embedding(max_len if causal is True else 2 * max_len - 1, self.nheads)
        else:
            self.rel_pos_bias = None
        self.rel_pos_indices = None  # to be set externally (--> same for all attention blocks)
        self.valid_mask = None  # to be set externally
        self.causal = causal

        self._init_weights(init_method)
        self.init_method = init_method

    def extra_repr(self) -> str:
        return 'init={}, nheads={}, masked={}, relative_position_bias={}'.format(
            self.init_method, self.nheads, self.causal, (self.rel_pos_bias is not None)
        )

    def set_rel_pos_indices(self, rel_pos_indices):
        self.rel_pos_indices = rel_pos_indices

    def set_valid_mask(self, valid_mask):
        self.valid_mask = valid_mask

    def _init_weights(self, init_method):
        if init_method == "trunc_normal":
            trunc_normal_(self.to_qkv.weight, std=.02)
            trunc_normal_(self.out_proj.weight, std=.02)
            if self.to_qkv.bias is not None:
                nn.init.constant_(self.to_qkv.bias, 0.)
            if self.out_proj.bias is not None:
                nn.init.constant_(self.out_proj.bias, 0.)
            if self.rel_pos_bias is not None:
                nn.init.normal_(self.rel_pos_bias.weight, mean=0.0, std=0.02)
        elif init_method == "he_glorot":
            nn.init.xavier_uniform_(self.to_qkv.weight)
            nn.init.xavier_uniform_(self.out_proj.weight)
            if self.to_qkv.bias is not None:
                nn.init.constant_(self.to_qkv.bias, 0.)
            if self.out_proj.bias is not None:
                nn.init.constant_(self.out_proj.bias, 0.)
            if self.rel_pos_bias is not None:
                nn.init.normal_(self.rel_pos_bias.weight, mean=0.0, std=0.02)
        elif init_method == "default":
            pass  # use pytorch standard initialization
        else:
            raise ValueError("Unknown init_method {}".format(init_method))

    def forward(self, t):  # t = (x, mask); x is of shape N x S x C, mask is N x S (1 --> keep, 0 --> mask)
        if self.causal is True:
            assert (self.valid_mask is not None)

        x, mask = t
        N, S, C = x.shape

        q, k, v = self.to_qkv(x).chunk(3, dim=-1)

        # split heads --> tensor shape N x nheads x S x (C // nheads)
        q, k, v = map(lambda tensor: rearrange(tensor, 'n s (h d ) -> n h s d', h=self.nheads), (q, k, v))

        attn = (q @ k.permute(0, 1, 3, 2)) * self.scale  # N x nheads x S x S
        # attn = einsum('n h i d, n h j d -> n h i j', q, k) * self.scale

        if self.rel_pos_bias is not None:
            bias = self.rel_pos_bias(self.rel_pos_indices[:S, :S])  # S x S x nheads
            attn = attn + rearrange(bias, 'i j h -> h i j')

        min_val = float('-inf')  # TODO: might be necessary to adjust min val in order to avoid NaN in softmax
        if self.valid_mask is not None:  # masking
            attn = attn.masked_fill(torch.logical_not(self.valid_mask[:S, :S]).view(1, 1, S, S), min_val)
        # prevent attending to padded positions
        attn = attn.masked_fill(torch.logical_not(mask).view(N, 1, 1, S), min_val)

        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)

        # set attention of padded queries to zero
        attn = attn.masked_fill(torch.logical_not(mask).view(N, 1, S, 1), 0)

        y = attn @ v  # N x nheads x S x (C // nheads)
        # y = einsum('n h i j, n h j d -> n h i d', attn, v)

        # merge heads --> tensor shape N x S x C
        # y = y.transpose(1, 2).contiguous().view(N, S, C)
        y = rearrange(y, 'n h s d -> n s (h d )',)

        y = self.out_proj(y)
        y = y * mask.unsqueeze(-1)

        return {
            FEATURES: y,
            ATTENTION_WEIGHTS: attn,
        }


def create_causal_mask(length):
    mask = torch.tril(torch.ones(length, length))  # True --> allowed to attend

    return mask.bool()


def create_local_mask(length, window=7):
    mask = torch.zeros(length, length)
    for i in range(length):
        mask[i, i] = 1
        for j in range(1, window + 1):
            if i - j >= 0:
                mask[i, i - j] = 1
            if i + j < length:
                mask[i, i + j] = 1

    return mask.bool()  # True --> allowed to attend


def create_relative_position_index(length, causal=True):
    pos = torch.arange(length)
    rel_pos = pos.unsqueeze(0) - pos.unsqueeze(-1)  # broad casting magic !
    rel_pos += length - 1

    if causal is True:
        mask = create_causal_mask(length)
        rel_pos = rel_pos * mask

    assert (torch.min(rel_pos) == 0)
    assert (torch.max(rel_pos) == (length - 1 if causal is True else 2 * length - 2))
    return rel_pos

