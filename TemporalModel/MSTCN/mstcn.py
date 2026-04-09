import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

# ***** MS-TCN adapted from https://github.com/yabufarha/ms-tcn, https://github.com/tobiascz/TeCNO *****

"""
Multi-Stage Temporal Convolutional Network (Original)

Training (Paper):
Standard Adam Optimizer
Cross-Entropy Loss & Truncated MSE Smoothing Loss (* 0.15) 

num_epochs: 50
learning_rate: 0.0005
sample_rate: 15 fps ?
(batch_size = 1)
(input_dim = 2 * 1024, num_classes)

num_stages: 4 
num_layers (per stage): 10 
feature_dim: 64
dropout: 0.5

***** ***** ***** ***** *****

TeCNO:

num_epochs: 25
learning_rate: 0.0005
batch_size = 1
input_dim: 2048 (ResNet-50)
num_stages: 2
num_layers: 9
feature_dim: 64

"""


def _build_activation(name: str) -> nn.Module:
    """Return a PyTorch activation module from a small string registry."""
    key = (name or "relu").lower()
    if key == "relu":
        return nn.ReLU()
    if key == "gelu":
        return nn.GELU()
    if key == "elu":
        return nn.ELU()
    if key in ("leaky_relu", "lrelu"):
        return nn.LeakyReLU(negative_slope=0.01)
    if key == "prelu":
        return nn.PReLU()
    if key in ("silu", "swish"):
        return nn.SiLU()
    if key == "tanh":
        return nn.Tanh()
    if key == "sigmoid":
        return nn.Sigmoid()
    if key in ("identity", "none"):
        return nn.Identity()
    raise ValueError(f"Unsupported activation: {name!r}")


class DilatedResidualLayer(nn.Module):
    def __init__(self, dilation, in_channels, out_channels, causal_conv=False, kernel_size=3,
                 activation="relu", dropout=0.5):
        super().__init__()
        self.causal_conv = causal_conv
        self.padding = dilation * ((kernel_size - 1) if self.causal_conv else (kernel_size // 2))
        self.conv_dilated = nn.Conv1d(in_channels,
                                      out_channels,
                                      kernel_size,
                                      padding=self.padding,
                                      dilation=dilation)
        self.conv_1x1 = nn.Conv1d(out_channels, out_channels, 1)
        self.activation = _build_activation(activation)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):  # x is of shape N x C x S, mask is N x S (1 --> keep, 0 --> mask)
        out = self.activation(self.conv_dilated(x * mask.unsqueeze(1)))
        if self.causal_conv:
            out = out[:, :, :-self.padding]
        out = self.conv_1x1(out)
        out = self.dropout(out)
        return (x + out) * mask


class SingleStageModel(nn.Module):
    def __init__(self, num_layers, num_f_maps, dim, num_classes, causal_model=False,
                 layer_activation="relu", layer_dropout=0.5):
        super().__init__()
        self.conv_1x1 = nn.Conv1d(dim, num_f_maps, 1)
        self.layers = nn.ModuleList([copy.deepcopy(
            DilatedResidualLayer(2 ** i,
                                 num_f_maps,
                                 num_f_maps,
                                 causal_conv=causal_model,
                                 activation=layer_activation,
                                 dropout=layer_dropout))
            for i in range(num_layers)
        ])
        self.conv_out = nn.Conv1d(num_f_maps, num_classes, 1)

    def forward(self, x, mask):  # x is of shape N x C x S, mask is N x S (1 --> keep, 0 --> mask)
        out = self.conv_1x1(x)
        for layer in self.layers:
            out = layer(out, mask)
        out = self.conv_out(out) * mask.unsqueeze(1)
        return out


class MultiStageModel(nn.Module):
    def __init__(self, num_stages, num_layers, num_f_maps, dim, num_classes, causal_model):
        super().__init__()
        self.stage1 = SingleStageModel(num_layers,
                                       num_f_maps,
                                       dim,
                                       num_classes,
                                       causal_model=causal_model)
        self.stages = nn.ModuleList([copy.deepcopy(
            SingleStageModel(num_layers,
                             num_f_maps,
                             num_classes,
                             num_classes,
                             causal_model=causal_model))
            for _ in range(num_stages - 1)
        ])

    def forward(self, x, mask):  # x is of shape N x C x S, mask is N x S (1 --> keep, 0 --> mask)
        if mask.dtype != torch.float32:
            mask = mask.float()
        out = self.stage1(x, mask)
        outputs = (out, )
        for s in self.stages:
            out = s(F.softmax(out, dim=1), mask)
            outputs += (out, )
        return outputs  # tuple (stage1, ..., stageK), each tensor is N x C x S
