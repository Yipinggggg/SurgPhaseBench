import copy

import torch
import torch.nn as nn

from .template import ModelTemplate
from .model import MyModel as TUNeS


LOGITS, FEATURES = "logits", "features"


class MsTUNeS(nn.Module, ModelTemplate):
    def __init__(self, d_in, num_class, causal_model,
                 down_up_cfg, conv_block_cfg, attn_cfg, mlp_cfg, conv_attn_block_cfg, transformer_cfg, max_seq_len,
                 num_stages=2, d_model=64, transformer_add_tokens=True,
                 down_blocks=[2, 2, 2], up_blocks=[2, 2, 2], temporal_scales=[2, 2, 2], channel_scales=[2, 2, 2],
                 up_kernels=[], up_dilations=[], skip_connections=True, weighted_fusion=False, fusion_weight_init=1.0,
                 forward_features=False):
        super().__init__()
        assert (num_stages > 1)
        self.num_stages = num_stages
        self.forward_features = forward_features

        self.causal = causal_model
        self.channels_last = True

        self.stages = nn.ModuleList()
        for i in range(num_stages):
            self.stages.append(
                TUNeS(
                    d_in if i == 0 else (-1 if forward_features else num_class), num_class, causal_model,
                    copy.deepcopy(down_up_cfg), copy.deepcopy(conv_block_cfg), copy.deepcopy(attn_cfg),
                    copy.deepcopy(mlp_cfg), copy.deepcopy(conv_attn_block_cfg), copy.deepcopy(transformer_cfg),
                    max_seq_len,
                    d_model, transformer_add_tokens, down_blocks, up_blocks, temporal_scales, channel_scales,
                    up_kernels, up_dilations, skip_connections, weighted_fusion, fusion_weight_init, attention_idx=-1
                )
            )

    def get_temporal_scales(self):
        return self.stages[0].temporal_scales

    def setup(self):  # call this after model has been sent to GPU
        for stage in self.stages:
            stage.setup()

    def forward(self, x, mask, feature_mask=None):
        # x is of shape N x S x C, mask is N x S (1 --> keep, 0 --> mask), feature_mask is N x S (1 --> mask)
        result = tuple()
        for stage in self.stages:
            # Some stage outputs can differ by 1 timestep due to down/up sampling rounding.
            # Keep mask aligned with the current input sequence length before every stage.
            if mask is None:
                mask = torch.ones((x.shape[0], x.shape[1]), dtype=torch.bool, device=x.device)
            else:
                if mask.device != x.device:
                    mask = mask.to(x.device)
                if mask.shape[1] != x.shape[1]:
                    if mask.shape[1] > x.shape[1]:
                        mask = mask[:, :x.shape[1]]
                    else:
                        pad = torch.zeros(
                            (mask.shape[0], x.shape[1] - mask.shape[1]),
                            dtype=mask.dtype,
                            device=mask.device,
                        )
                        mask = torch.cat([mask, pad], dim=1)

            out = stage(x, mask, feature_mask)
            result += (out, )
            if self.forward_features:
                x = out[FEATURES]
            else:
                x = torch.nn.Softmax(dim=-1)(out[LOGITS][-1])

        return result

    def parse_batch(self, batch, device_gpu, train=True, get_target=True, deep_supervision=True):
        return self.stages[0].parse_batch(batch, device_gpu, train, get_target, deep_supervision)

    def infer_batch(self, batch, device_gpu, return_logits=False):
        input, in_valid_mask, _, _, _, _, _, _, _, _, _, _, results = \
            self.parse_batch(batch, device_gpu, train=False, get_target=False, deep_supervision=False)

        result = self.forward(input, in_valid_mask, None)
        out = result[-1]  # output of final stage

        logits = out[LOGITS][-1]
        if return_logits:
            results["logits_gpu"] = logits.detach()
        else:
            _, predicted = torch.max(torch.nn.Softmax(dim=-1)(logits), dim=-1)
            results["predicted_gpu"] = predicted.detach()

        return results
