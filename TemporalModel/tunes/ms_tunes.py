import copy

import torch
import torch.nn as nn

from .template import ModelTemplate
from ..plot import visualize_predictions, visualize_bottleneck_outputs, visualize_masked_bottleneck_outputs, \
    visualize_attention_weights
from .model import MyModel as TUNeS
from .model import MyTrainer as TUNeSTrainer
from ...utils import Cholec80
from ...utils import LOGITS, FEATURES


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
                    down_up_cfg, copy.deepcopy(conv_block_cfg), attn_cfg, mlp_cfg, conv_attn_block_cfg, transformer_cfg,
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


class MsTUNeSTrainer(TUNeSTrainer):
    def __init__(self, deep_supervision, phase_recognition_loss, smooth_logits_loss, multilabel_loss,
                 phase_recognition_factor, smooth_logits_factor, smoothing_loss_max_scale,
                 multi_label_factor, bottleneck_unmasked_factor, bottleneck_masked_factor, model_scales, mask_prob,
                 device_cpu, device_gpu, nplot=4, dataset=Cholec80):
        super().__init__(
            deep_supervision, phase_recognition_loss, smooth_logits_loss, multilabel_loss,
            phase_recognition_factor, smooth_logits_factor, smoothing_loss_max_scale,
            multi_label_factor, bottleneck_unmasked_factor, bottleneck_masked_factor, model_scales, mask_prob,
            device_cpu, device_gpu, nplot, dataset
        )

    def process_batch(self, model, batch, train=True):
        input, in_valid_mask, token_mask_in, token_mask_out, \
        target, multi_scale_targets, multi_scale_valid_masks, bottleneck_target, bottleneck_valid_mask, \
        nelems, N, S, results = model.parse_batch(
            batch, self.device_gpu, train, get_target=True, deep_supervision=self.deep_supervision
        )

        ms_out = model(input, in_valid_mask, token_mask_in)

        total_loss = 0
        ms_results = tuple()
        for out in ms_out:
            results_ = copy.deepcopy(results)
            self.process_(
                out, in_valid_mask, token_mask_in, token_mask_out,
                target, multi_scale_targets, multi_scale_valid_masks, bottleneck_target, bottleneck_valid_mask,
                nelems, N, results_, train
            )
            ms_results += (results_, )
            if "total_loss" in results_:
                total_loss = total_loss + results_["total_loss"]

        # for now: return results of final stage
        final_results = ms_results[-1]
        if train is True:
            final_results["total_loss"] = total_loss / model.num_stages

        return final_results

    def visualize_outputs(self, epoch, logger, logger_prefix=None, train=True):
        if train is True:
            logger_prefix = "train"
            logger.add_figure("{}/bottleneck_probs".format(logger_prefix),
                              visualize_masked_bottleneck_outputs(self.predictions, self.dataset.num_phases), epoch)
            # logger.add_figure("{}/attention".format(logger_prefix),
            #                   visualize_attention_weights(self.predictions), epoch)
        else:
            assert (logger_prefix is not None)
            achieved_metrics = sorted(sorted(list(self.predictions.keys()), key=lambda t: t[0]), key=lambda t: t[1])
            if len(achieved_metrics) > 0:
                # show results with lowest performance
                to_plot = [self.predictions[key_] for key_ in achieved_metrics[:self.nplot]]
                logger.add_figure("{}/predictions".format(logger_prefix), visualize_predictions(to_plot, self.dataset.num_phases), epoch)
                logger.add_figure("{}/bottleneck_probs".format(logger_prefix),
                                  visualize_bottleneck_outputs(to_plot, self.dataset.num_phases), epoch)
                # logger.add_figure("{}/attention".format(logger_prefix), visualize_attention_weights(to_plot), epoch)
