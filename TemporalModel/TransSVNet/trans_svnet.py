"""
Trans-SVNet wrapper under TemporalModel hierarchy.
https://arxiv.org/pdf/2103.09712
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import torch
import torch.nn as nn

from TemporalModel.MSTCN.mstcn import MultiStageModel


def _load_transsv_transformer_class():
    module_path = Path(__file__).resolve().parents[1] / "TransSVNet" / "transformer2_3_1.py"
    spec = importlib.util.spec_from_file_location("temporalmodel_trans_svnet", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load Trans-SVNet module from: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Transformer


class TransSVNetModel(nn.Module):
    """MS-TCN + local transformer attention.

    Expected input: (B, T, C) [NTC]
    Returns: {'logits': (B, T, num_classes)} tuple-wrapped for compatibility.
    """

    def __init__(
        self,
        feature_dim: int,
        num_classes: int,
        num_stages: int = 2,
        num_layers: int = 9,
        num_f_maps: int = 64,
        causal_model: bool = True,
        local_window: int = 30,
    ):
        super().__init__()
        self.tecno = MultiStageModel(
            num_stages=num_stages,
            num_layers=num_layers,
            num_f_maps=num_f_maps,
            dim=feature_dim,
            num_classes=num_classes,
            causal_model=causal_model,
        )
        Transformer = _load_transsv_transformer_class()
        self.attention = Transformer(
            mstcn_f_maps=num_f_maps,
            mstcn_f_dim=feature_dim,
            out_features=num_classes,
            len_q=local_window,
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None):
        if mask is None:
            mask = torch.ones(x.shape[0], x.shape[1], dtype=torch.bool, device=x.device)

        tecno_in = x.permute(0, 2, 1)  # (B, C, T)
        tecno_out = self.tecno(tecno_in, mask)

        tecno_logits = tecno_out["logits"][-1] if isinstance(tecno_out, dict) else tecno_out[-1]
        attn_logits = self.attention(tecno_logits, x)

        # Normalize old implementation output to (B, T, C)
        if attn_logits.ndim == 3 and attn_logits.shape[0] == x.shape[1] and attn_logits.shape[1] == x.shape[0]:
            attn_logits = attn_logits.permute(1, 0, 2)

        return {"logits": (attn_logits,)}
