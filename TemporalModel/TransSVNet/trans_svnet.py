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
        tecno_weights_path: str | None = None,
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

        if tecno_weights_path:
            print(f"Loading TeCNO weights from {tecno_weights_path}")
            state_dict = torch.load(tecno_weights_path, map_location="cpu")
            # If it's a Lightning checkpoint, extract model state
            if "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]
                # Strip 'model.' prefix if present (usually from LightningModule)
                state_dict = {k.replace("model.", ""): v for k, v in state_dict.items() if k.startswith("model.")}
            
            self.tecno.load_state_dict(state_dict)
            
            print("Freezing TeCNO weights...")
            for param in self.tecno.parameters():
                param.requires_grad = False

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
        
        # In Trans-SVNet original training, TeCNO (MS-TCN) is often run in eval mode 
        # to ensure BN and Dropout don't introduce noise to the pre-trained features.
        # We check if it's frozen; if so, we force eval mode.
        is_frozen = not next(self.tecno.parameters()).requires_grad
        was_training = self.tecno.training
        if is_frozen:
            self.tecno.eval()

        with torch.set_grad_enabled(not is_frozen):
            tecno_out = self.tecno(tecno_in, mask)

        # Restore training mode if it was on
        if is_frozen and was_training:
            self.tecno.train()

        tecno_logits = tecno_out["logits"][-1] if isinstance(tecno_out, dict) else tecno_out[-1]
        
        # Trans-SVNet expects (B, C, T) where C is classes for attention input
        # and (B, T, C) where C is feature_dim for the skip connection (long_feature)
        # The internal self.fc in Transformer wrapper handles (B, T, C) -> (B, T, num_classes)
        attn_logits = self.attention(tecno_logits, x)

        # Normalize old implementation output to (B, T, C)
        if attn_logits.ndim == 3 and attn_logits.shape[0] == x.shape[1] and attn_logits.shape[1] == x.shape[0]:
            attn_logits = attn_logits.permute(1, 0, 2)

        return {"logits": (attn_logits,)}

        return {"logits": (attn_logits,)}
