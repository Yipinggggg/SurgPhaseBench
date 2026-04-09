"""
Factory for frame-level feature encoders.

Wraps the existing CNN class from networks.py and exposes a clean interface
for use in Stage 1 training and Stage 2 feature extraction.

Usage:
    encoder, feature_dim = build_encoder(cfg["model"])
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn

# Import CNN from the package-relative path
from src.models.networks import CNN

# Ensure repo root and FeatureEncoder sub-packages are on path
_repo_root = Path(__file__).resolve().parents[3]
for _p in [_repo_root,
           _repo_root / "FeatureEncoder" / "convnext",
           _repo_root / "FeatureEncoder" / "group_norm",
           _repo_root / "FeatureEncoder" / "SurgeNet",
           _repo_root / "FeatureEncoder" / "phase_gastro_temp",
           _repo_root / "FeatureEncoder" / "EndoFM",
           _repo_root / "FeatureEncoder" / "GSViT",
           _repo_root / "FeatureEncoder" / "EndoViT",
           _repo_root / "FeatureEncoder" / "ViT",
           ]:
    _s = str(_p)
    if _s not in sys.path:
        sys.path.insert(0, _s)


def build_encoder(model_cfg: dict) -> tuple[nn.Module, int]:
    """
    Build a frame-level feature encoder.

    Args:
        model_cfg: dict with keys:
            backbone (str): backbone name (e.g. 'surgenet_big', 'vitb_dinov2', ...)
            num_classes (int): number of phase classes (used for the CNN head)
            freeze (bool): whether to freeze most backbone parameters
            batch_size (int, optional): needed by some models (e.g. GSViT)

    Returns:
        (encoder, feature_dim)
            encoder:     nn.Module with a .featureNet attribute (backbone without head)
            feature_dim: int — feature vector dimensionality
    """

    # CNN expects an argparse-like namespace
    opts = SimpleNamespace(
        freeze=model_cfg.get("freeze", False),
        batch_size=model_cfg.get("batch_size", 16),
    )

    backbone = model_cfg["backbone"]
    num_classes = model_cfg["num_classes"]

    cnn = CNN(out_size=num_classes, backbone=backbone, opts=opts)
    return cnn, cnn.feature_size


class FrameEncoder(nn.Module):
    """
    Thin wrapper around CNN for use in Stage 1 / Stage 2.

    forward(x):
        x: (B, H, W, 3) or (B, 3, H, W)
        returns: (B, feature_dim) — backbone features *before* the linear head
    """

    def __init__(self, model_cfg: dict):
        super().__init__()
        self.cnn, self.feature_dim = build_encoder(model_cfg)
        self.backbone_name = model_cfg["backbone"]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, 3, H, W)
        returns: (B, feature_dim) — raw backbone features (no classification head)
        """
        if self.backbone_name == "endofm":
            x = x.unsqueeze(2)          # (B, 3, 1, H, W)
            feats = self.cnn.featureNet(x)
        elif self.backbone_name == "endovit":
            feats = self.cnn.featureNet.forward_features(x)
            feats = feats[:, 0, :]      # CLS token
        else:
            feats = self.cnn.featureNet(x)
        return feats

    def forward_with_head(self, x: torch.Tensor) -> torch.Tensor:
        """Returns (B, num_classes) logits (for Stage 1 classification training)."""
        feats = self.forward(x)
        return self.cnn.out_layer(feats)
