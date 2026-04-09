"""End-to-end temporal model: frame encoder + LSTM + classifier."""

from __future__ import annotations

import torch
import torch.nn as nn

from src.models.encoders.factory import FrameEncoder


class EndToEndLSTM(nn.Module):
    """Jointly trains frame encoder and LSTM temporal head."""

    def __init__(self, model_cfg: dict):
        super().__init__()
        self.encoder = FrameEncoder(model_cfg)
        hidden_size = int(model_cfg.get("hidden_size", 512))
        num_layers = int(model_cfg.get("lstm_layers", 1))
        bidirectional = bool(model_cfg.get("bidirectional", False))
        dropout = float(model_cfg.get("dropout", 0.0)) if num_layers > 1 else 0.0

        self.temporal = nn.LSTM(
            input_size=self.encoder.feature_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=bidirectional,
        )
        out_dim = hidden_size * (2 if bidirectional else 1)
        self.classifier = nn.Linear(out_dim, int(model_cfg["num_classes"]))
        self.hidden_state = None

    def reset(self) -> None:
        self.hidden_state = None

    def extract_image_features(self, frames: torch.Tensor) -> torch.Tensor:
        """Match legacy TemporalCNN.extract_image_features behavior.

        Args:
            frames: (B, T, C, H, W)

        Returns:
            features: (B, T, F)
        """
        b, t, c, h, w = frames.shape
        flat = frames.reshape(b * t, c, h, w)
        return self.encoder(flat).reshape(b, t, -1)

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Args:
            frames: (B, T, C, H, W)

        Returns:
            logits: (B, T, num_classes)
        """
        feats = self.extract_image_features(frames)
        lstm_out, self.hidden_state = self.temporal(feats, self.hidden_state)
        self.hidden_state = tuple(h.detach() for h in self.hidden_state)
        return self.classifier(lstm_out)
