"""TMRNet end-to-end implementation: Encoder + LSTM + Memory Bank + Relational Head."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.init as init

from src.models.encoders.factory import FrameEncoder
from src.models.temporal.tmr_head import TMRNetHead


class EndToEndTMRNet(nn.Module):
    """
    TMRNet: Temporal Relation Networks for surgical phase recognition.
    
    Combines three streams:
    1. Short-term (S_t): Frame encoder features of current sequence
    2. Temporal (LSTM): End-to-end temporal modeling of sequence
    3. Long-term (L_t): Prebuilt memory bank from historical context (fixed)
    
    The relational head fuses the LSTM output with memory bank attention.
    """
    def __init__(self, model_cfg: dict):
        super().__init__()
        # 1. Frame encoder (extracts spatial features)
        self.encoder = FrameEncoder(model_cfg)
        
        # 2. LSTM temporal modeling (end-to-end temporal learning)
        hidden_size = int(model_cfg.get("hidden_size", 512))
        num_layers = int(model_cfg.get("lstm_layers", 1))
        bidirectional = bool(model_cfg.get("bidirectional", False))
        dropout_lstm = float(model_cfg.get("dropout", 0.0)) if num_layers > 1 else 0.0
        
        self.temporal = nn.LSTM(
            input_size=self.encoder.feature_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_lstm,
            bidirectional=bidirectional,
        )
        lstm_out_dim = hidden_size * (2 if bidirectional else 1)
        
        # 3. Memory bank configuration
        memory_dim = int(model_cfg.get("memory_dim", 512))  # SV-RCNet features
        self.memory_window = int(model_cfg.get("memory_window", 30))
        
        # 4. Relational head: fuses LSTM output (S_t) with memory bank (L_t)
        self.tmr_head = TMRNetHead(
            short_dim=lstm_out_dim,  # LSTM output becomes the "short-term" query
            memory_dim=memory_dim,
            num_classes=int(model_cfg["num_classes"]),
            dropout=float(model_cfg.get("dropout", 0.2))
        )
        
        self.hidden_state = None
        self._init_weights()

    def _init_weights(self) -> None:
        for name, param in self.temporal.named_parameters():
            if "weight_ih" in name or "weight_hh" in name:
                init.xavier_normal_(param)

    def reset(self) -> None:
        """Kept for API compatibility with other modules."""
        self.hidden_state = None

    def extract_image_features(self, frames: torch.Tensor) -> torch.Tensor:
        """Extract spatial features from frames.
        
        Args:
            frames: (B, T, C, H, W)
            
        Returns:
            features: (B, T, F)
        """
        b, t, c, h, w = frames.shape
        flat = frames.reshape(b * t, c, h, w)
        return self.encoder(flat).reshape(b, t, -1)

    def forward(self, frames, memory_bank=None):
        """Process a sequence with temporal modeling + memory bank attention.
        
        Args:
            frames: (B, T, 3, H, W) - Video sequence
            memory_bank: (B, T_mem, C_mem) - Global prebuilt historical context (same for all frames)
            
        Returns:
            logits: (B, num_classes)
        """
        # 1. Extract spatial features: (B, T, C, H, W) -> (B, T, F)
        frame_features = self.extract_image_features(frames)
        b, t, _ = frame_features.shape
        
        # Match the original implementation: no hidden-state carry-over across batches.
        lstm_out, _ = self.temporal(frame_features)
        
        # 3. Memory bank fallback (if not provided)
        if memory_bank is None:
            raise ValueError("Memory bank must be provided for TMRNet forward pass.")
        
        # Match the original supervision target: classify only from the final sequence step.
        lstm_last = lstm_out[:, t - 1]  # (B, hidden_size)
        return self.tmr_head(lstm_last, memory_bank)
