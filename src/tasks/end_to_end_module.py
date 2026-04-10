"""LightningModule for end-to-end training (feature encoder + LSTM)."""

from __future__ import annotations

import torch
import torch.nn as nn
from src.models.temporal.end_to_end_lstm import EndToEndLSTM
from src.tasks.base_module import BasePhaseModule

class EndToEndModule(BasePhaseModule):
    def __init__(self, cfg: dict):
        super().__init__(cfg)
        self.model = EndToEndLSTM(cfg["model"])
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    def _normalize_batch(self, batch):
        frames = batch["frames"]
        labels = batch["labels"]

        # FrameDataModule may provide (B, 3, H, W) or (B, T, 3, H, W)
        if frames.ndim == 4:
            frames = frames.unsqueeze(1)
        if labels.ndim == 1:
            labels = labels.unsqueeze(1)

        return frames, labels

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        return self.model(frames)

    def _shared_step(self, batch):
        frames, labels = self._normalize_batch(batch)
        logits = self(frames)  # (B, T, C)
        # Flatten temporal predictions so CE uses 2D input (N, C) + 1D targets (N,).
        loss = self.loss_fn(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
        preds = logits.argmax(dim=-1)
        return loss, preds, labels

    def training_step(self, batch, batch_idx):
        loss, preds, labels = self._shared_step(batch)
        batch_size = labels.shape[0]
        self.train_acc(preds.reshape(-1), labels.reshape(-1))
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=batch_size)
        self.log("train/acc", self.train_acc, on_step=True, on_epoch=True, batch_size=batch_size)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, preds, labels = self._shared_step(batch)
        batch_size = labels.shape[0]
        flat_preds = preds.reshape(-1)
        flat_labels = labels.reshape(-1)
        self.val_acc(flat_preds, flat_labels)
        self.val_f1(flat_preds, flat_labels)
        self.log("val/loss", loss, on_epoch=True, prog_bar=True, batch_size=batch_size)
        self.log("val/acc", self.val_acc, on_epoch=True, prog_bar=True, batch_size=batch_size)
        self.log("val/f1", self.val_f1, on_epoch=True, prog_bar=True, batch_size=batch_size)
        return loss

    def on_train_epoch_start(self) -> None:
        pass

    def on_validation_epoch_start(self) -> None:
        pass

    def on_test_epoch_start(self) -> None:
        super().on_test_epoch_start()

    def test_step(self, batch, batch_idx):
        frames, labels = self._normalize_batch(batch)
        logits = self(frames)  # (B, T, C)
        loss = self.loss_fn(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
        preds = logits.argmax(dim=-1)
        confs = torch.softmax(logits, dim=-1).amax(dim=-1)

        flat_preds = preds.reshape(-1)
        flat_labels = labels.reshape(-1)
        batch_size = labels.shape[0]

        self.log("test/loss", loss, batch_size=batch_size)
        self.log("test/acc", self.test_acc(flat_preds, flat_labels), batch_size=batch_size)
        self.log("test/f1", self.test_f1(flat_preds, flat_labels), batch_size=batch_size)

        video_ids = batch.get("video_id", [f"batch_{batch_idx}"] * batch_size)
        # Handle cases where video_ids might be a single string (from collate) but batch_size is 1
        if isinstance(video_ids, str):
            video_ids = [video_ids]
            
        mask = labels != -100
        self._append_test_rows(video_ids=video_ids, labels=labels, preds=preds, confs=confs, mask=mask)
        return loss

    def configure_optimizers(self):
        return super().configure_optimizers()

