"""
LightningModule for Stage 1: frame-level feature encoder training.

Input:  individual frames (B, 3, H, W) + integer phase labels (B,)
Output: frame-level cross-entropy loss + frame accuracy

The encoder can also be used for Stage 2 (feature extraction) via the
predict_step, which returns (B, feature_dim) feature vectors per batch.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from src.tasks.base_module import BasePhaseModule
from src.models.encoders.factory import FrameEncoder

class EncoderModule(BasePhaseModule):
    def __init__(self, cfg: dict):
        super().__init__(cfg)
        self.encoder = FrameEncoder(cfg["model"])
        self.loss_fn = nn.CrossEntropyLoss(
            label_smoothing=cfg.get("training", {}).get("label_smoothing", 0.0)
        )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        """Returns (B, num_classes) logits."""
        return self.encoder.forward_with_head(frames)

    # ------------------------------------------------------------------
    # Steps
    # ------------------------------------------------------------------

    def training_step(self, batch, batch_idx):
        frames = batch["frames"]
        labels = batch["labels"]
        batch_size = frames.shape[0]
        logits = self(frames)
        loss   = self.loss_fn(logits, labels)
        preds  = logits.argmax(dim=1)

        self.train_acc(preds, labels)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=batch_size)
        self.log("train/acc",  self.train_acc, on_step=True, on_epoch=True, batch_size=batch_size)
        return loss

    def validation_step(self, batch, batch_idx):
        frames = batch["frames"]
        labels = batch["labels"]
        batch_size = frames.shape[0]
        logits = self(frames)
        loss   = self.loss_fn(logits, labels)
        preds  = logits.argmax(dim=1)

        self.val_acc(preds, labels)
        self.val_f1(preds, labels)
        self.log("val/loss", loss, on_epoch=True, prog_bar=True, batch_size=batch_size)
        self.log("val/acc",  self.val_acc, on_epoch=True, prog_bar=True, batch_size=batch_size)
        self.log("val/f1",   self.val_f1,  on_epoch=True, batch_size=batch_size)
        return loss

    def test_step(self, batch, batch_idx):
        frames = batch["frames"]
        labels = batch["labels"]
        batch_size = frames.shape[0]
        logits = self(frames)
        loss   = self.loss_fn(logits, labels)
        preds  = logits.argmax(dim=1)
        confs = torch.softmax(logits, dim=1).amax(dim=1)
        video_ids = batch.get("video_id", [f"batch_{batch_idx}"] * batch_size)

        self.log("test/loss", loss, batch_size=batch_size)
        self.log("test/acc",  self.test_acc(preds, labels), batch_size=batch_size)
        self.log("test/f1",   self.test_f1(preds, labels), batch_size=batch_size)
        self._append_test_rows(video_ids=video_ids, labels=labels, preds=preds, confs=confs)
        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        """
        Stage 2: extract features (no grad).
        Returns dict per batch suitable for saving to disk.
        """
        frames = batch["frames"]
        with torch.no_grad():
            features = self.encoder(frames)   # (B, feature_dim)
        return {
            "video_id": batch.get("video_id", ""),
            "features": features.cpu(),
            "labels":   batch["labels"].cpu(),
        }

    # ------------------------------------------------------------------
    # Optimiser / scheduler
    # ------------------------------------------------------------------

    def configure_optimizers(self):
        opt_cfg = self.cfg.get("optimizer", {})
        sch_cfg = self.cfg.get("scheduler", {})

        name = opt_cfg.get("name", "adamw").lower()
        lr   = float(opt_cfg.get("lr", 1e-4))
        wd   = float(opt_cfg.get("weight_decay", 1e-5))

        if name == "sgd":
            optimizer = torch.optim.SGD(
                self.parameters(), lr=lr, weight_decay=wd,
                momentum=float(opt_cfg.get("momentum", 0.9)),
            )
        elif name == "adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=wd)
        else:
            optimizer = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=wd)

        sch_name = (sch_cfg.get("name", "none") or "none").lower()
        if sch_name == "none":
            return optimizer
        elif sch_name == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.trainer.max_epochs,
                eta_min=float(sch_cfg.get("eta_min", 0)),
            )
            return {"optimizer": optimizer,
                    "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"}}
        elif sch_name == "step":
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=int(sch_cfg.get("step_size", 30)),
                gamma=float(sch_cfg.get("gamma", 0.1)),
            )
            return {"optimizer": optimizer,
                    "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"}}
        return optimizer
