"""LightningModule for TMRNet (End-to-End LSTM + Memory Bank)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim
from src.models.temporal.end_to_end_tmrnet import EndToEndTMRNet
from src.tasks.base_module import BasePhaseModule


class TMRNetModule(BasePhaseModule):
    """
    TMRNet Lightning module: Combines temporal LSTM with prebuilt memory bank.
    
    Expected batch keys:
    - "frames": (B, T, 3, H, W) - Full video sequence
    - "memory_bank": (B, T, T_mem, C_mem) - Historical context per frame
    - "labels": (B, T) - Phase labels per frame
    """
    def __init__(self, cfg: dict):
        super().__init__(cfg)
        self.model = EndToEndTMRNet(cfg["model"])
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-100, reduction="sum")

    def _normalize_batch(self, batch):
        """Ensure batch tensors have correct shapes."""
        frames = batch["frames"]
        labels = batch["labels"]
        memory_bank = batch.get("memory_bank")  # (B, T_mem, C_mem)

        # Ensure (B, T, ...) format
        if frames.ndim == 4:
            frames = frames.unsqueeze(1)  # (B, 3, H, W) -> (B, 1, 3, H, W)
        if labels.ndim == 1:
            labels = labels.unsqueeze(1)  # (B,) -> (B, 1)

        return frames, labels, memory_bank

    def forward(self, frames, memory_bank=None):
        """
        Forward pass with full sequence processing via LSTM.
        
        Args:
            frames: (B, T, 3, H, W)
            memory_bank: (B, T, T_mem, C_mem)
            
        Returns:
            logits: (B, T, num_classes)
        """
        return self.model(frames, memory_bank)

    def _shared_step(self, batch):
        """Compute loss and predictions."""
        frames, labels, memory_bank = self._normalize_batch(batch)
        logits = self(frames, memory_bank)
        target = labels[:, -1]
        loss = self.loss_fn(logits, target)
        preds = logits.argmax(dim=-1)
        return loss, preds, target

    def training_step(self, batch, batch_idx):
        loss, preds, labels = self._shared_step(batch)
        batch_size = labels.shape[0]
        self.train_acc(preds, labels)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=batch_size)
        self.log("train/acc", self.train_acc, on_step=True, on_epoch=True, batch_size=batch_size)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, preds, labels = self._shared_step(batch)
        batch_size = labels.shape[0]
        self.val_acc(preds, labels)
        self.val_f1(preds, labels)
        self.log("val/loss", loss, on_epoch=True, prog_bar=True, batch_size=batch_size)
        self.log("val/acc", self.val_acc, on_epoch=True, prog_bar=True, batch_size=batch_size)
        self.log("val/f1", self.val_f1, on_epoch=True, prog_bar=True, batch_size=batch_size)
        return loss

    def on_train_epoch_start(self) -> None:
        """Reset LSTM hidden state at epoch start."""
        if hasattr(self.model, "reset"):
            self.model.reset()

    def on_validation_epoch_start(self) -> None:
        """Reset LSTM hidden state at validation start."""
        if hasattr(self.model, "reset"):
            self.model.reset()

    def on_test_epoch_start(self) -> None:
        """Reset LSTM hidden state at test start."""
        super().on_test_epoch_start()
        if hasattr(self.model, "reset"):
            self.model.reset()

    def test_step(self, batch, batch_idx):
        frames, labels, memory_bank = self._normalize_batch(batch)
        logits = self(frames, memory_bank)
        labels = labels[:, -1]
        loss = self.loss_fn(logits, labels)
        preds = logits.argmax(dim=-1)
        confs = torch.softmax(logits, dim=-1).amax(dim=-1)

        batch_size = labels.shape[0]

        self.log("test/loss", loss, batch_size=batch_size)
        self.log("test/acc", self.test_acc(preds, labels), batch_size=batch_size)
        self.log("test/f1", self.test_f1(preds, labels), batch_size=batch_size)

        video_ids = batch.get("video_id", [f"batch_{batch_idx}"] * batch_size)
        if isinstance(video_ids, str):
            video_ids = [video_ids]

        mask = labels != -100
        self._append_test_rows(video_ids=video_ids, labels=labels, preds=preds, confs=confs, mask=mask)
        return loss

    def configure_optimizers(self):
        opt_cfg = self.cfg.get("optimizer", {})
        sch_cfg = self.cfg.get("scheduler", {})

        name = str(opt_cfg.get("name", "adam")).lower()
        lr = float(opt_cfg.get("lr", 1e-5))
        wd = float(opt_cfg.get("weight_decay", 0.0))
        momentum = float(opt_cfg.get("momentum", 0.9))

        multi_group = bool(opt_cfg.get("multi_group", True))
        backbone_lr_factor = float(opt_cfg.get("backbone_lr_factor", 0.1))

        if multi_group:
            param_groups = [
                {"params": self.model.encoder.parameters(), "lr": lr * backbone_lr_factor},
                {"params": self.model.temporal.parameters(), "lr": lr * backbone_lr_factor},
                {"params": self.model.tmr_head.parameters(), "lr": lr},
            ]
        else:
            param_groups = [{"params": self.parameters(), "lr": lr}]

        if name == "sgd":
            optimizer = optim.SGD(
                param_groups,
                momentum=momentum,
                weight_decay=wd,
                dampening=float(opt_cfg.get("dampening", 0.0)),
                nesterov=bool(opt_cfg.get("nesterov", False)),
            )
        elif name == "adamw":
            optimizer = optim.AdamW(param_groups, weight_decay=wd)
        else:
            optimizer = optim.Adam(param_groups, weight_decay=wd)

        sch_name = str((sch_cfg.get("name", "none") or "none")).lower()
        if sch_name == "none":
            return optimizer

        if sch_name == "step":
            scheduler = optim.lr_scheduler.StepLR(
                optimizer,
                step_size=int(sch_cfg.get("step_size", 5)),
                gamma=float(sch_cfg.get("gamma", 0.1)),
            )
            return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"}}

        if sch_name in {"plateau", "reduce_on_plateau"}:
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode=str(sch_cfg.get("mode", "min")),
                factor=float(sch_cfg.get("gamma", 0.1)),
                patience=int(sch_cfg.get("patience", 2)),
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": str(sch_cfg.get("monitor", "val/loss")),
                    "interval": "epoch",
                },
            }

        if sch_name == "cosine":
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=int(self.cfg.get("training", {}).get("epochs", 100)),
                eta_min=float(sch_cfg.get("eta_min", 0.0)),
            )
            return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"}}

        return optimizer
