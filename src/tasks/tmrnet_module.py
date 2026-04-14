"""LightningModule for TMRNet (End-to-End LSTM + Memory Bank)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from src.models.temporal.end_to_end_tmrnet import EndToEndTMRNet
from src.tasks.base_module import BasePhaseModule


class TMRNetModule(BasePhaseModule):
    """
    TMRNet Lightning module: Combines temporal LSTM with prebuilt memory bank.
    
    Expected batch keys:
    - "frames": (B, T, 3, H, W) - Full video sequence
    - "memory_bank": (B, T_mem, C_mem) - Prebuilt memory bank features
    - "labels": (B, T) - Phase labels per frame
    """
    def __init__(self, cfg: dict):
        super().__init__(cfg)
        self.model = EndToEndTMRNet(cfg["model"])
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-100, reduction="sum")
        # Use dict of dicts: {base_video_id: {global_frame_idx: list[(gt, pred, conf)]}}
        self._test_frames = None

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
        # Override to use frame-level storage: {base_video_id: {global_frame_idx: (gt, pred, conf)}}
        self._test_frames = {}
        if hasattr(self.model, "reset"):
            self.model.reset()

    @staticmethod
    def _parse_segment_id(segment_id: str) -> tuple[str, int]:
        """
        Parse segment_id to extract base video_id and frame offset.
        
        Examples:
            "P0100" -> ("P0100", 0)
            "P0100:0-256" -> ("P0100", 0)
            "P0100:256-512" -> ("P0100", 256)
        
        Returns:
            (base_video_id, frame_offset)
        """
        if ":" not in segment_id:
            return segment_id, 0
        
        parts = segment_id.split(":")
        base_id = parts[0]
        frame_range = parts[1]  # e.g., "0-256"
        start_frame = int(frame_range.split("-")[0])
        return base_id, start_frame

    def test_step(self, batch, batch_idx):
        frames, labels, memory_bank = self._normalize_batch(batch)
        logits = self(frames, memory_bank)  # (B, C) for TMRNet

        # TMRNet predicts one label per clip, supervised by the last frame.
        target = labels[:, -1]
        loss = self.loss_fn(logits, target)
        preds = logits.argmax(dim=-1)
        confs = torch.softmax(logits, dim=-1).amax(dim=-1)

        batch_size = labels.shape[0]
        self.log("test/loss", loss, batch_size=batch_size)
        self.log("test/acc", self.test_acc(preds, target), batch_size=batch_size)
        self.log("test/f1", self.test_f1(preds, target), batch_size=batch_size)

        video_ids = batch.get("video_id", [f"batch_{batch_idx}"] * batch_size)
        if isinstance(video_ids, str):
            video_ids = [video_ids]
        
        # Accumulate frame predictions per base video with global frame indices
        labels_np = labels.detach().cpu().numpy()
        preds_np = preds.detach().cpu().numpy()
        confs_np = confs.detach().cpu().numpy()

        for b, segment_id in enumerate(video_ids):
            base_video_id, frame_offset = self._parse_segment_id(str(segment_id))
            
            if base_video_id not in self._test_frames:
                self._test_frames[base_video_id] = {}
            
            # Map the clip prediction to the GLOBAL index of the LAST frame in the window
            # This follows the original TMRNet supervision logic.
            clip_len = int(labels_np[b].shape[0])
            last_frame_global_idx = frame_offset + (clip_len - 1)
            
            # Store (GT of last frame, Pred, Conf)
            # Use tuple instead of list since we only need the latest/only prediction per index
            self._test_frames[base_video_id][last_frame_global_idx] = (
                int(labels_np[b][-1]), int(preds_np[b]), float(confs_np[b])
            )
        
        return loss

    def on_test_epoch_end(self):
        """Write per-video prediction files. Fills the first (seq_len-1) frames with first prediction."""
        if not getattr(self.trainer, "is_global_zero", True):
            return
        if not getattr(self, "_test_frames", None):
            return

        from pathlib import Path
        
        by_video_dir, eval_dir = self._resolve_test_output_dirs()
        by_video_dir.mkdir(parents=True, exist_ok=True)
        
        # Write one file per base video with all frame predictions
        for video_id in sorted(self._test_frames.keys()):
            frames_dict = self._test_frames[video_id]  # {global_frame_idx: (gt, pred, conf)}
            if not frames_dict:
                continue
            
            sorted_indices = sorted(frames_dict.keys())
            first_idx = sorted_indices[0]
            first_data = frames_dict[first_idx] # (gt, pred, conf)
            
            out_file = by_video_dir / f"{video_id}.txt"
            with out_file.open("w", encoding="utf-8") as f:
                f.write("frame\tgt\tpred\tconf\n")
                
                # 1. Back-fill the first (seq_len - 1) frames using the first window's result
                # This ensures the prediction file length matches the GT length exactly.
                for i in range(first_idx):
                    f.write(f"{i}\tgt={first_data[0]}\tpred={first_data[1]}\tconf={first_data[2]:.6f}\n")
                
                # 2. Write the actual predictions at their index
                for idx in sorted_indices:
                    gt, pred, conf = frames_dict[idx]
                    f.write(f"{idx}\tgt={int(gt)}\tpred={int(pred)}\tconf={float(conf):.6f}\n")


        # Run evaluation on the predictions
        eval_cfg = self.cfg.get("evaluation", {})
        plot_predictions = bool(eval_cfg.get("plot_predictions", False))
        plots_dir = eval_cfg.get("plots_dir")

        from Evaluation.eval_phase import report_results

        report = report_results(
            predictions_dir=str(by_video_dir),
            out_dir=str(eval_dir),
            nlabels=self.num_classes,
            force_overwrite=True,
            plot_predictions=plot_predictions,
            plots_dir=plots_dir,
        )

        if "accuracy" in report and "mean" in report["accuracy"]:
            self.log("test/eval_accuracy", float(report["accuracy"]["mean"]), prog_bar=True)
        if "macro_f1" in report and "B" in report["macro_f1"]:
            self.log("test/eval_macro_f1", float(report["macro_f1"]["B"]["mean"]), prog_bar=True)

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
