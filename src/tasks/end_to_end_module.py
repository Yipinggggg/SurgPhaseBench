"""LightningModule for end-to-end training (feature encoder + LSTM)."""

from __future__ import annotations

import torch
import torch.nn as nn
import numpy as np
from src.models.temporal.end_to_end_lstm import EndToEndLSTM
from src.tasks.base_module import BasePhaseModule

class EndToEndModule(BasePhaseModule):
    def __init__(self, cfg: dict):
        super().__init__(cfg)
        self.model = EndToEndLSTM(cfg["model"])
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
        # Use dict of dicts: {base_video_id: {global_frame_idx: list[(gt, pred, conf)]}}
        self._test_frames = None
        # Prediction mode: "last" (override with latest window) or "majority" (vote across windows)
        self.prediction_mode = cfg.get("test", {}).get("prediction_mode", "last")

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
        # Override to use frame-level storage: {base_video_id: {global_frame_idx: list[(gt, pred, conf)]}}
        self._test_frames = {}
        # Re-fetch prediction mode from config in case it wasn't set in __init__ properly
        self.prediction_mode = self.cfg.get("test", {}).get("prediction_mode", "last")
        print(f"[DEBUG] EndToEndModule.on_test_epoch_start: prediction_mode='{self.prediction_mode}'")

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
        frames, labels = self._normalize_batch(batch)
        logits = self(frames)  # (B, T, C)
        loss = self.loss_fn(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
        preds = logits.argmax(dim=-1)  # (B, T)
        confs = torch.softmax(logits, dim=-1).amax(dim=-1)  # (B, T)

        flat_preds = preds.reshape(-1)
        flat_labels = labels.reshape(-1)
        batch_size = labels.shape[0]

        self.log("test/loss", loss, batch_size=batch_size)
        self.log("test/acc", self.test_acc(flat_preds, flat_labels), batch_size=batch_size)
        self.log("test/f1", self.test_f1(flat_preds, flat_labels), batch_size=batch_size)

        video_ids = batch.get("video_id", [f"batch_{batch_idx}"] * batch_size)
        if isinstance(video_ids, str):
            video_ids = [video_ids]

        mask = labels != -100  # (B, T)
        
        # Accumulate frame predictions per base video with global frame indices
        labels_np = labels.detach().cpu().numpy()
        preds_np = preds.detach().cpu().numpy()
        confs_np = confs.detach().cpu().numpy()
        mask_np = mask.detach().cpu().numpy()

        for b, segment_id in enumerate(video_ids):
            # Parse segment_id to get base video_id and frame offset
            base_video_id, frame_offset = self._parse_segment_id(str(segment_id))
            
            # Initialize dict for this video if needed
            if base_video_id not in self._test_frames:
                self._test_frames[base_video_id] = {}
            
            # Extract valid frames for this batch element
            valid_mask = mask_np[b]
            if valid_mask.any():
                valid_labels = labels_np[b][valid_mask].astype(np.int64)
                valid_preds = preds_np[b][valid_mask].astype(np.int64)
                valid_confs = confs_np[b][valid_mask].astype(np.float32)
                
                # Store each frame with its global index
                for local_idx, (gt, pred, conf) in enumerate(
                    zip(valid_labels.tolist(), valid_preds.tolist(), valid_confs.tolist())
                ):
                    global_frame_idx = frame_offset + local_idx
                    if global_frame_idx not in self._test_frames[base_video_id]:
                        self._test_frames[base_video_id][global_frame_idx] = []
                    self._test_frames[base_video_id][global_frame_idx].append((gt, pred, conf))
        
        return loss

    def on_test_epoch_end(self):
        """Write per-video prediction files with all frames from sliding windows."""
        if not getattr(self.trainer, "is_global_zero", True):
            return
        if not getattr(self, "_test_frames", None):
            print("[WARN] EndToEndModule.on_test_epoch_end: No test frames found in self._test_frames.")
            return

        from pathlib import Path
        from collections import Counter
        
        print(f"[DEBUG] EndToEndModule.on_test_epoch_end: Processing {len(self._test_frames)} videos. Mode: {self.prediction_mode}")
        
        by_video_dir, eval_dir = self._resolve_test_output_dirs()
        by_video_dir.mkdir(parents=True, exist_ok=True)
        
        # Write one file per base video with all frame predictions
        for video_id in sorted(self._test_frames.keys()):
            frames_dict = self._test_frames[video_id]  # {global_frame_idx: list[(gt, pred, conf)]}
            
            if not frames_dict:
                continue
            
            out_file = by_video_dir / f"{video_id}.txt"
            with out_file.open("w", encoding="utf-8") as f:
                f.write("frame\tgt\tpred\tconf\n")
                # Write frames in order of global frame index
                for frame_idx in sorted(frames_dict.keys()):
                    predictions = frames_dict[frame_idx]
                    
                    if self.prediction_mode == "majority":
                        # Majority vote across windows for this frame
                        preds = [p[1] for p in predictions]
                        pred = Counter(preds).most_common(1)[0][0]
                        # For gt and conf, we still need representatives
                        gt = predictions[0][0]
                        conf = np.mean([p[2] for p in predictions])
                    else:
                        # Use the prediction from the last window that covered this frame
                        gt, pred, conf = predictions[-1]
                    
                    f.write(f"{frame_idx}\tgt={int(gt)}\tpred={int(pred)}\tconf={float(conf):.6f}\n")
            print(f"[INFO] Wrote predictions for video {video_id} to {out_file}")

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
        return super().configure_optimizers()

