"""
LightningModule for Stage 3: temporal model training on full video sequences
using precomputed frame features.

Supports all models in src/models/temporal/factory.py.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from src.tasks.base_module import BasePhaseModule
from src.models.temporal.factory import build_temporal_model, get_input_format

class TemporalModule(BasePhaseModule):
    def __init__(self, cfg: dict):
        super().__init__(cfg)
        model_cfg = cfg["model"]
        self.model = build_temporal_model(
            model_name=model_cfg["name"],
            feature_dim=model_cfg["feature_dim"],
            num_classes=model_cfg["num_classes"],
            max_seq_len=model_cfg.get("max_seq_len", 8000),
            **{k: v for k, v in model_cfg.items()
               if k not in ("name", "feature_dim", "num_classes", "max_seq_len")},
        )
        self.input_format = get_input_format(model_cfg["name"])
        self.model_name = model_cfg["name"].lower()
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=-100)
        self._build_smooth_loss()
    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, features: torch.Tensor) -> list[torch.Tensor]:
        """Run model forward; always returns a list of logit tensors."""
        features = self._to_model_format(features)
        T = self._temporal_dim(features)
        mask = self._build_mask(features, T)
        outputs = self._forward_with_mask(features, mask)
        return outputs  # list of tensors in NCT or NTC depending on model

    def _to_model_format(self, features: torch.Tensor) -> torch.Tensor:
        """Convert (B, T, C) → (B, C, T) for NCT models, leave NTC as-is."""
        if self.input_format == "NCT" and features.ndim == 3:
            return features.permute(0, 2, 1)
        return features

    def _temporal_dim(self, features: torch.Tensor) -> int:
        if self.input_format == "NCT":
            return features.shape[2]
        return features.shape[1]

    def _build_mask(self, features: torch.Tensor, T: int):
        B = features.shape[0]
        if self.model_name in ("asformer", "asformer_causal", "matransformer"):
            return torch.ones(B, 1, T, dtype=torch.float32, device=features.device)
        elif self.model_name in ("mstcn", "actionformer", "mamba", "mamba_multistage", "mstunes"):
            return torch.ones(B, T, dtype=torch.bool, device=features.device)
        return None  # opera, sahc handle masks internally

    def _forward_with_mask(self, features, mask):
        if self.model_name in ("asformer", "asformer_causal", "matransformer", "mstcn", "actionformer",
                               "mamba", "mamba_multistage"):
            out = self.model(features, mask)
        elif self.model_name == "mstunes":
            out = self.model(features, mask)
        else:
            out = self.model(features)

        # Normalise output to a list
        if isinstance(out, (list, tuple)):
            final_out = []
            for item in out:
                if isinstance(item, dict) and "logits" in item:
                    final_out.append(item["logits"][-1])
                else:
                    final_out.append(item)
            out = final_out
        elif isinstance(out, dict):
            out = out.get("logits", list(out.values())[0])
            if isinstance(out, (list, tuple)):
                out = out[-1]
        
        if not isinstance(out, (list, tuple)):
            out = [out]
        return list(out)

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------

    def _build_smooth_loss(self):
        # Only used for multi-stage NCT models
        if self.model_name in ("mstcn", "asformer", "asformer_causal", "matransformer", "mamba_multistage", "mstunes"):
            from TemporalModel.loss import TruncatedMSELoss
            channels_last = (self.input_format == "NTC")
            self.smooth_loss_fn = TruncatedMSELoss(clamp_max=4, reduction="mean", channels_last=channels_last)
            # get weight from config, default to 0.15 as in MS-TCN paper
            self.smoothing_weight = float(self.cfg["model"].get("smoothing_weight", 0.15))
        else:
            self.smooth_loss_fn = None

    def _compute_loss(self, outputs: list[torch.Tensor], labels: torch.Tensor) -> torch.Tensor:
        """
        labels: (B, T) long, padding = −100
        outputs: list of logit tensors in model's native format
        """
        labels_flat = labels.reshape(-1)  # (B*T,)

        # --- CE loss averaged over stages ---
        ce = torch.tensor(0.0, device=labels.device)
        for logits in outputs:
            if self.input_format == "NTC":
                # (B, T, C) -> (B*T, C)
                logits_flat = logits.reshape(-1, logits.shape[-1])
            else:
                # (B, C, T) -> (B*T, C)
                logits_flat = logits.permute(0, 2, 1).reshape(-1, logits.shape[1])

            ce = ce + self.ce_loss(logits_flat, labels_flat)
        ce = ce / len(outputs)

        if self.smooth_loss_fn is None or len(outputs) == 1:
            return ce

        # --- Smoothing loss (only for multi-stage NCT models) ---
        mask = (labels != -100).float()  # (B, T)
        smooth = torch.tensor(0.0, device=labels.device)
        for logits in outputs:
            smooth = smooth + self.smooth_loss_fn(logits, mask, logprobs=False)
        smooth = smooth / len(outputs)

        return ce + self.smoothing_weight * smooth

    def _get_predictions(self, outputs: list[torch.Tensor]) -> torch.Tensor:
        """Return (B, T) predicted class indices from the last stage."""
        logits = outputs[-1]
        if self.input_format == "NTC":
            return logits.argmax(dim=-1)       # (B, T)
        return logits.argmax(dim=1)            # (B, T) from (B, C, T)

    # ------------------------------------------------------------------
    # Training / validation steps
    # ------------------------------------------------------------------

    def training_step(self, batch, batch_idx):
        features = batch["features"]   # (B, T, C)
        labels   = batch["labels"]     # (B, T)
        batch_size = features.shape[0]

        outputs = self(features)
        loss = self._compute_loss(outputs, labels)
        preds = self._get_predictions(outputs)

        self.train_acc(preds.reshape(-1), labels.reshape(-1))
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=batch_size)
        self.log("train/acc",  self.train_acc, on_step=True, on_epoch=True, batch_size=batch_size)
        return loss

    def validation_step(self, batch, batch_idx):
        features = batch["features"]
        labels   = batch["labels"]
        batch_size = features.shape[0]

        outputs = self(features)
        loss = self._compute_loss(outputs, labels)
        preds = self._get_predictions(outputs)

        flat_preds  = preds.reshape(-1)
        flat_labels = labels.reshape(-1)

        self.val_acc(flat_preds, flat_labels)
        self.val_f1(flat_preds, flat_labels)
        self.val_prec(flat_preds, flat_labels)
        self.val_rec(flat_preds, flat_labels)
        self.val_jac(flat_preds, flat_labels)

        self.log("val/loss", loss, on_epoch=True, prog_bar=True, batch_size=batch_size)
        self.log("val/acc",  self.val_acc,  on_epoch=True, prog_bar=True, batch_size=batch_size)
        self.log("val/f1",   self.val_f1,   on_epoch=True, prog_bar=True, batch_size=batch_size)
        self.log("val/prec", self.val_prec, on_epoch=True, batch_size=batch_size)
        self.log("val/rec",  self.val_rec,  on_epoch=True, batch_size=batch_size)
        self.log("val/jac",  self.val_jac,  on_epoch=True, batch_size=batch_size)
        return loss

    def test_step(self, batch, batch_idx):
        features = batch["features"]
        labels   = batch["labels"]
        outputs  = self(features)
        loss     = self._compute_loss(outputs, labels)
        preds    = self._get_predictions(outputs)
        logits = outputs[-1]

        if self.input_format == "NTC":
            confs = torch.softmax(logits, dim=-1).amax(dim=-1)
        else:
            confs = torch.softmax(logits, dim=1).amax(dim=1)

        flat_preds  = preds.reshape(-1)
        flat_labels = labels.reshape(-1)
        batch_size = labels.shape[0]

        self.log("test/loss", loss, batch_size=batch_size)
        self.log("test/acc", self.test_acc(flat_preds, flat_labels), batch_size=batch_size)
        self.log("test/f1", self.test_f1(flat_preds, flat_labels), batch_size=batch_size)

        video_ids = batch.get("video_id", [f"batch_{batch_idx}"] * batch_size)
        mask = labels != -100
        self._append_test_rows(video_ids=video_ids, labels=labels, preds=preds, confs=confs, mask=mask)
        return loss

    # ------------------------------------------------------------------
    # Optimiser / scheduler
    # ------------------------------------------------------------------

    def configure_optimizers(self):
        opt_cfg  = self.cfg.get("optimizer", {})
        sch_cfg  = self.cfg.get("scheduler", {})

        name = opt_cfg.get("name", "adamw").lower()
        lr   = float(opt_cfg.get("lr", 7e-4))
        wd   = float(opt_cfg.get("weight_decay", 1e-4))

        if name == "adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=wd)
        elif name == "sgd":
            optimizer = torch.optim.SGD(
                self.parameters(), lr=lr, weight_decay=wd,
                momentum=float(opt_cfg.get("momentum", 0.9)),
            )
        else:  # default: adamw
            optimizer = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=wd)

        if not sch_cfg or sch_cfg.get("name", "none").lower() == "none":
            return optimizer

        scheduler_name = sch_cfg["name"].lower()
        if scheduler_name == "cosine":
            total_steps = self.trainer.estimated_stepping_batches
            warmup_steps = int(total_steps * float(sch_cfg.get("warmup_ratio", 0.1)))
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=lr,
                total_steps=total_steps,
                pct_start=float(sch_cfg.get("warmup_ratio", 0.1)),
                anneal_strategy="cos",
            )
            return {"optimizer": optimizer,
                    "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}

        elif scheduler_name == "step":
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=int(sch_cfg.get("step_size", 50)),
                gamma=float(sch_cfg.get("gamma", 0.1)),
            )
            return {"optimizer": optimizer,
                    "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"}}

        return optimizer
