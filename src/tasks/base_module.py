"""Base LightningModule for all SurgPhaseBench tasks."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchmetrics import Accuracy, F1Score, JaccardIndex, Precision, Recall

class BasePhaseModule(pl.LightningModule):
    def __init__(self, cfg: dict):
        super().__init__()
        self.save_hyperparameters(cfg)
        self.cfg = cfg
        self.num_classes = cfg["model"].get("num_classes")
        
        # Loss configured in subclass
        self.loss_fn = None

        # Standard phase metrics
        metric_kwargs = dict(task="multiclass", num_classes=self.num_classes, average="macro", ignore_index=-100)
        self.train_acc = Accuracy(**metric_kwargs)
        self.val_acc = Accuracy(**metric_kwargs)
        self.val_f1 = F1Score(**metric_kwargs)
        self.val_prec = Precision(**metric_kwargs)
        self.val_rec = Recall(**metric_kwargs)
        self.val_jac = JaccardIndex(task="multiclass", num_classes=self.num_classes, ignore_index=-100, average="macro")

        self.test_acc = Accuracy(**metric_kwargs)
        self.test_f1 = F1Score(**metric_kwargs)

    def on_test_epoch_start(self):
        self._test_rows = defaultdict(list)

    def _append_test_rows(self, video_ids, labels, preds, confs, mask=None):
        labels_np = labels.detach().cpu().numpy()
        preds_np = preds.detach().cpu().numpy()
        confs_np = confs.detach().cpu().numpy()

        if labels_np.ndim == 1:
            labels_np = labels_np[:, None]
            preds_np = preds_np[:, None]
            confs_np = confs_np[:, None]

        if mask is None:
            valid_mask = labels_np != -100
        else:
            valid_mask = mask.detach().cpu().numpy().astype(bool)

        for b, video_id in enumerate(video_ids):
            valid = valid_mask[b]
            gt_seq = labels_np[b][valid].astype(np.int64)
            pred_seq = preds_np[b][valid].astype(np.int64)
            conf_seq = confs_np[b][valid].astype(np.float32)

            rows = self._test_rows[str(video_id)]
            for gt, pred, conf in zip(gt_seq.tolist(), pred_seq.tolist(), conf_seq.tolist()):
                rows.append((gt, pred, conf))

    def _resolve_test_output_dirs(self):
        eval_cfg = self.cfg.get("evaluation", {})
        predictions_dir_cfg = eval_cfg.get("predictions_dir")
        output_dir_cfg = eval_cfg.get("output_dir")

        if predictions_dir_cfg:
            by_video_dir = Path(predictions_dir_cfg)
            root_dir = by_video_dir.parent
        else:
            base_dir = Path(self.cfg.get("logging", {}).get("checkpoints", {}).get("dirpath", "outputs"))
            root_dir = base_dir / "test_predictions"
            by_video_dir = root_dir / "by_video"

        eval_dir = Path(output_dir_cfg) if output_dir_cfg else (root_dir / "eval")
        return by_video_dir, eval_dir

    def _write_test_prediction_files(self, by_video_dir: Path):
        by_video_dir.mkdir(parents=True, exist_ok=True)
        for video_id in sorted(self._test_rows.keys()):
            rows = self._test_rows[video_id]
            out_file = by_video_dir / f"{video_id}.txt"
            with out_file.open("w", encoding="utf-8") as f:
                f.write("frame\tgt\tpred\tconf\n")
                for frame_idx, (gt, pred, conf) in enumerate(rows):
                    f.write(f"{frame_idx}\tgt={int(gt)}\tpred={int(pred)}\tconf={float(conf):.6f}\n")

    def on_test_epoch_end(self):
        if not getattr(self.trainer, "is_global_zero", True):
            return
        if not getattr(self, "_test_rows", None):
            return

        by_video_dir, eval_dir = self._resolve_test_output_dirs()
        self._write_test_prediction_files(by_video_dir)

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

        name = opt_cfg.get("name", "adamw").lower()
        lr = float(opt_cfg.get("lr", 1e-4))
        wd = float(opt_cfg.get("weight_decay", 1e-5))

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
            
        if sch_name == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.cfg.get("training", {}).get("epochs", 100),
                eta_min=float(sch_cfg.get("eta_min", 0.0))
            )
            return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"}}
        
        if sch_name == "step":
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=int(sch_cfg.get("step_size", 30)),
                gamma=float(sch_cfg.get("gamma", 0.1))
            )
            return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"}}

        return optimizer
