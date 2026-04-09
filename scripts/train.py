"""
Unified training entry point for SurgPhaseBench.

Usage:
    # Stage 1 – train frame-level feature encoder
    python scripts/train.py --config configs/stage1_encoder.yaml

    # Stage 3 – train temporal model on precomputed features
    python scripts/train.py --config configs/stage3_temporal.yaml

    # End-to-end – feature encoder + LSTM trained jointly
    python scripts/train.py --config configs/end_to_end_lstm.yaml

    # Override any config key on the command line
    python scripts/train.py --config configs/stage3_temporal.yaml training.epochs=200 model.name=mstcn
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make repo root importable
repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

import os
import random

import numpy as np
import torch
import yaml
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    LearningRateMonitor,
    EarlyStopping,
)
from pytorch_lightning.loggers import WandbLogger


TEMPORAL_DEFAULTS_DIR = repo_root / "configs" / "defaults" / "temporal"


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _load_config(config_path: str, overrides: list[str]) -> dict:
    """Load YAML config and apply dot-notation overrides."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    for override in overrides:
        key, _, value = override.partition("=")
        parts = key.split(".")
        d = cfg
        for part in parts[:-1]:
            d = d.setdefault(part, {})
        # Try to cast to int/float/bool before storing as string
        d[parts[-1]] = _cast(value)

    return cfg


def _deep_update(base: dict, updates: dict) -> dict:
    for k, v in updates.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = v
    return base


def _autoload_temporal_defaults(cfg: dict) -> dict:
    """Auto-load default model config by selected temporal model name.

    Defaults are loaded from:
      configs/defaults/temporal/<model_name>.yaml

    User config values keep precedence over defaults.
    """
    mode = str(cfg.get("mode", "auto")).lower()
    data = cfg.get("data", {})
    is_stage3 = (mode == "stage3") or (mode == "auto" and isinstance(data, dict) and "feature_dir" in data)
    if not is_stage3:
        return cfg

    model_name = str(cfg.get("model", {}).get("name", "")).lower().strip()
    if not model_name:
        return cfg

    defaults_path = TEMPORAL_DEFAULTS_DIR / f"{model_name}.yaml"
    if not defaults_path.exists():
        return cfg

    with open(defaults_path) as f:
        defaults_cfg = yaml.safe_load(f) or {}

    merged = _deep_update(defaults_cfg, cfg)
    return merged


def _cast(value: str):
    for fn in (int, float):
        try:
            return fn(value)
        except ValueError:
            pass
    if value.lower() in ("true", "yes"):
        return True
    if value.lower() in ("false", "no"):
        return False
    if value.lower() in ("null", "none", "~"):
        return None
    return value


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _build_callbacks(cfg: dict) -> list:
    callbacks = [LearningRateMonitor(logging_interval="epoch")]

    log_cfg = cfg.get("logging", {})
    ckpt_cfg = log_cfg.get("checkpoints", {})
    if ckpt_cfg:
        base_dir = Path(ckpt_cfg.get("dirpath", "outputs/stage1"))
        ckpt_dir = base_dir / "checkpoints"
        
        # Saving Best F1 model - flattened filename (no slashes)
        callbacks.append(
            ModelCheckpoint(
                dirpath=str(ckpt_dir),
                filename="best_f1-epoch={epoch:03d}-val_f1={val/f1:.4f}",
                monitor="val/f1",
                mode="max",
                save_top_k=1,
                auto_insert_metric_name=False
            )
        )
        # Saving Best Acc model - flattened filename (no slashes)
        callbacks.append(
            ModelCheckpoint(
                dirpath=str(ckpt_dir),
                filename="best_acc-epoch={epoch:03d}-val_acc={val/acc:.4f}",
                monitor="val/acc",
                mode="max",
                save_top_k=1,
                auto_insert_metric_name=False
            )
        )
        # Always save last model
        if ckpt_cfg.get("save_last", True):
            callbacks.append(
                ModelCheckpoint(
                    dirpath=str(ckpt_dir),
                    filename="last-epoch={epoch:03d}",
                    save_last=False,  # Use our filename instead of the default last.ckpt
                )
            )

    es_cfg = cfg.get("training", {}).get("early_stopping", {"enabled": True})
    if es_cfg.get("enabled", True):
        callbacks.append(
            EarlyStopping(
                monitor=es_cfg.get("monitor", "val/f1"),
                patience=es_cfg.get("patience", 20),
                mode=es_cfg.get("mode", "max"),
            )
        )

    return callbacks


def _build_logger(cfg: dict):
    wb = cfg.get("logging", {}).get("wandb", {})
    if not wb.get("enabled", True):
        return None
    if os.environ.get("WANDB_MODE") == "disabled":
        return None
    
    # Place wandb directory at the root of the output directory, sibling to checkpoints/
    base_dir = Path(cfg.get("logging", {}).get("checkpoints", {}).get("dirpath", "outputs"))
    
    return WandbLogger(
        project=wb.get("project", "surgical-phase-recognition"),
        entity=wb.get("entity") or None,
        name=wb.get("name") or None,
        tags=wb.get("tags", []),
        config=cfg,
        save_dir=str(base_dir),
    )


def _resolve_output_dir(cfg: dict) -> Path:
    """Resolve the run output directory from config."""
    ckpt_dirpath = cfg.get("logging", {}).get("checkpoints", {}).get("dirpath")
    if ckpt_dirpath:
        return Path(ckpt_dirpath)
    return repo_root / "outputs"


def _save_effective_config(cfg: dict, output_dir: Path):
    """Persist the effective config used by this run."""
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = output_dir / "config.yaml"
    with open(config_path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


def _resolve_precision(train_cfg: dict) -> str:
    """Map user config to a Lightning precision string.

    Supported:
    - training.precision: "16-mixed" | "bf16-mixed" | "32-true" | etc.
    - training.mixed_precision: true/false (legacy convenience switch)
    """
    precision = train_cfg.get("precision")
    if precision is not None:
        return str(precision)

    if bool(train_cfg.get("mixed_precision", False)):
        # Prefer bf16 when available; otherwise use fp16.
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            return "bf16-mixed"
        return "16-mixed"

    return "32-true"


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="SurgPhaseBench training")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("overrides", nargs="*", help="Dot-notation overrides: key=value")
    args = parser.parse_args()

    cfg = _load_config(args.config, args.overrides)
    cfg = _autoload_temporal_defaults(cfg)
    set_seed(cfg.get("seed", 42))

    output_dir = _resolve_output_dir(cfg)
    _save_effective_config(cfg, output_dir)

    # Detect training mode from config
    mode = cfg.get("mode", "auto")
    if mode == "auto":
        if "data" in cfg and "feature_dir" in cfg.get("data", {}):
            mode = "stage3"
        elif str(cfg.get("model", {}).get("name", "")).lower() in ("featureencoder+lstm", "end_to_end_lstm", "lstm"):
            mode = "end_to_end"
        else:
            mode = "stage1"

    print(f"Training mode: {mode}")

    # ------ Stage 3: temporal model on precomputed features ------
    if mode == "stage3" or "stage3" in mode:
        from src.tasks.temporal_module import TemporalModule
        from src.data.datamodules.sequence_datamodule import SequenceDataModule

        datamodule = SequenceDataModule(cfg)
        module = TemporalModule(cfg)

    # ------ End-to-end: feature encoder + LSTM ------
    elif mode == "end_to_end" or "end_to_end" in mode:
        from src.tasks.end_to_end_module import EndToEndModule
        from src.data.datamodules.end_to_end_sequence_datamodule import EndToEndSequenceDataModule

        datamodule = EndToEndSequenceDataModule(cfg)
        module = EndToEndModule(cfg)

    # ------ Stage 1: frame-level feature encoder ------
    elif mode == "stage1" or "stage1" in mode:
        from src.tasks.encoder_module import EncoderModule
        from src.data.datamodules.frame_datamodule import FrameDataModule

        datamodule = FrameDataModule(cfg)
        module = EncoderModule(cfg)

    else:
        raise ValueError(f"Unknown training mode: {mode!r}. Set 'mode' to 'stage1', 'stage3', or 'end_to_end'.")

    # Build Trainer
    train_cfg = cfg.get("training", {})
    grad_clip = train_cfg.get("grad_clip", 0) or 0
    precision = _resolve_precision(train_cfg)

    # fp16 mixed precision is unsupported on CPU.
    if not torch.cuda.is_available() and precision == "16-mixed":
        print("[WARN] training.precision=16-mixed requires CUDA; falling back to 32-true.")
        precision = "32-true"

    trainer = pl.Trainer(
        max_epochs=train_cfg.get("epochs", 100),
        precision=precision,
        gradient_clip_val=float(grad_clip) if grad_clip else None,
        limit_train_batches=train_cfg.get("limit_train_batches", 1.0),
        limit_val_batches=train_cfg.get("limit_val_batches", 1.0),
        val_check_interval=train_cfg.get("val_check_interval", 1.0),
        callbacks=_build_callbacks(cfg),
        logger=_build_logger(cfg),
        log_every_n_steps=1,
        deterministic=True,
        enable_progress_bar=True,
    )

    trainer.fit(module, datamodule=datamodule)
    trainer.test(module, datamodule=datamodule)


if __name__ == "__main__":
    main()
