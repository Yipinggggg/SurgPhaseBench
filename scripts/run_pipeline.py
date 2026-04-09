"""Automated pipeline runner for two-stage and end-to-end training.

Two-stage workflow (automatic):
  1) Stage 1 encoder training
  2) Stage 2 feature extraction
  3) Stage 3 temporal training + test evaluation

End-to-end workflow (automatic):
  1) Joint feature-encoder + LSTM training
  2) Automatic test evaluation
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import yaml


def _run(cmd: list[str]) -> None:
    print("\n$", " ".join(cmd))
    subprocess.run(cmd, check=True)


def _read_yaml(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _find_best_checkpoint(checkpoint_dir: Path) -> Path:
    candidates = sorted(checkpoint_dir.rglob("*.ckpt"))
    if not candidates:
        raise FileNotFoundError(f"No checkpoint found in: {checkpoint_dir}")

    # Prefer explicit best* names when available, otherwise newest checkpoint.
    best_named = [p for p in candidates if "best" in p.name.lower()]
    pool = best_named if best_named else candidates
    return max(pool, key=lambda p: p.stat().st_mtime)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run full SurgPhaseBench pipeline")
    parser.add_argument("--config", required=True, help="Pipeline YAML config")
    args = parser.parse_args()

    pipeline_cfg = _read_yaml(Path(args.config))
    mode = str(pipeline_cfg["pipeline"]["mode"]).lower()

    repo_root = Path(__file__).resolve().parents[1]
    python = sys.executable

    if mode == "two_stage":
        stage1_cfg = str((repo_root / pipeline_cfg["pipeline"]["stage1_config"]).resolve())
        stage3_cfg = str((repo_root / pipeline_cfg["pipeline"]["stage3_config"]).resolve())
        feature_dir = str((repo_root / pipeline_cfg["pipeline"]["feature_dir"]).resolve())

        # Stage 1 train + eval
        _run([python, str(repo_root / "scripts" / "train.py"), "--config", stage1_cfg])

        # Resolve best encoder checkpoint from stage1 config checkpoint dir
        stage1_data = _read_yaml(Path(stage1_cfg))
        stage1_ckpt_dir = Path(stage1_data["logging"]["checkpoints"]["dirpath"]).resolve()
        best_encoder = _find_best_checkpoint(stage1_ckpt_dir)
        print(f"Using Stage 1 checkpoint: {best_encoder}")

        # Stage 2 extract features
        _run([
            python,
            str(repo_root / "scripts" / "extract_features.py"),
            "--config",
            stage1_cfg,
            "--checkpoint",
            str(best_encoder),
            "--output_dir",
            feature_dir,
            "--split",
            "all",
        ])

        # Stage 3 train + eval (automatic test inside train.py)
        _run([
            python,
            str(repo_root / "scripts" / "train.py"),
            "--config",
            stage3_cfg,
            f"data.feature_dir={feature_dir}",
            f"data.split_dir={feature_dir}",
        ])

    elif mode == "end_to_end":
        end_to_end_cfg = str((repo_root / pipeline_cfg["pipeline"]["end_to_end_config"]).resolve())
        _run([python, str(repo_root / "scripts" / "train.py"), "--config", end_to_end_cfg])

    else:
        raise ValueError("pipeline.mode must be 'two_stage' or 'end_to_end'")

    print("\nPipeline completed successfully.")


if __name__ == "__main__":
    main()
