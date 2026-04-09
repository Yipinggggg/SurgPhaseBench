"""
Stage 2: extract and save per-video frame features using a trained encoder.

Usage:
  python scripts/extract_features.py \\
    --config configs/stage1_encoder.yaml \\
    --checkpoint /path/to/best_encoder.ckpt \\
    --output_dir /path/to/features/ \\
    [--split train]   # extract only one split (default: all three)

Each video produces a file: {output_dir}/{split}/{video_id}.pt
  {'features': FloatTensor(T, C), 'labels': LongTensor(T)}

These files are the input for Stage 3 (SequenceDataModule).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
import yaml
import pytorch_lightning as pl
from torch.utils.data import DataLoader

# Repo root on path
repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))


def parse_args():
    parser = argparse.ArgumentParser(description="Extract frame features (Stage 2)")
    parser.add_argument("--config",     required=True,  help="Path to stage1 YAML config")
    parser.add_argument("--checkpoint", required=True,  help="Path to trained encoder .ckpt")
    parser.add_argument("--output_dir", required=True,  help="Directory where .pt files are saved")
    parser.add_argument("--split",      default="all",  help="train | val | test | all")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Override batch size (larger = faster extraction)")
    parser.add_argument("--num_workers", type=int, default=None,
                        help="Override num_workers")
    return parser.parse_args()


def main():
    args = parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # Override settings for extraction (no shuffle, larger batches OK)
    if args.batch_size is not None:
        cfg["data"]["batch_size"] = args.batch_size
    if args.num_workers is not None:
        cfg["data"]["num_workers"] = args.num_workers
    cfg["data"]["no_data_aug"] = True

    # Load trained encoder
    from src.tasks.encoder_module import EncoderModule
    # Load checkpoint weights on CPU first to avoid CUDA OOM during deserialization
    # for large models, then move to GPU later for forward passes.
    module = EncoderModule.load_from_checkpoint(
        args.checkpoint,
        cfg=cfg,
        map_location="cpu",
    )
    module.eval()

    splits = ["train", "val", "test"] if args.split == "all" else [args.split]

    for split in splits:
        print(f"\n--- Extracting {split} split ---")
        _extract_split(module, cfg, split, args.output_dir)

    print("\nDone. Features saved to:", args.output_dir)


def _extract_split(module, cfg: dict, split: str, output_dir: str):
    from src.data.datamodules.frame_datamodule import FrameDataModule

    dm = FrameDataModule(cfg)
    dm.setup(stage="fit" if split != "test" else "test")

    if split == "train":
        # For extraction we need deterministic ordering so we can stream-save
        # each video and avoid accumulating all features in memory.
        loader = DataLoader(
            dm.train_ds,
            batch_size=dm.batch_size,
            shuffle=False,
            num_workers=dm.num_workers,
            pin_memory=True,
        )
    elif split == "val":
        loader = dm.val_dataloader()
    else:
        loader = dm.test_dataloader()

    out_dir = Path(output_dir) / split
    out_dir.mkdir(parents=True, exist_ok=True)

    def _save_video(vid: str, feat_list: list[torch.Tensor], label_list: list[torch.Tensor]):
        if not feat_list:
            return
        features_tensor = torch.stack(feat_list)    # (T, C)
        labels_tensor = torch.stack(label_list)      # (T,)
        out_path = out_dir / f"{vid}.pt"
        torch.save({"features": features_tensor, "labels": labels_tensor}, out_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    module = module.to(device)
    module.encoder.eval()

    current_vid: str | None = None
    current_feats: list[torch.Tensor] = []
    current_labels: list[torch.Tensor] = []
    saved_videos: list[str] = []

    with torch.inference_mode():
        for batch_idx, batch in enumerate(loader):
            frames    = batch["frames"].to(device)
            labels    = batch["labels"]           # keep on CPU
            video_ids = batch["video_id"]

            feats = module.encoder(frames).cpu()  # (B, feature_dim)

            if isinstance(video_ids, str):
                video_ids = [video_ids] * feats.shape[0]

            for i, vid in enumerate(video_ids):
                if current_vid is None:
                    current_vid = vid

                if vid != current_vid:
                    _save_video(current_vid, current_feats, current_labels)
                    saved_videos.append(current_vid)
                    current_vid = vid
                    current_feats = []
                    current_labels = []

                current_feats.append(feats[i])
                current_labels.append(labels[i])

            if (batch_idx + 1) % 50 == 0:
                print(f"  Processed {batch_idx + 1} batches...")

    if current_vid is not None:
        _save_video(current_vid, current_feats, current_labels)
        saved_videos.append(current_vid)

    print(f"  Saved {len(saved_videos)} videos → {out_dir}")

    # Also write split list file for SequenceDataModule
    split_list_path = Path(output_dir) / f"{split}.split1.txt"
    with open(split_list_path, "w") as f:
        for vid in sorted(saved_videos):
            f.write(vid + "\n")
    print(f"  Split list → {split_list_path}")


if __name__ == "__main__":
    main()
