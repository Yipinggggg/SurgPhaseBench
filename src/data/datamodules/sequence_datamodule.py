"""
LightningDataModule for Stage 3: loads precomputed per-video features.

Single supported feature format:
    - Directory of per-video .pt files (output of scripts/extract_features.py)
        Each file contains: {'features': Tensor(T, C), 'labels': Tensor(T)}
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Optional

import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset

AMBIGUOUS = object()


# ------------------------------------------------------------------
# Per-video .pt directory dataset (new format)
# ------------------------------------------------------------------

class VideoFeatureDataset(Dataset):
    """
    Loads per-video .pt feature files from a directory.

    Each .pt file must contain:
        {'features': FloatTensor(T, C), 'labels': LongTensor(T)}

    Split files are plain text lists of video stems, one per line.
    """

    def __init__(self, feature_dir: str, split_file: str, split: Optional[str] = None):
        self.feature_dir = Path(feature_dir)
        with open(split_file) as f:
            video_ids = [self._normalize_video_id(line) for line in f if line.strip()]

        self.video_ids, self.feature_paths = self._resolve_feature_paths(video_ids, split)

    @staticmethod
    def _normalize_video_id(raw_id: str) -> str:
        vid = Path(raw_id.strip()).name
        return vid[:-3] if vid.endswith(".pt") else vid

    def _resolve_feature_paths(
        self, video_ids: list[str], split: Optional[str]
    ) -> tuple[list[str], list[Path]]:
        stem_index: dict[str, list[Path]] = defaultdict(list)
        for pt_file in self.feature_dir.rglob("*.pt"):
            stem_index[pt_file.stem].append(pt_file)

        resolved_ids: list[str] = []
        resolved_paths: list[Path] = []
        missing: list[str] = []
        ambiguous: list[str] = []

        for vid in video_ids:
            path = self._pick_feature_path(vid, split, stem_index)
            if path is None:
                missing.append(vid)
                continue
            if path is AMBIGUOUS:
                ambiguous.append(vid)
                continue
            resolved_ids.append(vid)
            resolved_paths.append(path)

        if missing or ambiguous:
            details = []
            if missing:
                details.append(f"missing ({len(missing)}): {', '.join(missing[:8])}")
            if ambiguous:
                details.append(f"ambiguous ({len(ambiguous)}): {', '.join(ambiguous[:8])}")
            raise FileNotFoundError(
                "Could not resolve feature files. "
                f"feature_dir={self.feature_dir} | "
                + " | ".join(details)
                + "\nExpected feature_dir/{video_id}.pt or feature_dir/{split}/{video_id}.pt"
            )

        return resolved_ids, resolved_paths

    def _pick_feature_path(
        self, vid: str, split: Optional[str], stem_index: dict[str, list[Path]]
    ) -> Optional[Path | object]:
        candidates = [self.feature_dir / f"{vid}.pt"]
        if split:
            candidates.append(self.feature_dir / split / f"{vid}.pt")
        for p in candidates:
            if p.exists():
                return p

        matches = stem_index.get(vid, [])
        if not matches:
            return None
        if len(matches) == 1:
            return matches[0]
        if split:
            in_split = [p for p in matches if p.parent.name == split]
            if len(in_split) == 1:
                return in_split[0]
        return AMBIGUOUS

    def __len__(self) -> int:
        return len(self.video_ids)

    def __getitem__(self, idx: int) -> dict:
        vid, path = self.video_ids[idx], self.feature_paths[idx]
        data = torch.load(path, map_location="cpu")
        return {"video_id": vid,
                "features": data["features"].float(),
                "labels":   data["labels"].long()}


# ------------------------------------------------------------------
# Collate: pad variable-length sequences in a batch
# ------------------------------------------------------------------

def collate_sequences(batch: list[dict]) -> dict:
    """
    Pad sequences to the longest in the batch.
    Padded label positions are set to −100 (ignored by CrossEntropyLoss).
    """
    max_len = max(item["features"].shape[0] for item in batch)
    C = batch[0]["features"].shape[1]

    features_list, labels_list, masks_list, video_ids = [], [], [], []
    for item in batch:
        T = item["features"].shape[0]
        pad = max_len - T
        feat = torch.zeros(max_len, C)
        feat[:T] = item["features"]
        lbl = torch.full((max_len,), fill_value=-100, dtype=torch.long)
        lbl[:T] = item["labels"]
        mask = torch.zeros(max_len, dtype=torch.bool)
        mask[:T] = True

        features_list.append(feat)
        labels_list.append(lbl)
        masks_list.append(mask)
        video_ids.append(item["video_id"])

    return {
        "video_id": video_ids,
        "features": torch.stack(features_list),   # (B, T_max, C)
        "labels":   torch.stack(labels_list),     # (B, T_max)  padding = −100
        "mask":     torch.stack(masks_list),      # (B, T_max)  True = valid
    }


# ------------------------------------------------------------------
# LightningDataModule
# ------------------------------------------------------------------

class SequenceDataModule(pl.LightningDataModule):
    """
    DataModule for Stage 3 temporal model training.

        YAML config section (data):
      feature_dir:  /path/to/features/
      split_dir:    /path/to/splits/
      split_name:   split1              # → train.split1.txt, val.split1.txt, test.split1.txt

      batch_size:   1
      num_workers:  4
    """

    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg = cfg["data"]

    def _make_dataset(self, mode: str) -> Dataset:
        split_dir = Path(self.cfg["split_dir"])
        split_name = self.cfg.get("split_name", "split1")
        candidates = [
            split_dir / f"{mode}.{split_name}.txt",
            split_dir / f"{mode}.{split_name}.bundle",
        ]
        split_file = next((str(p) for p in candidates if p.exists()), None)
        if split_file is None:
            checked = " and ".join(str(p) for p in candidates)
            raise FileNotFoundError(f"No split file found for mode='{mode}'. Checked: {checked}")

        return VideoFeatureDataset(
            feature_dir=self.cfg["feature_dir"],
            split_file=split_file,
            split=mode,
        )

    def setup(self, stage: Optional[str] = None):
        if stage in ("fit", None):
            self.train_ds = self._make_dataset("train")
            self.val_ds   = self._make_dataset("val")
        if stage in ("test", None):
            self.test_ds  = self._make_dataset("test")

        counts = []
        for name, ds in [
            ("train", getattr(self, "train_ds", None)),
            ("val", getattr(self, "val_ds", None)),
            ("test", getattr(self, "test_ds", None)),
        ]:
            if ds is not None:
                counts.append(f"{name}={len(ds)}")
        print("Loaded videos: " + ", ".join(counts))

    def _loader(self, dataset: Dataset, shuffle: bool) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.cfg.get("batch_size", 1),
            shuffle=shuffle,
            num_workers=self.cfg.get("num_workers", 4),
            pin_memory=True,
            collate_fn=collate_sequences,
        )

    def train_dataloader(self) -> DataLoader:
        return self._loader(self.train_ds, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self._loader(self.val_ds, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        return self._loader(self.test_ds, shuffle=False)
