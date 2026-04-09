"""Shared utilities and base classes for SurgPhaseBench DataModules."""

from __future__ import annotations

import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from types import SimpleNamespace
from src.data.datasets.dataset import prepare_dataset

class VideoDatasetWrapper(Dataset):
    """Base wrapper for per-video datasets."""
    def __init__(
        self,
        inner: Dataset,
        video_id: str,
        mode: str = "frame",
        seq_len: int | None = None,
        seq_stride: int | None = None,
    ):
        self.inner = inner
        self.video_id = video_id
        self.mode = mode
        self.seq_len = int(seq_len) if seq_len else None
        self.seq_stride = int(seq_stride) if seq_stride else None

    def __len__(self) -> int:
        if self.mode != "sequence":
            return len(self.inner)

        total = len(self.inner)
        if not self.seq_len or self.seq_len <= 0 or total <= self.seq_len:
            return 1

        stride = self.seq_stride or self.seq_len
        if stride <= 0:
            stride = self.seq_len
        n = ((total - self.seq_len) // stride) + 1
        # Add a final tail window only when the stride does not land exactly on the end.
        if (total - self.seq_len) % stride != 0:
            n += 1
        return n

    def __getitem__(self, idx: int) -> dict:
        if self.mode == "sequence":
            return self._get_sequence(idx)
        return self._get_frame(idx)

    def _get_frame(self, idx: int) -> dict:
        sample = self.inner[idx]
        if isinstance(sample, dict):
            frames = sample.get("frames", sample.get("img", sample.get("image")))
            labels = sample.get("labels", sample.get("label", sample.get("phase")))
        else:
            frames, labels = sample[0], sample[1]
        
        # Ensure correct shape for single frames
        if isinstance(frames, torch.Tensor) and frames.ndim == 4 and frames.shape[0] == 1:
            frames = frames.squeeze(0)
        return {"frames": frames, "labels": labels, "video_id": self.video_id}

    def _get_sequence(self, idx: int) -> dict:
        total = len(self.inner)
        if not self.seq_len or self.seq_len <= 0 or total <= self.seq_len:
            start = 0
            end = total
        else:
            stride = self.seq_stride or self.seq_len
            if stride <= 0:
                stride = self.seq_len
            start = idx * stride
            max_start = max(0, total - self.seq_len)
            if start > max_start:
                start = max_start
            end = min(total, start + self.seq_len)

        frames, labels = [], []
        for i in range(start, end):
            item = self._get_frame(i)
            frames.append(item["frames"])
            labels.append(item["labels"])

        clip_video_id = self.video_id if start == 0 and end == total else f"{self.video_id}:{start}-{end}"
        return {
            "video_id": clip_video_id,
            "frames": torch.stack(frames),
            "labels": torch.as_tensor(labels, dtype=torch.long)
        }

def build_base_datasets(cfg_data, mode="frame"):
    """Common logic for building legacy datasets via prepare_dataset."""
    opts = SimpleNamespace(
        dataset=cfg_data["dataset"],
        location=cfg_data.get("location", "cluster"),
        annotation_folder=cfg_data.get("data_dir", ""),
        split=cfg_data["split"],
        no_data_aug=cfg_data.get("no_data_aug", False),
        img_size=cfg_data.get("img_size", 256),
    )
    # Add any extra opts
    for k, v in cfg_data.get("extra_opts", {}).items():
        setattr(opts, k, v)

    train_ds_raw, val_ds_raw, test_ds_raw = prepare_dataset(opts)
    
    seq_len = cfg_data.get("seq_len") if mode == "sequence" else None
    seq_stride = cfg_data.get("seq_stride") if mode == "sequence" else None

    def _wrap(ds_list):
        return [
            VideoDatasetWrapper(ds, vid, mode=mode, seq_len=seq_len, seq_stride=seq_stride)
            for vid, ds in ds_list
        ]

    return _wrap(train_ds_raw), _wrap(val_ds_raw), _wrap(test_ds_raw)
