from __future__ import annotations

import torch
from torch.utils.data import ConcatDataset, DataLoader
import pytorch_lightning as pl
from src.data.datamodules.base_datamodule import build_base_datasets, print_dataset_summary

def _collate_video_sequences(batch: list[dict]) -> dict:
    max_len = max(item["frames"].shape[0] for item in batch)
    c, h, w = batch[0]["frames"].shape[1:]

    padded_frames, padded_labels, masks, video_ids = [], [], [], []

    for item in batch:
        t = item["frames"].shape[0]
        frames = torch.zeros((max_len, c, h, w), dtype=item["frames"].dtype)
        frames[:t] = item["frames"]

        labels = torch.full((max_len,), fill_value=-100, dtype=torch.long)
        labels[:t] = item["labels"]

        mask = torch.zeros((max_len,), dtype=torch.bool)
        mask[:t] = True

        padded_frames.append(frames)
        padded_labels.append(labels)
        masks.append(mask)
        video_ids.append(item["video_id"])

    return {
        "video_id": video_ids,
        "frames": torch.stack(padded_frames),
        "labels": torch.stack(padded_labels),
        "mask": torch.stack(masks),
    }

class EndToEndSequenceDataModule(pl.LightningDataModule):
    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg = cfg["data"]
        self.batch_size = int(self.cfg.get("batch_size", 1))
        self.shuffle_train = bool(self.cfg.get("shuffle_train", True))

        self.num_workers = self.cfg.get("num_workers", 4)
        self.seq_len = self.cfg.get("seq_len")
        self.seq_stride = self.cfg.get("seq_stride", self.seq_len)
        if not self.seq_len:
            print(
                "[WARN] data.seq_len is not set for end-to-end sequence mode; "
                "each sample will contain a full video, which can cause CUDA OOM."
            )

    def setup(self, stage: str | None = None):
        train_sets, val_sets, test_sets = build_base_datasets(self.cfg, mode="sequence")
        self.train_ds = ConcatDataset(train_sets) if train_sets else None
        self.val_ds = ConcatDataset(val_sets) if val_sets else None
        self.test_ds = ConcatDataset(test_sets) if test_sets else None

        extra_info = {
            "batch_size": self.cfg.get("batch_size", 1),
            "seq_len": self.seq_len,
            "seq_stride": self.seq_stride,
        }
        print_dataset_summary(train_sets, val_sets, test_sets, 
                            "EndToEndSequenceDataModule", extra_info)

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=self.shuffle_train,
                          num_workers=self.num_workers, collate_fn=_collate_video_sequences)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.num_workers, collate_fn=_collate_video_sequences)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.num_workers, collate_fn=_collate_video_sequences)
