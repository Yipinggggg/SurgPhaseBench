from __future__ import annotations

import torch
from torch.utils.data import ConcatDataset, DataLoader
import pytorch_lightning as pl
from src.data.datamodules.base_datamodule import build_base_datasets, print_dataset_summary

class FrameDataModule(pl.LightningDataModule):
    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg = cfg["data"]
        self.batch_size = self.cfg.get("batch_size", 16)
        self.num_workers = self.cfg.get("num_workers", 4)

    def setup(self, stage: str | None = None):
        train_sets, val_sets, test_sets = build_base_datasets(self.cfg, mode="frame")
        self.train_ds = ConcatDataset(train_sets) if train_sets else None
        self.val_ds = ConcatDataset(val_sets) if val_sets else None
        self.test_ds = ConcatDataset(test_sets) if test_sets else None

        print_dataset_summary(train_sets, val_sets, test_sets, "FrameDataModule (Stage 1 - Frame-Level Encoder)")

    def train_dataloader(self):
        if not self.train_ds or len(self.train_ds) == 0:
            return None
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True,
                          num_workers=self.num_workers, pin_memory=True)

    def val_dataloader(self):
        if not self.val_ds or len(self.val_ds) == 0:
            return []
        return DataLoader(self.val_ds, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.num_workers)

    def test_dataloader(self):
        if not self.test_ds or len(self.test_ds) == 0:
            return []
        return DataLoader(self.test_ds, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.num_workers)
