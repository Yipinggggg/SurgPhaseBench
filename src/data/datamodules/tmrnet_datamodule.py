"""Hybrid TMRNet DataModule supporting frames + prebuilt memory bank features."""

from __future__ import annotations

import torch
from pathlib import Path
from torch.utils.data import DataLoader, Dataset, ConcatDataset
import pytorch_lightning as pl
from src.data.datamodules.base_datamodule import build_base_datasets

def _collate_tmrnet(batch: list[dict]) -> dict:
    """Collate frames, labels, and the global memory bank context."""
    video_ids = [item["video_id"] for item in batch]
    frames = torch.stack([item["frames"] for item in batch])    # (B, T, 3, H, W)
    labels = torch.stack([item["labels"] for item in batch])    # (B, T)
    
    # Memory bank: single global context per video (same for all frames in sequence)
    memory_bank = torch.stack([item["memory_bank"] for item in batch]) # (B, T_mem, C_mem)
    
    return {
        "video_id": video_ids,
        "frames": frames,
        "labels": labels,
        "memory_bank": memory_bank
    }

class TMRNetDatasetWrapper(Dataset):
    """
    Wraps a standard video dataset to inject prebuilt SV-RCNet features.
    It expects inner_ds to be a VideoDatasetWrapper.
    """
    def __init__(self, inner_ds, memory_bank_dir: str, memory_window: int = 30, split: str | None = None):
        self.inner = inner_ds
        self.memory_window = memory_window
        self.memory_bank_dir = Path(memory_bank_dir)
        self.video_id = getattr(inner_ds, "video_id", "unknown")
        self.split = split
        
        # Load the prebuilt feature tensor (T, C)
        # Search for .pt file matching video_id in memory_bank_dir
        self.mem_features = self._load_mem_features()
        if self.mem_features is None:
            print(f"[TMRNet] Warning: Memory bank file not found for {self.video_id} in {memory_bank_dir}")

    def _load_mem_features(self) -> torch.Tensor | None:
        # Search strategies for video files.
        # 1) Prefer split subfolder (train/val/test), then root.
        search_roots = []
        if self.split:
            search_roots.append(self.memory_bank_dir / self.split)
        search_roots.append(self.memory_bank_dir)

        aliases = [
            self.video_id,
            self.video_id.replace("video", ""),
            f"video{self.video_id}",
        ]

        candidates = []
        for root in search_roots:
            for alias in aliases:
                candidates.append(root / f"{alias}.pt")
        
        for cand in candidates:
            if cand.exists():
                data = torch.load(cand, map_location="cpu")
                return data["features"] if isinstance(data, dict) and "features" in data else data

        # Last resort: recursive glob search.
        if self.memory_bank_dir.exists():
            patterns = [f"*{alias}*.pt" for alias in aliases]
            for pattern in patterns:
                for p in self.memory_bank_dir.rglob(pattern):
                    data = torch.load(p, map_location="cpu")
                    return data["features"] if isinstance(data, dict) and "features" in data else data
            for p in self.memory_bank_dir.glob(f"*{self.video_id}*.pt"):
                data = torch.load(p, map_location="cpu")
                return data["features"] if isinstance(data, dict) and "features" in data else data
        return None

    def __len__(self):
        return len(self.inner)

    def __getitem__(self, idx):
        item = self.inner[idx]  # {"frames": (T, 3, H, W), "labels": (T), "video_id": ...}
        C_mem = self.mem_features.shape[1] if self.mem_features is not None else 512
        
        # Extract causal memory: the `memory_window` frames BEFORE the training sequence.
        # video_id may be "video_name:start-end" if sequences are being used.
        video_id = item.get("video_id", self.video_id)
        seq_start = 0
        
        # Parse absolute frame index from video_id if available (e.g., "P0006:128-256")
        if ":" in video_id:
            try:
                range_str = video_id.split(":")[-1]
                seq_start = int(range_str.split("-")[0])
            except:
                seq_start = 0
        
        # Causal memory: frames [seq_start - memory_window : seq_start]
        mem_start = max(0, seq_start - self.memory_window)
        mem_end = seq_start
        
        if self.mem_features is not None and self.mem_features.shape[0] > 0:
            memory_bank = self.mem_features[mem_start:mem_end]  # (up to memory_window, C_mem)
        else:
            memory_bank = torch.zeros(1, C_mem)
        
        # Match original behavior: for missing history, repeat the nearest available feature
        # instead of padding with zeros.
        if memory_bank.shape[0] < self.memory_window:
            pad_len = self.memory_window - memory_bank.shape[0]
            if self.mem_features is not None and self.mem_features.shape[0] > 0:
                if memory_bank.shape[0] > 0:
                    seed_feat = memory_bank[0:1]
                else:
                    seed_idx = min(max(seq_start, 0), self.mem_features.shape[0] - 1)
                    seed_feat = self.mem_features[seed_idx:seed_idx + 1]
                padding = seed_feat.repeat(pad_len, 1)
            else:
                padding = torch.zeros(pad_len, C_mem)
            memory_bank = torch.cat([padding, memory_bank], dim=0)
        
        item["memory_bank"] = memory_bank  # (memory_window, C_mem)
        return item

class TMRNetDataModule(pl.LightningDataModule):
    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg = cfg["data"]
        self.memory_bank_dir = self.cfg["memory_bank_dir"]
        self.memory_window = self.cfg.get("memory_window", 30)
        self.train_batch_size = int(self.cfg.get("train_batch_size", self.cfg.get("batch_size", 1)))
        self.val_batch_size = int(self.cfg.get("val_batch_size", self.cfg.get("batch_size", 1)))
        self.test_batch_size = int(self.cfg.get("test_batch_size", self.val_batch_size))
        self.num_workers = self.cfg.get("num_workers", 4)
        self.shuffle_train = self.cfg.get("shuffle_train", True)  # Respect config

    def _wrap_datasets(self, datasets, split: str):
        return [TMRNetDatasetWrapper(ds, self.memory_bank_dir, self.memory_window, split=split) for ds in datasets]

    def setup(self, stage=None):
        # build_base_datasets expects 'annotation_folder' for the split discovery and data loading.
        # We ensure it gets the value from 'data_dir' in our config.
        cfg_for_base = self.cfg.copy()
        if "data_dir" in cfg_for_base:
            cfg_for_base["annotation_folder"] = cfg_for_base["data_dir"]
        if "seq_stride" not in cfg_for_base:
            cfg_for_base["seq_stride"] = 1

        train, val, test = build_base_datasets(cfg_for_base, mode="sequence")

        self.train_ds = ConcatDataset(self._wrap_datasets(train, split="train")) if train else None
        self.val_ds = ConcatDataset(self._wrap_datasets(val, split="val")) if val else None
        self.test_ds = ConcatDataset(self._wrap_datasets(test, split="test")) if test else None

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.train_batch_size, shuffle=self.shuffle_train,
                          num_workers=self.num_workers, collate_fn=_collate_tmrnet)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.val_batch_size, shuffle=False,
                          num_workers=self.num_workers, collate_fn=_collate_tmrnet)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.test_batch_size, shuffle=False,
                          num_workers=self.num_workers, collate_fn=_collate_tmrnet)
