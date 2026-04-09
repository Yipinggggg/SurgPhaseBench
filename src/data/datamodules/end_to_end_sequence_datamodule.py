from __future__ import annotations

import torch
from torch.utils.data import ConcatDataset, DataLoader
import pytorch_lightning as pl
from src.data.datamodules.base_datamodule import build_base_datasets

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
        configured_batch_size = int(self.cfg.get("batch_size", 1))
        configured_shuffle_train = bool(self.cfg.get("shuffle_train", False))

        if configured_batch_size != 1:
            print(
                f"[EndToEndSequenceDataModule] Overriding data.batch_size={configured_batch_size} -> 1 "
                "for strict sequential end-to-end training."
            )
        if configured_shuffle_train:
            print(
                "[EndToEndSequenceDataModule] Overriding data.shuffle_train=True -> False "
                "for strict sequential end-to-end training."
            )

        self.batch_size = 1
        self.shuffle_train = False
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

        # Print dataset info
        batch_sz = self.cfg.get("batch_size", 1)
        print("\n" + "="*50)
        print(
            f"Loaded EndToEndSequenceDataModule (batch_size={batch_sz}, "
            f"seq_len={self.seq_len}, seq_stride={self.seq_stride})"
        )
        for name, ds_list, ds_concat in [("Train", train_sets, self.train_ds),
                                         ("Val",   val_sets, self.val_ds),
                                         ("Test",  test_sets, self.test_ds)]:
            num_vids = len(ds_list) if ds_list else 0
            num_imgs = sum(len(ds) for ds in ds_list) if ds_list else 0
            num_seq = len(ds_concat) if ds_concat is not None else 0
            print(f"  {name:5s}: {num_vids:3d} videos, {num_imgs:8d} images total, {num_seq:6d} sequences")
        print("="*50 + "\n")

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=self.shuffle_train,
                          num_workers=self.num_workers, collate_fn=_collate_video_sequences)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.num_workers, collate_fn=_collate_video_sequences)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.num_workers, collate_fn=_collate_video_sequences)
