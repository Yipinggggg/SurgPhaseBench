"""Dataset adapter utilities for frame-based training."""

from __future__ import annotations

import csv
import os
import re
import zipfile
from pathlib import Path
from pathlib import PurePosixPath

import albumentations as A
import cv2
import numpy as np
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset


PHASE_MAPS = {
    "Cholec80": {
        "Preparation": 0,
        "CalotTriangleDissection": 1,
        "ClippingCutting": 2,
        "GallbladderDissection": 3,
        "GallbladderPackaging": 4,
        "CleaningCoagulation": 5,
        "GallbladderRetraction": 6,
    },
    "RAMIE": {str(i): i - 1 for i in range(1, 14)},
    "RARP": {str(i): i for i in range(8)},
    "AutoLaparo": {str(i): i - 1 for i in range(1, 8)},
}


def natural_key(text: str) -> list:
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", text)]


def get_transform(img_size: int, augment: bool = False) -> A.Compose:
    if augment:
        return A.Compose(
            [
                A.Resize(img_size + 32, img_size + 32),
                A.ShiftScaleRotate(
                    shift_limit=0.05,
                    scale_limit=0.05,
                    rotate_limit=15,
                    p=0.5,
                ),
                A.RandomCrop(img_size, img_size),
                A.RandomBrightnessContrast(p=0.5),
                A.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                ),
                ToTensorV2(),
            ]
        )

    return A.Compose(
        [
            A.Resize(img_size, img_size),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )


class VideoFrameDataset(Dataset):
    """Frame dataset for one video (directory or zip-backed)."""

    def __init__(
        self,
        frames_path: Path,
        ann_file: Path,
        dataset_name: str,
        img_size: int = 224,
        augment: bool = False,
    ):
        self.frames_path = Path(frames_path)
        self.ann_file = Path(ann_file)
        self.dataset_name = dataset_name
        self.transform = get_transform(img_size, augment)
        self.phase_map = PHASE_MAPS.get(dataset_name, {})

        self.is_zip = self.frames_path.is_file() and self.frames_path.suffix.lower() == ".zip"
        self.zip_ref = None
        self.pid = os.getpid()

        if self.is_zip:
            self.zip_ref = zipfile.ZipFile(self.frames_path, "r")
            self.img_files = sorted(
                [
                    name
                    for name in self.zip_ref.namelist()
                    if name.lower().endswith((".jpg", ".jpeg", ".png"))
                ],
                key=self._zip_image_sort_key,
            )
        elif self.frames_path.is_dir():
            self.img_files = sorted(
                [
                    p
                    for p in self.frames_path.iterdir()
                    if p.suffix.lower() in (".jpg", ".jpeg", ".png")
                ],
                key=lambda x: natural_key(x.name),
            )
        else:
            self.img_files = []

        self.labels = self._load_labels(self.ann_file, dataset_name)
        self._check_lengths_and_log()

        if isinstance(self.labels, list):
            min_len = min(len(self.img_files), len(self.labels))
            self.img_files = self.img_files[:min_len]
            self.labels = self.labels[:min_len]

    def _zip_image_sort_key(self, name: str):
        # Sort by each path segment naturally so zip-backed ordering is deterministic.
        parts = PurePosixPath(name).parts
        return [natural_key(part) for part in parts]

    def _check_lengths_and_log(self) -> None:
        num_images = len(self.img_files)
        labels_kind = "dict" if isinstance(self.labels, dict) else "list"
        num_labels = len(self.labels)

        print(
            "[VideoFrameDataset:init] "
            f"dataset={self.dataset_name} "
            f"source={'zip' if self.is_zip else 'dir'} "
            f"frames={self.frames_path} "
            f"ann={self.ann_file} "
            f"images={num_images} labels({labels_kind})={num_labels}"
        )

        if isinstance(self.labels, list):
            if num_images != num_labels:
                print(
                    "[VideoFrameDataset:init][WARN] "
                    f"Image/label length mismatch: images={num_images}, labels={num_labels}. "
                    f"Will truncate to min={min(num_images, num_labels)}."
                )
            return

        missing_label_count = 0
        filtered_img_files = []
        for idx, img_entry in enumerate(self.img_files):
            img_name = Path(img_entry).name if self.is_zip else img_entry.name
            m = re.search(r"(\d+)(?=\.\w+$)", img_name)
            frame_num = int(m.group(1)) if m else idx
            if frame_num not in self.labels:
                missing_label_count += 1
                continue
            filtered_img_files.append(img_entry)

        if num_images != num_labels:
            print(
                "[VideoFrameDataset:init][INFO] "
                "Label dictionary size differs from image count. "
                f"images={num_images}, unique_label_frames={num_labels}."
            )

        if missing_label_count > 0:
            print(
                "[VideoFrameDataset:init][WARN] "
                f"{missing_label_count}/{num_images} images have no matched frame label; "
                "they will be skipped."
            )
            self.img_files = filtered_img_files

    def _load_labels(self, path: Path, dataset_name: str) -> list[int] | dict[int, int]:
        if not path.exists():
            print(
                "[VideoFrameDataset:init][WARN] "
                f"Missing annotation file: {path}. This video will be skipped."
            )
            return []

        labels: dict[int, int] = {}
        delimiter = "\t" if dataset_name != "RARP" else ","
        with open(path, "r") as f:
            reader = csv.reader(f, delimiter=delimiter)
            if dataset_name in ("Cholec80", "AutoLaparo"):
                next(reader, None)

            for row in reader:
                if not row:
                    continue
                try:
                    frame_idx = int(row[0])
                    raw_phase = row[1] if len(row) > 1 else row[0]
                    mapped = self.phase_map.get(str(raw_phase))
                    if mapped is None:
                        try:
                            mapped = int(raw_phase)
                        except Exception:
                            continue
                    labels[frame_idx] = mapped
                except Exception:
                    continue
        return labels

    def __len__(self) -> int:
        return len(self.img_files)

    def _ensure_worker_local_zip(self):
        current_pid = os.getpid()
        if self.zip_ref is None or self.pid != current_pid:
            if self.zip_ref is not None:
                self.zip_ref.close()
            self.zip_ref = zipfile.ZipFile(self.frames_path, "r")
            self.pid = current_pid

    def __getitem__(self, idx: int):
        img_entry = self.img_files[idx]

        if self.is_zip:
            self._ensure_worker_local_zip()
            try:
                img_data = self.zip_ref.read(img_entry)
            except Exception as e:
                raise RuntimeError(
                    f"Failed to read '{img_entry}' from '{self.frames_path}': {e}"
                ) from e
            img = cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_COLOR)
            img_name = Path(img_entry).name
        else:
            img = cv2.imread(str(img_entry))
            img_name = img_entry.name

        if img is None:
            raise RuntimeError(f"Could not decode image '{img_name}'")

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = self.transform(image=img)["image"]

        if isinstance(self.labels, dict):
            m = re.search(r"(\d+)(?=\.\w+$)", img_name)
            frame_num = int(m.group(1)) if m else idx
            label = self.labels[frame_num]
        else:
            label = self.labels[idx]

        return img, label

    def __del__(self):
        if getattr(self, "zip_ref", None) is not None:
            self.zip_ref.close()


def get_video_dataset(
    data_dir: Path,
    dataset_name: str,
    video_id: str,
    img_size: int = 224,
    augment: bool = False,
) -> VideoFrameDataset:
    """Return dataset for one video id if frames are found."""
    search_dirs = ["frames", "frames_1fps", "images", "results"]
    frames_path = None

    for d in search_dirs:
        as_dir = data_dir / d / video_id
        as_zip = data_dir / d / f"{video_id}.zip"
        if as_dir.is_dir():
            frames_path = as_dir
            break
        if as_zip.is_file():
            frames_path = as_zip
            break

    if frames_path is None:
        return VideoFrameDataset(Path("__missing__"), Path("__missing__"), dataset_name)

    ann_dir = next(
        (data_dir / d for d in ("phase_annotations", "labels", "annotation") if (data_dir / d).exists()),
        None,
    )
    ann_file = ann_dir / f"{video_id}-phase.txt" if ann_dir else Path("__missing__")
    if ann_dir and not ann_file.exists():
        for candidate in (
            f"label_{video_id}.txt",
            f"{video_id}.txt",
            f"video{video_id}-phase.txt",
        ):
            cpath = ann_dir / candidate
            if cpath.exists():
                ann_file = cpath
                break

    return VideoFrameDataset(
        frames_path=frames_path,
        ann_file=ann_file,
        dataset_name=dataset_name,
        img_size=img_size,
        augment=augment,
    )


def prepare_dataset(opts):
    """Adapter for legacy pipeline code, returns (train, val, test) triplets."""
    data_dir = Path(opts.annotation_folder)

    m = re.search(r"(\d+)$", str(opts.split))
    split_num = m.group(1) if m else "1"

    def get_dataset_list(subset_keyword: str):
        split_candidates = [
            f"{subset_keyword}.split{split_num}.bundle",
            f"{subset_keyword}.split{split_num}.txt",
            f"{subset_keyword}.txt",
        ]

        split_file = None
        for name in split_candidates:
            candidate = data_dir / "splits" / name
            if candidate.exists():
                split_file = candidate
                break

        if split_file is None:
            return []

        with open(split_file, "r") as f:
            video_ids = [line.strip().split(".")[0] for line in f if line.strip()]

        datasets = []
        for vid in video_ids:
            ds = get_video_dataset(
                data_dir=data_dir,
                dataset_name=opts.dataset,
                video_id=vid,
                img_size=opts.img_size,
                augment=not bool(getattr(opts, "no_data_aug", False)),
            )
            if len(ds) > 0:
                datasets.append((vid, ds))

        return datasets

    train = get_dataset_list("train")
    val = get_dataset_list("val")
    test = get_dataset_list("test")
    return train, val, test


__all__ = ["VideoFrameDataset", "get_video_dataset", "prepare_dataset"]
