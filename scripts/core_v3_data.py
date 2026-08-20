"""Data loading helpers for frozen core-v3 manifests."""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


def read_core_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError(f"Empty core-v3 manifest: {path}")
    return rows


def resolve_source_path(row: dict[str, str], data_root: Path | None = None) -> Path:
    if data_root is None:
        return Path(row["source_path"])
    return data_root.joinpath(*row["source_id"].replace("\\", "/").split("/"))


def load_fixed_tile(row: dict[str, str], data_root: Path | None = None):
    import torch

    if row.get("tile_mode") != "fixed":
        raise ValueError("load_fixed_tile requires tile_mode=fixed")
    path = resolve_source_path(row, data_root)
    if not path.is_file():
        raise FileNotFoundError(path)
    crop_size = int(row["crop_size"])
    resized_width = int(row["resized_width"])
    resized_height = int(row["resized_height"])
    crop_left = int(row["crop_left"])
    crop_top = int(row["crop_top"])
    with Image.open(path) as handle:
        image = handle.convert("RGB")
    if image.size != (resized_width, resized_height):
        image = image.resize((resized_width, resized_height), Image.Resampling.BICUBIC)
    image = image.crop((crop_left, crop_top, crop_left + crop_size, crop_top + crop_size))
    array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def _stable_seed(*values: object) -> int:
    digest = hashlib.sha256(":".join(str(value) for value in values).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little", signed=False)


def load_random_crop(
    row: dict[str, str],
    crop_seed: int,
    data_root: Path | None = None,
) -> tuple[Any, dict[str, Any]]:
    import torch

    if row.get("tile_mode") != "random":
        raise ValueError("load_random_crop requires tile_mode=random")
    path = resolve_source_path(row, data_root)
    if not path.is_file():
        raise FileNotFoundError(path)
    crop_size = int(row["crop_size"])
    with Image.open(path) as handle:
        array = np.asarray(handle.convert("RGB"), dtype=np.uint8)
    height, width = array.shape[:2]
    pad_bottom = max(0, crop_size - height)
    pad_right = max(0, crop_size - width)
    if pad_bottom or pad_right:
        array = np.pad(array, ((0, pad_bottom), (0, pad_right), (0, 0)), mode="edge")
    padded_height, padded_width = array.shape[:2]
    rng = np.random.Generator(np.random.PCG64(crop_seed))
    crop_top = int(rng.integers(0, padded_height - crop_size + 1))
    crop_left = int(rng.integers(0, padded_width - crop_size + 1))
    crop = array[crop_top : crop_top + crop_size, crop_left : crop_left + crop_size]
    tensor = torch.from_numpy(crop.copy()).permute(2, 0, 1).contiguous().float().div_(255.0)
    metadata = {
        **row,
        "resolved_source_path": str(path),
        "crop_left_runtime": crop_left,
        "crop_top_runtime": crop_top,
        "pad_right_runtime": pad_right,
        "pad_bottom_runtime": pad_bottom,
        "crop_seed": crop_seed,
    }
    return tensor, metadata


class CoreV3FixedTileDataset:
    def __init__(self, manifest: Path, limit: int | None = None, data_root: Path | None = None) -> None:
        self.manifest = manifest
        self.rows = read_core_manifest(manifest)
        self.data_root = data_root
        if limit is not None:
            self.rows = self.rows[:limit]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        return {"image": load_fixed_tile(row, self.data_root), "metadata": row}


class CoreV3TrainingDataset:
    """Deterministic virtual stream of frozen source images and random crops."""

    def __init__(
        self,
        manifest: Path,
        seed: int,
        total_samples: int,
        start_sample: int = 0,
        data_root: Path | None = None,
    ) -> None:
        self.rows = read_core_manifest(manifest)
        if any(row.get("tile_mode") != "random" for row in self.rows):
            raise ValueError("Training manifest must contain only tile_mode=random rows")
        self.seed = int(seed)
        self.total_samples = int(total_samples)
        self.start_sample = int(start_sample)
        self.data_root = data_root
        self._permutations: dict[int, np.ndarray] = {}

    def __len__(self) -> int:
        return self.total_samples

    def _source_index(self, global_index: int) -> int:
        cycle, offset = divmod(global_index, len(self.rows))
        if cycle not in self._permutations:
            rng = np.random.Generator(np.random.PCG64(_stable_seed(self.seed, "source-cycle", cycle)))
            self._permutations[cycle] = rng.permutation(len(self.rows))
        return int(self._permutations[cycle][offset])

    def __getitem__(self, index: int) -> dict[str, Any]:
        global_index = self.start_sample + int(index)
        source_index = self._source_index(global_index)
        row = self.rows[source_index]
        crop_seed = _stable_seed(self.seed, "training-crop", global_index, row["source_id"])
        image, metadata = load_random_crop(row, crop_seed, self.data_root)
        metadata["global_sample_index"] = global_index
        metadata["source_index"] = source_index
        return {"image": image, "metadata": metadata}


class CoreV3CropBankDataset:
    """One frozen random crop per training source for decoder bitstream caching."""

    def __init__(self, manifest: Path, seed: int, data_root: Path | None = None) -> None:
        self.rows = read_core_manifest(manifest)
        self.seed = int(seed)
        self.data_root = data_root

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[int(index)]
        crop_seed = _stable_seed(self.seed, "decoder-crop-bank", int(index), row["source_id"])
        image, metadata = load_random_crop(row, crop_seed, self.data_root)
        metadata["source_index"] = int(index)
        return {"image": image, "metadata": metadata}
