from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from core_v3_contract import ROOT, contract_sha256, file_sha256, load_contract


FIELDS = [
    "dataset",
    "split",
    "source_split",
    "source_id",
    "source_path",
    "image_id",
    "source_width",
    "source_height",
    "sequence_id",
    "tile_id",
    "crop_size",
    "resize_scale",
    "resized_width",
    "resized_height",
    "crop_left",
    "crop_top",
    "pad_left",
    "pad_top",
    "pad_right",
    "pad_bottom",
    "tile_mode",
    "contract_version",
    "contract_sha256",
    "source_manifest_sha256",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def source_id(path: Path, root: Path) -> str:
    try:
        relative = path.resolve().relative_to(root.resolve())
    except ValueError as exc:
        raise RuntimeError(f"Dataset path is outside configured root: {path}") from exc
    return relative.as_posix()


def stable_rows(rows: list[dict[str, str]], root: Path) -> list[dict[str, str]]:
    prepared: list[dict[str, str]] = []
    for row in rows:
        item = dict(row)
        item["source_id"] = source_id(Path(row["path"]), root)
        prepared.append(item)
    return sorted(prepared, key=lambda row: row["source_id"].encode("utf-8"))


def deterministic_geometry(
    width: int,
    height: int,
    crop_size: int,
    seed: int,
    source: str,
    occurrence: int,
) -> dict[str, int | float]:
    scale = max(1.0, crop_size / float(min(width, height)))
    resized_width = max(crop_size, int(round(width * scale)))
    resized_height = max(crop_size, int(round(height * scale)))
    digest = hashlib.sha256(f"{seed}:{source}:{occurrence}".encode("utf-8")).digest()
    left_space = resized_width - crop_size
    top_space = resized_height - crop_size
    crop_left = int.from_bytes(digest[:8], "little") % (left_space + 1)
    crop_top = int.from_bytes(digest[8:16], "little") % (top_space + 1)
    return {
        "resize_scale": f"{scale:.12f}",
        "resized_width": resized_width,
        "resized_height": resized_height,
        "crop_left": crop_left,
        "crop_top": crop_top,
    }


def sequence_from_path(path: Path) -> str:
    for part in reversed(path.parts):
        if part.lower().startswith("seq"):
            return part
    return "unknown"


def manifest_row(
    row: dict[str, str],
    split: str,
    tile_id: str,
    crop_size: int,
    contract: dict[str, Any],
    contract_hash: str,
    source_manifest_hash: str,
    geometry: dict[str, int | float] | None,
) -> dict[str, Any]:
    path = Path(row["path"])
    output: dict[str, Any] = {
        "dataset": row["dataset"],
        "split": split,
        "source_split": row["split"],
        "source_id": row["source_id"],
        "source_path": str(path),
        "image_id": row["image_id"],
        "source_width": int(row["width"]),
        "source_height": int(row["height"]),
        "sequence_id": sequence_from_path(path),
        "tile_id": tile_id,
        "crop_size": crop_size,
        "resize_scale": "",
        "resized_width": "",
        "resized_height": "",
        "crop_left": "",
        "crop_top": "",
        "pad_left": 0,
        "pad_top": 0,
        "pad_right": 0,
        "pad_bottom": 0,
        "tile_mode": "random" if geometry is None else "fixed",
        "contract_version": contract["contract_version"],
        "contract_sha256": contract_hash,
        "source_manifest_sha256": source_manifest_hash,
    }
    if geometry is not None:
        output.update(geometry)
    return output


def fixed_tiles(
    rows: list[dict[str, str]],
    count: int,
    split: str,
    crop_size: int,
    seed: int,
    contract: dict[str, Any],
    contract_hash: str,
    source_manifest_hash: str,
) -> list[dict[str, Any]]:
    if not rows:
        raise RuntimeError(f"No source rows available for {split}")
    cycles = int(math.ceil(count / len(rows)))
    selected: list[tuple[dict[str, str], int]] = []
    for cycle in range(cycles):
        order = np.random.Generator(np.random.PCG64(seed + cycle)).permutation(len(rows))
        for index in order:
            selected.append((rows[int(index)], cycle))
            if len(selected) == count:
                break
        if len(selected) == count:
            break
    result: list[dict[str, Any]] = []
    for tile_index, (row, occurrence) in enumerate(selected):
        geometry = deterministic_geometry(
            int(row["width"]),
            int(row["height"]),
            crop_size,
            seed,
            row["source_id"],
            occurrence,
        )
        result.append(
            manifest_row(
                row,
                split,
                f"{split}_{tile_index:06d}",
                crop_size,
                contract,
                contract_hash,
                source_manifest_hash,
                geometry,
            )
        )
    return result


def build_action_schedules(contract: dict[str, Any], contract_hash: str) -> dict[str, Any]:
    actions = [(q_index, prefix_k) for q_index in contract["model"]["q_indices"] for prefix_k in range(1, 5)]
    schedules: dict[str, Any] = {}
    for seed in contract["training"]["model_seeds"]:
        order = np.random.Generator(np.random.PCG64(seed)).permutation(len(actions))
        cycle = [
            {"q_index": actions[int(index)][0], "prefix_k": actions[int(index)][1]}
            for index in order
        ]
        schedules[str(seed)] = {
            "warmup_steps": contract["training"]["reconstruction_warmup_steps"],
            "warmup_action": {"q_index": 0, "prefix_k": 4},
            "periodic_cycle": cycle,
            "maximum_steps": contract["training"]["joint"]["maximum_steps"],
        }
    return {
        "contract_version": contract["contract_version"],
        "contract_sha256": contract_hash,
        "balance_window_steps": contract["training"]["balanced_action_window_steps"],
        "schedules": schedules,
    }


def main() -> None:
    contract = load_contract()
    contract_hash = contract_sha256()
    crop_size = int(contract["data"]["crop_size"])
    split_seed = int(contract["data"]["split_seed"])
    paths = json.loads((ROOT / "configs" / "paths.local.json").read_text(encoding="utf-8"))["datasets"]
    visdrone_root = Path(paths["visdrone_root"])
    uavid_root = Path(paths["uavid_root"])
    vis_source = ROOT / "data" / "visdrone_manifest.csv"
    uavid_source = ROOT / "data" / "uavid_manifest.csv"
    vis_hash = file_sha256(vis_source)
    uavid_hash = file_sha256(uavid_source)

    vis_rows = stable_rows(read_csv(vis_source), visdrone_root)
    vis_train = [row for row in vis_rows if row["split"] == "train"]
    vis_val = [row for row in vis_rows if row["split"] == "val"]
    permutation = np.random.Generator(np.random.PCG64(split_seed)).permutation(len(vis_train))
    development_indices = set(int(index) for index in permutation[: contract["data"]["visdrone_development_source_images"]])
    development_sources = [row for index, row in enumerate(vis_train) if index in development_indices]
    training_sources = [row for index, row in enumerate(vis_train) if index not in development_indices]

    train_rows = [
        manifest_row(
            row,
            "train",
            "random",
            crop_size,
            contract,
            contract_hash,
            vis_hash,
            None,
        )
        for row in training_sources
    ]
    development_rows = fixed_tiles(
        development_sources,
        contract["data"]["visdrone_development_source_images"],
        "development",
        crop_size,
        split_seed,
        contract,
        contract_hash,
        vis_hash,
    )
    vis_test_rows = fixed_tiles(
        vis_val,
        contract["data"]["visdrone_test_tiles"],
        "test",
        crop_size,
        split_seed + 1,
        contract,
        contract_hash,
        vis_hash,
    )

    uavid_rows = stable_rows(read_csv(uavid_source), uavid_root)
    uavid_val_images = [
        row for row in uavid_rows if row["split"] == "val" and row.get("scene", "").lower() == "images"
    ]
    uavid_test_rows = fixed_tiles(
        uavid_val_images,
        contract["data"]["uavid_test_tiles"],
        "test",
        crop_size,
        split_seed + 2,
        contract,
        contract_hash,
        uavid_hash,
    )

    outputs = {
        "core_v3_visdrone_train.csv": train_rows,
        "core_v3_visdrone_development.csv": development_rows,
        "core_v3_visdrone_development_monitor.csv": development_rows[: contract["data"]["development_monitor_tiles"]],
        "core_v3_visdrone_test.csv": vis_test_rows,
        "core_v3_uavid_test.csv": uavid_test_rows,
    }
    for name, rows in outputs.items():
        write_csv(ROOT / "data" / name, rows)

    schedule_path = ROOT / "configs" / "core_v3_action_schedules.json"
    schedule_path.write_text(
        json.dumps(build_action_schedules(contract, contract_hash), indent=2) + "\n",
        encoding="utf-8",
    )

    train_ids = {row["source_id"] for row in train_rows}
    development_ids = {row["source_id"] for row in development_rows}
    if train_ids & development_ids:
        raise RuntimeError("VisDrone training/development source leakage detected")
    if len({row["source_id"] for row in vis_test_rows}) != len(vis_test_rows):
        raise RuntimeError("VisDrone test must contain at most one tile per source")
    if any("labels" in row["source_path"].lower() for row in uavid_test_rows):
        raise RuntimeError("UAVid label files leaked into the image test manifest")

    metadata = {
        "contract_version": contract["contract_version"],
        "contract_sha256": contract_hash,
        "split_seed": split_seed,
        "source_manifest_sha256": {"visdrone": vis_hash, "uavid": uavid_hash},
        "stable_source_list_sha256": hashlib.sha256(
            "\n".join(row["source_id"] for row in vis_train).encode("utf-8")
        ).hexdigest(),
        "permutation_sha256": hashlib.sha256(permutation.astype("<i8").tobytes()).hexdigest(),
        "action_schedule_sha256": file_sha256(schedule_path),
        "outputs": {
            name: {"rows": len(rows), "sha256": file_sha256(ROOT / "data" / name)}
            for name, rows in outputs.items()
        },
    }
    metadata_path = ROOT / "runs" / "core_v3_local" / "data_freeze_metadata.json"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(
        "PASS: core-v3 manifests frozen "
        f"(train={len(train_rows)}, development={len(development_rows)}, "
        f"vis_test={len(vis_test_rows)}, uavid_test={len(uavid_test_rows)})"
    )


if __name__ == "__main__":
    main()
