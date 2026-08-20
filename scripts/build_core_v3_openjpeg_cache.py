"""Build the four-layer LRCP OpenJPEG operational-reference action cache."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import sqlite3
import struct
import subprocess
import sys
from tempfile import TemporaryDirectory
import time
from typing import Any

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core_v3_contract import contract_sha256, file_sha256, files_sha256, load_contract
from core_v3_data import CoreV3FixedTileDataset


EOC = b"\xff\xd9"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--baa-cache", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--opj-compress", type=Path, required=True)
    parser.add_argument("--opj-decompress", type=Path, required=True)
    parser.add_argument("--visdrone-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--calibration-iterations", type=int, default=3)
    parser.add_argument("--mode", choices=("formal", "smoke"), default="formal")
    parser.add_argument("--smoke-limit", type=int, default=2)
    return parser.parse_args()


def run_codec(command: list[str]) -> float:
    started = time.perf_counter()
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    if completed.returncode != 0:
        excerpt = (completed.stdout + "\n" + completed.stderr)[-1000:]
        raise RuntimeError(f"Codec failed ({completed.returncode}): {' '.join(command)}\n{excerpt}")
    return elapsed_ms


def layer_prefixes(codestream: bytes) -> list[bytes]:
    if not codestream.startswith(b"\xff\x4f"):
        raise ValueError("OpenJPEG output is not a raw JPEG2000 codestream")
    boundaries: list[int] = []
    offset = codestream.find(b"\xff\x90")
    while offset >= 0:
        if offset + 12 > len(codestream):
            raise ValueError("Truncated SOT marker")
        psot = struct.unpack(">I", codestream[offset + 6 : offset + 10])[0]
        if psot <= 0 or offset + psot > len(codestream):
            raise ValueError("Unsupported or invalid JPEG2000 tile-part length")
        boundary = offset + psot
        boundaries.append(boundary)
        offset = codestream.find(b"\xff\x90", boundary)
    if len(boundaries) != 4:
        raise ValueError(f"Expected four -TP L tile parts, found {len(boundaries)}")
    prefixes = [codestream[:boundary] + EOC for boundary in boundaries[:3]] + [codestream]
    if any(len(prefixes[index]) >= len(prefixes[index + 1]) for index in range(3)):
        raise ValueError("OpenJPEG layer prefixes are not strictly increasing")
    return prefixes


def tensor_to_png(tensor: Any, path: Path) -> None:
    array = (
        tensor.detach().cpu().clamp(0.0, 1.0).mul(255.0).round().byte().permute(1, 2, 0).numpy()
    )
    Image.fromarray(array, mode="RGB").save(path)


def encode_image(compressor: Path, input_path: Path, output_path: Path, ratios: list[float]) -> float:
    return run_codec(
        [
            str(compressor),
            "-i", str(input_path),
            "-o", str(output_path),
            "-r", ",".join(f"{value:.8f}" for value in ratios),
            "-p", "LRCP",
            "-TP", "L",
            "-I",
            "-n", "6",
        ]
    )


def target_bpps(path: Path) -> tuple[list[float], dict[str, Any]]:
    connection = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)
    metadata = {key: json.loads(value) for key, value in connection.execute("SELECT key, value FROM metadata")}
    action_meta = metadata.get("action_cache")
    if not action_meta or action_meta.get("status") != "pass":
        raise RuntimeError("BAA action cache is incomplete")
    values = []
    for prefix_k in range(1, 5):
        rows = [
            float(row[0])
            for row in connection.execute(
                "SELECT bpp FROM action_entries WHERE dataset='development' AND q_index=1 AND prefix_k=?",
                (prefix_k,),
            )
        ]
        if len(rows) != 256:
            raise RuntimeError("Formal OpenJPEG targets require 256 BAA development rows per layer")
        values.append(float(np.median(rows)))
    connection.close()
    return values, metadata


def calibrate_ratios(
    compressor: Path,
    dataset: Any,
    target: list[float],
    iterations: int,
    temporary: Path,
) -> tuple[list[float], list[dict[str, Any]]]:
    ratios = [max(1.001, 24.0 / max(value, 1e-6)) for value in target]
    history: list[dict[str, Any]] = []
    for iteration in range(iterations):
        bpps: list[list[float]] = [[] for _ in range(4)]
        for image_index in range(len(dataset)):
            input_path = temporary / "calibration.png"
            output_path = temporary / "calibration.j2k"
            tensor_to_png(dataset[image_index]["image"], input_path)
            encode_image(compressor, input_path, output_path, ratios)
            prefixes = layer_prefixes(output_path.read_bytes())
            for prefix_k, prefix in enumerate(prefixes):
                bpps[prefix_k].append(len(prefix) * 8.0 / (512 * 512))
        medians = [float(np.median(values)) for values in bpps]
        history.append({"iteration": iteration, "ratios": ratios, "median_bpp": medians})
        ratios = [
            max(1.001, ratios[index] * medians[index] / max(target[index], 1e-9))
            for index in range(4)
        ]
        for index in range(1, 4):
            if ratios[index] >= ratios[index - 1]:
                ratios[index] = max(1.001, ratios[index - 1] * 0.95)
    return ratios, history


def main() -> None:
    import torch
    from pytorch_msssim import ms_ssim

    args = parse_args()
    contract = load_contract()
    if args.seed not in contract["training"]["model_seeds"]:
        raise ValueError("Seed is outside the frozen contract")
    if not args.opj_compress.is_file() or not args.opj_decompress.is_file():
        raise FileNotFoundError("OpenJPEG command-line tools are required")
    if args.mode == "formal" and args.calibration_iterations != 3:
        raise ValueError("Formal OpenJPEG calibration uses three frozen iterations")
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    manifests = {
        "development": ROOT / "data" / "core_v3_visdrone_development.csv",
        "visdrone": ROOT / "data" / "core_v3_visdrone_test.csv",
    }
    limits = {"development": None, "visdrone": None}
    if args.mode == "smoke":
        limits = {key: args.smoke_limit for key in limits}
    datasets = {
        name: CoreV3FixedTileDataset(path, limit=limits[name], data_root=args.visdrone_root)
        for name, path in manifests.items()
    }
    target, baa_metadata = target_bpps(args.baa_cache)
    args.cache.parent.mkdir(parents=True, exist_ok=True)
    sample_dir = args.cache.parent / "openjpeg_samples"
    sample_dir.mkdir(parents=True, exist_ok=True)
    codec_sha = files_sha256([args.opj_compress, args.opj_decompress])
    with TemporaryDirectory(prefix="core_v3_openjpeg_") as temporary_name:
        temporary = Path(temporary_name)
        calibration_dataset = datasets["development"]
        ratios, history = calibrate_ratios(
            args.opj_compress,
            calibration_dataset,
            target,
            args.calibration_iterations,
            temporary,
        )
        connection = sqlite3.connect(args.cache)
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=NORMAL")
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS action_entries (
                dataset TEXT NOT NULL,
                image_index INTEGER NOT NULL,
                image_id TEXT NOT NULL,
                sequence_id TEXT NOT NULL,
                q_index INTEGER NOT NULL,
                prefix_k INTEGER NOT NULL,
                stream_sha256 TEXT NOT NULL,
                payload_bytes INTEGER NOT NULL,
                network_bytes INTEGER NOT NULL,
                bpp REAL NOT NULL,
                psnr REAL NOT NULL,
                ms_ssim REAL NOT NULL,
                encode_ms REAL NOT NULL,
                entropy_decode_ms REAL NOT NULL,
                decoder_ms REAL NOT NULL,
                decode_ms REAL NOT NULL,
                PRIMARY KEY (dataset, image_index, q_index, prefix_k)
            );
            """
        )
        existing_metadata = {
            key: json.loads(value) for key, value in connection.execute("SELECT key, value FROM metadata")
        }
        if existing_metadata:
            prior = existing_metadata.get("action_cache", {})
            if prior.get("codec_sha256") != codec_sha or prior.get("ratios") != ratios:
                raise RuntimeError("Existing OpenJPEG cache uses another codec or calibration")
        existing = {
            (str(dataset), int(image_index), int(prefix_k))
            for dataset, image_index, prefix_k in connection.execute(
                "SELECT dataset, image_index, prefix_k FROM action_entries"
            )
        }
        inserted = 0
        for dataset_name, dataset in datasets.items():
            for image_index in range(len(dataset)):
                sample = dataset[image_index]
                input_path = temporary / "input.png"
                stream_path = temporary / "stream.j2k"
                tensor_to_png(sample["image"], input_path)
                encode_ms = encode_image(args.opj_compress, input_path, stream_path, ratios)
                stream = stream_path.read_bytes()
                prefixes = layer_prefixes(stream)
                if dataset_name == "development" and image_index == 0:
                    shutil.copy2(stream_path, sample_dir / "development_image0_full.j2k")
                    for prefix_k, prefix in enumerate(prefixes, start=1):
                        (sample_dir / f"development_image0_layer{prefix_k}.j2k").write_bytes(prefix)
                target_tensor = sample["image"].unsqueeze(0).to(device)
                for prefix_k, prefix in enumerate(prefixes, start=1):
                    if (dataset_name, image_index, prefix_k) in existing:
                        continue
                    prefix_path = temporary / f"prefix_{prefix_k}.j2k"
                    output_path = temporary / f"decoded_{prefix_k}.png"
                    prefix_path.write_bytes(prefix)
                    decode_ms = run_codec(
                        [str(args.opj_decompress), "-i", str(prefix_path), "-o", str(output_path)]
                    )
                    with Image.open(output_path) as handle:
                        reconstruction_array = np.asarray(handle.convert("RGB"), dtype=np.float32) / 255.0
                    reconstruction = torch.from_numpy(reconstruction_array).permute(2, 0, 1).unsqueeze(0).to(device)
                    mse = float((reconstruction - target_tensor).square().mean().clamp_min(1e-12).cpu())
                    psnr = -10.0 * math.log10(mse)
                    quality = float(
                        ms_ssim(reconstruction.float(), target_tensor.float(), data_range=1.0).cpu()
                    )
                    payload_bytes = len(prefix)
                    packets = (payload_bytes + 1459) // 1460
                    connection.execute(
                        """
                        INSERT INTO action_entries(
                            dataset, image_index, image_id, sequence_id, q_index, prefix_k,
                            stream_sha256, payload_bytes, network_bytes, bpp, psnr, ms_ssim,
                            encode_ms, entropy_decode_ms, decoder_ms, decode_ms
                        ) VALUES (?, ?, ?, ?, 0, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?, ?)
                        """,
                        (
                            dataset_name,
                            image_index,
                            sample["metadata"]["tile_id"],
                            sample["metadata"].get("sequence_id", "unknown"),
                            prefix_k,
                            hashlib.sha256(prefix).hexdigest(),
                            payload_bytes,
                            payload_bytes + packets * 40,
                            payload_bytes * 8.0 / (512 * 512),
                            psnr,
                            quality,
                            encode_ms,
                            decode_ms,
                            decode_ms,
                        ),
                    )
                    inserted += 1
                if (image_index + 1) % 8 == 0:
                    connection.commit()
                    print(f"OPENJPEG: {dataset_name} {image_index + 1}/{len(dataset)}", flush=True)
        expected = sum(len(dataset) for dataset in datasets.values()) * 4
        final_count = int(connection.execute("SELECT COUNT(*) FROM action_entries").fetchone()[0])
        if final_count != expected:
            raise RuntimeError(f"OpenJPEG cache incomplete: {final_count} != {expected}")
        achieved_median_bpp = []
        for prefix_k in range(1, 5):
            values = [
                float(row[0])
                for row in connection.execute(
                    "SELECT bpp FROM action_entries WHERE dataset='development' AND prefix_k=?",
                    (prefix_k,),
                )
            ]
            achieved_median_bpp.append(float(np.median(values)))
        summary = {
            "status": "pass",
            "contract_version": contract["contract_version"],
            "contract_sha256": contract_sha256(),
            "seed": args.seed,
            "codec_sha256": codec_sha,
            "baa_cache_sha256": file_sha256(args.baa_cache),
            "baa_sender_state_dict_sha256": baa_metadata["sender_state_dict_sha256"],
            "target_bpp": target,
            "ratios": ratios,
            "calibration_history": history,
            "achieved_median_bpp": achieved_median_bpp,
            "relative_bpp_difference": [
                abs(achieved_median_bpp[index] - target[index]) / max(target[index], 1e-12)
                for index in range(4)
            ],
            "rows": final_count,
            "new_rows": inserted,
            "manifest_sha256": {name: file_sha256(path) for name, path in manifests.items()},
            "code_hash": files_sha256([Path(__file__).resolve()]),
            "codec_command": "opj_compress -r <four-ratios> -p LRCP -TP L -I -n 6",
        }
        connection.execute(
            "INSERT OR REPLACE INTO metadata(key, value) VALUES (?, ?)",
            ("action_cache", json.dumps(summary, sort_keys=True)),
        )
        connection.commit()
        connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        connection.close()
    summary["cache_sha256"] = file_sha256(args.cache)
    args.cache.with_suffix(args.cache.suffix + ".metadata.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(f"PASS: OpenJPEG action cache seed={args.seed} rows={summary['rows']}")


if __name__ == "__main__":
    main()
