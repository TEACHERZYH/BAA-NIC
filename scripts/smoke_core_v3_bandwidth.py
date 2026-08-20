"""Synthetic end-to-end smoke for the frozen core-v3 E4 replay."""

from __future__ import annotations

import json
from pathlib import Path
import sqlite3
import subprocess
import sys
from tempfile import TemporaryDirectory


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core_v3_contract import contract_sha256


def make_cache(path: Path, q_count: int, codec: str) -> None:
    connection = sqlite3.connect(path)
    connection.executescript(
        """
        CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE TABLE action_entries (
            dataset TEXT, image_index INTEGER, image_id TEXT, sequence_id TEXT,
            q_index INTEGER, prefix_k INTEGER, stream_sha256 TEXT, payload_bytes INTEGER,
            network_bytes INTEGER, bpp REAL, psnr REAL, ms_ssim REAL, encode_ms REAL,
            entropy_decode_ms REAL, decoder_ms REAL, decode_ms REAL,
            PRIMARY KEY(dataset, image_index, q_index, prefix_k)
        );
        """
    )
    for dataset, images in (("development", 256), ("visdrone", 512)):
        for image_index in range(images):
            for q_index in range(q_count):
                for prefix_k in range(1, 5):
                    payload = 400 + prefix_k * 600 - q_index * 50 + image_index % 11
                    network = payload + ((payload + 1459) // 1460) * 40
                    quality = min(0.999, 0.60 + 0.08 * prefix_k - 0.01 * q_index)
                    connection.execute(
                        "INSERT INTO action_entries VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                        (
                            dataset,
                            image_index,
                            f"{dataset}_{image_index:04d}",
                            "seq0",
                            q_index,
                            prefix_k,
                            "a" * 64,
                            payload,
                            network,
                            payload * 8.0 / (512 * 512),
                            25.0 + prefix_k,
                            quality,
                            2.0 + q_index * 0.1,
                            0.5,
                            1.0 + prefix_k * 0.1,
                            1.5 + prefix_k * 0.1,
                        ),
                    )
    action_meta = {
        "status": "pass",
        "contract_sha256": contract_sha256(),
        "sender_state_dict_sha256": "1" * 64,
        "heavy_decoder_state_dict_sha256": "2" * 64,
        "codec_sha256": "3" * 64,
    }
    connection.execute(
        "INSERT INTO metadata VALUES (?, ?)", ("action_cache", json.dumps(action_meta))
    )
    connection.commit()
    connection.close()


def main() -> None:
    traces = ROOT / "runs" / "core_v3_local" / "traces"
    if not (traces / "test_traces.npz").is_file():
        print("SKIP: core-v3 trace smoke assets are unavailable")
        return
    with TemporaryDirectory(prefix="core_v3_bandwidth_smoke_") as temporary_name:
        temporary = Path(temporary_name)
        baa_cache = temporary / "baa.sqlite"
        openjpeg_cache = temporary / "openjpeg.sqlite"
        output = temporary / "output"
        make_cache(baa_cache, 4, "baa")
        make_cache(openjpeg_cache, 1, "openjpeg")
        completed = subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "evaluate_core_v3_bandwidth.py"),
                "--seed", "17",
                "--cache", str(baa_cache),
                "--openjpeg-cache", str(openjpeg_cache),
                "--traces-dir", str(traces),
                "--output-dir", str(output),
                "--mode", "smoke",
                "--smoke-realizations", "1",
                "--omit-detail",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError((completed.stdout + "\n" + completed.stderr)[-2000:])
        metadata = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
        if metadata["summary_rows"] != 81 or metadata["detail_rows"] != 0:
            raise RuntimeError("Unexpected synthetic E4 row count")
    print("PASS: synthetic core-v3 bandwidth replay produced all 9 paired policies")


if __name__ == "__main__":
    main()
