#!/usr/bin/env python3
"""Initialize a compact evaluation-only cache from frozen sender metadata."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sqlite3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-cache", type=Path, required=True)
    parser.add_argument("--output-cache", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    return parser.parse_args()


def read_metadata(connection: sqlite3.Connection) -> dict[str, object]:
    return {key: json.loads(value) for key, value in connection.execute("SELECT key, value FROM metadata")}


def main() -> None:
    args = parse_args()
    source = sqlite3.connect(f"file:{args.source_cache.as_posix()}?mode=ro", uri=True)
    source.execute("PRAGMA query_only=ON")
    metadata_rows = list(source.execute("SELECT key, value FROM metadata"))
    source_metadata = {key: json.loads(value) for key, value in metadata_rows}
    source.close()
    required = ("contract_sha256", "sender_state_dict_sha256", "latent_storage_dtype", "seed")
    missing = [key for key in required if key not in source_metadata]
    if missing:
        raise RuntimeError(f"Source sender cache is missing metadata keys: {missing}")
    if int(source_metadata["seed"]) != args.seed:
        raise RuntimeError("Source sender cache belongs to another seed")
    if source_metadata["latent_storage_dtype"] != "float32_exact_entropy_decode":
        raise RuntimeError("Source sender cache does not contain exact float32 decoded latents")

    args.output_cache.parent.mkdir(parents=True, exist_ok=True)
    output = sqlite3.connect(args.output_cache)
    output.execute("CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
    existing = read_metadata(output)
    if existing:
        for key in required:
            if existing.get(key) != source_metadata[key]:
                raise RuntimeError("Existing evaluation-only cache has incompatible provenance")
        print(f"PASS: reusable evaluation-only cache seed={args.seed} path={args.output_cache}")
        output.close()
        return

    for key, value in metadata_rows:
        if key == "evaluation_cache":
            continue
        output.execute("INSERT INTO metadata(key, value) VALUES (?, ?)", (key, value))
    output.execute(
        "INSERT OR REPLACE INTO metadata(key, value) VALUES (?, ?)",
        ("evaluation_only_cache", json.dumps(True)),
    )
    output.execute(
        "INSERT OR REPLACE INTO metadata(key, value) VALUES (?, ?)",
        ("evaluation_source_cache", json.dumps(str(args.source_cache))),
    )
    output.commit()
    output.close()
    print(f"PASS: initialized evaluation-only cache seed={args.seed} path={args.output_cache}")


if __name__ == "__main__":
    main()
