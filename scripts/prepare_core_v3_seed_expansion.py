#!/usr/bin/env python3
"""Generate the versioned action schedule for an authorized seed expansion."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from core_v3_contract import (
    contract_sha256,
    load_contract,
    registered_model_seeds,
    seed_expansion_sha256,
)
from prepare_core_v3_manifests import build_action_schedules


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "configs" / "core_v3_action_schedules_5seed.json",
    )
    args = parser.parse_args()
    expansion_hash = seed_expansion_sha256()
    if expansion_hash is None:
        raise RuntimeError("CORE_V3_SEED_EXPANSION_PATH is required")
    contract = copy.deepcopy(load_contract())
    contract["training"]["model_seeds"] = registered_model_seeds(contract)
    payload = build_action_schedules(contract, contract_sha256())
    payload["seed_expansion_sha256"] = expansion_hash
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"PASS: five-seed action schedule seeds={sorted(payload['schedules'])}")


if __name__ == "__main__":
    main()
