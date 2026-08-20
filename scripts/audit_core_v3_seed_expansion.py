#!/usr/bin/env python3
"""Audit the five-seed decoder addendum without changing the base contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from core_v3_contract import (
    contract_sha256,
    load_contract,
    load_seed_expansion,
    registered_model_seeds,
    seed_expansion_sha256,
)
from core_v3_decoder_extension import (
    extension_contract_sha256,
    load_extension_contract,
    registered_extension_seeds,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--schedule",
        type=Path,
        default=ROOT / "configs" / "core_v3_action_schedules_5seed.json",
    )
    args = parser.parse_args()
    core = load_contract()
    extension = load_extension_contract()
    addendum = load_seed_expansion()
    if addendum is None:
        raise RuntimeError("CORE_V3_SEED_EXPANSION_PATH is required")
    combined = [17, 29, 43, 59, 71]
    checks = {
        "base_contract_unchanged": contract_sha256()
        == "82de33fe6659bd741d774251e01e7602e4460a32770790dc336b7a7ff199b26c",
        "extension_contract_unchanged": extension_contract_sha256()
        == "1148399a39b4a24357c253da6c624875c2262039061da9cf9f83bdf34197e05a",
        "core_seed_registration": registered_model_seeds(core) == combined,
        "extension_seed_registration": registered_extension_seeds(extension) == combined,
        "only_two_additional_seeds": addendum["additional_model_seeds"] == [59, 71],
        "frozen_protocol_change_scope": addendum["allowed_protocol_changes"]
        == ["model_seed_count", "hierarchical_bootstrap_seed_dimension"],
    }
    schedule = json.loads(args.schedule.read_text(encoding="utf-8"))
    checks["schedule_contract_match"] = schedule.get("contract_sha256") == contract_sha256()
    checks["schedule_addendum_match"] = (
        schedule.get("seed_expansion_sha256") == seed_expansion_sha256()
    )
    checks["schedule_seed_set"] = [int(value) for value in schedule["schedules"]] == combined
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise SystemExit(f"FAIL: five-seed addendum audit: {failed}")
    print("PASS: five-seed decoder addendum and action schedules")


if __name__ == "__main__":
    main()
