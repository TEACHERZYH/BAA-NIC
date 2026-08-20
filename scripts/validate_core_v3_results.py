from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import re

from core_v3_contract import ROOT, contract_sha256, file_sha256, load_contract


SCHEMA_PATH = ROOT / "configs" / "core_v3_result_schemas.json"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="+", type=Path)
    args = parser.parse_args()

    contract = load_contract()
    schema_document = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    if schema_document.get("contract_version") != contract["contract_version"]:
        raise SystemExit("Schema and experiment contract versions differ")
    schemas = schema_document["schemas"]
    failures: list[str] = []
    failure_count = 0

    def add_failure(message: str) -> None:
        nonlocal failure_count
        failure_count += 1
        if len(failures) < 20:
            failures.append(message)

    total_rows = 0

    for path in args.files:
        schema = schemas.get(path.name)
        if schema is None:
            add_failure(f"{path}: no frozen schema")
            continue
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            fields = reader.fieldnames or []
            if fields != schema["fields"]:
                add_failure(f"{path}: field order/schema mismatch")
                continue
            pairing_key = schema["pairing_key"]
            keys: set[tuple[str, ...]] | None = set()
            bandwidth_bits: bytearray | None = None
            policy_ids: dict[str, int] = {}
            if path.name == "core_v3_bandwidth_detail.csv":
                keys = None
                # Exact bitset over the frozen seed/profile/realization/deadline/frame/policy domain.
                bandwidth_bits = bytearray((3 * 3 * 1000 * 3 * 300 * 32 + 7) // 8)
            row_count = 0
            duplicate_found = False
            for row_number, row in enumerate(reader, start=2):
                row_count += 1
                if keys is not None:
                    key = tuple(row[field] for field in pairing_key)
                    if key in keys:
                        duplicate_found = True
                    else:
                        keys.add(key)
                else:
                    try:
                        seed_index = [17, 29, 43].index(int(row["model_seed"]))
                        profile_index = list(contract["network_replay"]["profiles"]).index(row["profile"])
                        realization = int(row["realization_id"])
                        deadline_index = list(contract["network_replay"]["deadlines_ms"]).index(
                            int(row["deadline_ms"])
                        )
                        frame = int(row["frame_id"])
                        if not 0 <= realization < 1000 or not 0 <= frame < 300:
                            raise ValueError
                        if row["policy"] not in policy_ids:
                            if len(policy_ids) >= 32:
                                raise ValueError
                            policy_ids[row["policy"]] = len(policy_ids)
                        policy_index = policy_ids[row["policy"]]
                        flat = (
                            (((((seed_index * 3 + profile_index) * 1000 + realization) * 3 + deadline_index) * 300 + frame) * 32)
                            + policy_index
                        )
                        byte_index, bit_index = divmod(flat, 8)
                        mask = 1 << bit_index
                        if bandwidth_bits[byte_index] & mask:
                            duplicate_found = True
                        bandwidth_bits[byte_index] |= mask
                    except (ValueError, IndexError):
                        add_failure(f"{path}:{row_number}: value outside frozen bandwidth pairing domain")
                if row["contract_version"] != contract["contract_version"]:
                    add_failure(f"{path}:{row_number}: contract version mismatch")
                if row["contract_sha256"] != contract_sha256():
                    add_failure(f"{path}:{row_number}: contract hash mismatch")
                for field in schema["finite_fields"]:
                    value = row[field]
                    if value == "" and field == "noninferiority_margin":
                        continue
                    try:
                        finite = math.isfinite(float(value))
                    except ValueError:
                        finite = False
                    if not finite:
                        add_failure(f"{path}:{row_number}: non-finite {field}")
                for field in schema["sha256_fields"]:
                    if not SHA256_RE.fullmatch(row[field]):
                        add_failure(f"{path}:{row_number}: invalid SHA256 field {field}")
            if not row_count:
                add_failure(f"{path}: empty result file")
                continue
            if duplicate_found:
                add_failure(f"{path}: duplicate pairing key")
            total_rows += row_count
        print(f"CHECKED: {path} rows={row_count} schema_sha256={file_sha256(SCHEMA_PATH)[:12]}")

    if failure_count:
        print(f"FAIL: {failure_count} core-v3 result issue(s)")
        for failure in failures:
            print(f"- {failure}")
        raise SystemExit(1)
    print(f"PASS: validated {len(args.files)} file(s), rows={total_rows}")


if __name__ == "__main__":
    main()
