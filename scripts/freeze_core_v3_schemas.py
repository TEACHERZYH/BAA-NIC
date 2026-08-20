from __future__ import annotations

import json

from core_v3_contract import ROOT, contract_sha256, file_sha256, load_contract


SCHEMA_PATH = ROOT / "configs" / "core_v3_result_schemas.json"
REQUIRED_SCHEMAS = {
    "core_v3_bitstream_roundtrip.csv",
    "core_v3_operating_points.csv",
    "core_v3_complexity_breakdown.csv",
    "core_v3_decoder_asymmetry.csv",
    "core_v3_progressive_ordering.csv",
    "core_v3_bandwidth_detail.csv",
    "core_v3_bandwidth_statistics.csv",
}


def main() -> None:
    contract = load_contract()
    document = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    if document.get("contract_version") != contract["contract_version"]:
        raise RuntimeError("Schema version differs from the core-v3 contract")
    available = set(document.get("schemas", {}))
    missing = REQUIRED_SCHEMAS - available
    if missing:
        raise RuntimeError(f"Missing core-v3 schemas: {sorted(missing)}")
    metadata = {
        "status": "frozen_before_formal_G0a",
        "contract_version": contract["contract_version"],
        "contract_sha256": contract_sha256(),
        "schema_sha256": file_sha256(SCHEMA_PATH),
        "schemas": sorted(available),
    }
    output = ROOT / "runs" / "core_v3_local" / "schema_freeze_metadata.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"PASS: frozen {len(available)} core-v3 schemas")


if __name__ == "__main__":
    main()
