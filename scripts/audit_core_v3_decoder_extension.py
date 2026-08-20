"""Static readiness audit for the versioned deterministic decoder extension."""

from __future__ import annotations

import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models import ServerDecoder, build_enhanced_server_decoder
from core_v3_contract import contract_sha256
from core_v3_decoder_extension import extension_contract_sha256, load_extension_contract


def main() -> None:
    contract = load_extension_contract()
    model = build_enhanced_server_decoder(contract)
    baseline = ServerDecoder()
    enhanced_params = sum(parameter.numel() for parameter in model.parameters())
    baseline_params = sum(parameter.numel() for parameter in baseline.parameters())
    required_files = [
        ROOT / "experiments" / "core_v3_decoder_extension_plan_20260717.md",
        ROOT / "experiments" / "core_v3_decoder_extension_plan_review_20260717.md",
        ROOT / "models" / "core_v3_enhanced_decoder.py",
        ROOT / "scripts" / "train_core_v3_decoder_extension.py",
        ROOT / "scripts" / "evaluate_core_v3_decoder_extension.py",
        ROOT / "scripts" / "analyze_core_v3_decoder_extension.py",
        ROOT / "scripts" / "measure_core_v3_decoder_extension.py",
        ROOT / "scripts" / "slurm_core_v3_decoder_extension_benchmark.sh",
        ROOT / "scripts" / "slurm_core_v3_decoder_extension_pilot.sh",
        ROOT / "scripts" / "slurm_core_v3_decoder_extension_train.sh",
        ROOT / "scripts" / "slurm_core_v3_decoder_extension_quality.sh",
        ROOT / "scripts" / "slurm_core_v3_decoder_extension_analyze.sh",
    ]
    checks = {
        "parent_contract_matches": contract["parent_contract_sha256"] == contract_sha256(),
        "scope_is_decoder_only": contract["architecture"]["decoder_only"] is True,
        "bitstream_unchanged": contract["architecture"]["changes_bitstream"] is False,
        "sixteen_actions_preserved": len(contract["q_indices"]) * len(contract["prefixes"]) == 16,
        "three_seeds_preserved": contract["model_seeds"] == [17, 29, 43],
        "enhanced_is_heavier": enhanced_params > baseline_params,
        "all_required_files_present": all(path.is_file() for path in required_files),
    }
    status = "pass" if all(checks.values()) else "fail"
    output = {
        "status": status,
        "extension_version": contract["extension_version"],
        "extension_contract_sha256": extension_contract_sha256(),
        "parent_contract_sha256": contract_sha256(),
        "enhanced_params": enhanced_params,
        "baseline_heavy_params": baseline_params,
        "parameter_ratio": enhanced_params / baseline_params,
        "checks": checks,
    }
    output_dir = ROOT / "experiments" / "core_v3_decoder_extension"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "readiness_audit.json").write_text(
        json.dumps(output, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# Core-v3 Decoder Extension Readiness",
        "",
        f"- Status: `{'PASS' if status == 'pass' else 'FAIL'}`",
        f"- Extension: `{contract['extension_version']}`",
        f"- Parameter ratio over frozen heavy: `{enhanced_params / baseline_params:.4f}x`",
        "",
    ]
    lines.extend(f"- {name}: `{'PASS' if value else 'FAIL'}`" for name, value in checks.items())
    (output_dir / "readiness_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if status != "pass":
        raise SystemExit("FAIL: decoder extension readiness audit")
    print(f"PASS: decoder extension readiness ratio={enhanced_params / baseline_params:.4f}x")


if __name__ == "__main__":
    main()
