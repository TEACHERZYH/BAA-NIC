from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models import DenseControlEncoder, LightServerDecoder, LightweightEncoder, ServerDecoder
from core_v3_contract import contract_sha256, files_sha256, load_contract, state_dict_sha256


def profile(module: Any, inputs: tuple[Any, ...]) -> dict[str, Any]:
    import torch
    from torch import nn

    macs = 0
    executed: set[str] = set()
    handles = []

    def hook(name: str):
        def count(layer: Any, layer_inputs: tuple[Any, ...], output: Any) -> None:
            nonlocal macs
            executed.add(name)
            if isinstance(layer, nn.Conv2d):
                batch, out_channels, out_h, out_w = output.shape
                kernel_h, kernel_w = layer.kernel_size
                macs += (
                    batch
                    * out_channels
                    * out_h
                    * out_w
                    * (layer.in_channels // layer.groups)
                    * kernel_h
                    * kernel_w
                )
            elif isinstance(layer, nn.ConvTranspose2d):
                batch, in_channels, in_h, in_w = layer_inputs[0].shape
                kernel_h, kernel_w = layer.kernel_size
                macs += (
                    batch
                    * in_channels
                    * in_h
                    * in_w
                    * (layer.out_channels // layer.groups)
                    * kernel_h
                    * kernel_w
                )
            elif isinstance(layer, nn.Linear):
                macs += layer_inputs[0].numel() // layer.in_features * layer.in_features * layer.out_features

        return count

    parameterized_leaves: set[str] = set()
    for name, layer in module.named_modules():
        if any(True for _ in layer.children()):
            continue
        if any(parameter.numel() for parameter in layer.parameters(recurse=False)):
            parameterized_leaves.add(name)
            handles.append(layer.register_forward_hook(hook(name)))
    module.eval()
    with torch.no_grad():
        output = module(*inputs)
    for handle in handles:
        handle.remove()
    missing = sorted(parameterized_leaves - executed)
    if missing:
        raise RuntimeError(f"Parameterized leaf modules were not executed: {missing}")
    return {
        "params": sum(parameter.numel() for parameter in module.parameters()),
        "macs": int(macs),
        "output_shape": list(output.shape),
        "executed_parameterized_leaves": len(executed),
        "architecture_sha256": state_dict_sha256(module),
    }


def main() -> None:
    import torch

    contract = load_contract()
    torch.manual_seed(20260715)
    image = torch.zeros((1, 3, 512, 512))
    latent = torch.zeros((1, 192, 32, 32))
    indicator = torch.ones((1, 4, 32, 32))
    profiles = {
        "baa_depthwise_encoder": profile(LightweightEncoder(), (image,)),
        "dense_control_encoder": profile(DenseControlEncoder(), (image,)),
        "server_heavy_decoder": profile(ServerDecoder(), (latent, indicator, (512, 512))),
        "symmetric_light_decoder": profile(LightServerDecoder(), (latent, indicator, (512, 512))),
    }
    encoder = profiles["baa_depthwise_encoder"]
    dense = profiles["dense_control_encoder"]
    heavy = profiles["server_heavy_decoder"]
    light = profiles["symmetric_light_decoder"]
    checks = {
        "encoder_params_reduction_at_least_50pct": encoder["params"] <= 0.5 * dense["params"],
        "encoder_macs_reduction_at_least_50pct": encoder["macs"] <= 0.5 * dense["macs"],
        "light_params_within_20pct_encoder": abs(light["params"] / encoder["params"] - 1.0) <= 0.2,
        "light_macs_within_10pct_encoder": abs(light["macs"] / encoder["macs"] - 1.0) <= 0.1,
        "heavy_params_at_least_2x_light": heavy["params"] >= 2.0 * light["params"],
        "heavy_macs_at_least_2x_light": heavy["macs"] >= 2.0 * light["macs"],
    }
    metadata = {
        "status": "pass" if all(checks.values()) else "fail",
        "contract_version": contract["contract_version"],
        "contract_sha256": contract_sha256(),
        "architecture_code_sha256": files_sha256(
            [ROOT / "models" / "baa_nic.py", ROOT / "models" / "core_v3_controls.py"]
        ),
        "profiles": profiles,
        "ratios": {
            "encoder_to_dense_params": encoder["params"] / dense["params"],
            "encoder_to_dense_macs": encoder["macs"] / dense["macs"],
            "light_to_encoder_params": light["params"] / encoder["params"],
            "light_to_encoder_macs": light["macs"] / encoder["macs"],
            "heavy_to_light_params": heavy["params"] / light["params"],
            "heavy_to_light_macs": heavy["macs"] / light["macs"],
        },
        "checks": checks,
    }
    output = ROOT / "runs" / "core_v3_local" / "architecture_freeze.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise SystemExit(f"FAIL: core-v3 architecture checks: {failed}")
    print("PASS: core-v3 dense/light/heavy architecture thresholds")


if __name__ == "__main__":
    main()
