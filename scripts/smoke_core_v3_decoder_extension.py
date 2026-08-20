"""Module-level smoke test for the deterministic enhanced server decoder."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys


if importlib.util.find_spec("torch") is None:
    print("SKIP: torch is not installed")
    raise SystemExit(0)

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from models import ServerDecoder, build_enhanced_server_decoder
from core_v3_decoder_extension import load_extension_contract


def indicator(prefix_k: int, device: torch.device) -> torch.Tensor:
    value = torch.zeros((1, 4, 4, 4), device=device)
    value[:, :prefix_k] = 1.0
    return value


def main() -> None:
    torch.manual_seed(20260717)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    contract = load_extension_contract()
    model = build_enhanced_server_decoder(contract).to(device)
    baseline = ServerDecoder().to(device)
    enhanced_params = sum(parameter.numel() for parameter in model.parameters())
    baseline_params = sum(parameter.numel() for parameter in baseline.parameters())
    if enhanced_params <= baseline_params:
        raise AssertionError("enhanced decoder must be heavier than the frozen core-v3 decoder")

    latent = torch.randn((1, 192, 4, 4), device=device, requires_grad=True)
    model.eval()
    with torch.no_grad():
        first = model(latent, indicator(1, device), q_index=0, output_size=(64, 64))
        second = model(latent, indicator(1, device), q_index=0, output_size=(64, 64))
        full = model(latent, indicator(4, device), q_index=3, output_size=(64, 64))
    if first.shape != (1, 3, 64, 64) or full.shape != first.shape:
        raise AssertionError("unexpected enhanced decoder output shape")
    if not torch.equal(first, second):
        raise AssertionError("enhanced decoder forward is not deterministic")
    if float(first.min()) < 0.0 or float(first.max()) > 1.0:
        raise AssertionError("enhanced decoder output is outside [0, 1]")

    model.train()
    output = model(latent, indicator(2, device), q_index=2, output_size=(64, 64))
    loss = output.square().mean()
    loss.backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    if not gradients or not all(bool(torch.isfinite(value).all()) for value in gradients):
        raise AssertionError("enhanced decoder backward produced missing or non-finite gradients")

    print(
        json.dumps(
            {
                "status": "pass",
                "device": str(device),
                "torch": torch.__version__,
                "enhanced_params": enhanced_params,
                "baseline_heavy_params": baseline_params,
                "parameter_ratio": enhanced_params / baseline_params,
                "shape": list(first.shape),
                "range": [float(first.min()), float(first.max())],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
