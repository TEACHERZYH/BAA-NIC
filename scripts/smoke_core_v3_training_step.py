from __future__ import annotations

import json
import math
import os
from pathlib import Path
import sys


os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models import BAANICModel
from core_v3_contract import contract_sha256, load_contract
from core_v3_data import CoreV3FixedTileDataset


def main() -> None:
    import torch
    import torch.nn.functional as F

    contract = load_contract()
    seed = int(contract["training"]["model_seeds"][0])
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = CoreV3FixedTileDataset(ROOT / "data" / "core_v3_visdrone_development.csv", limit=1)
    image = dataset[0]["image"].unsqueeze(0).to(device)

    model = BAANICModel(use_entropy_model=True, entropy_model_type="group_hyperprior").to(device)
    named_parameters = list(model.named_parameters())
    aux_parameters = [parameter for name, parameter in named_parameters if name.endswith(".quantiles")]
    main_parameters = [parameter for name, parameter in named_parameters if not name.endswith(".quantiles")]
    if not aux_parameters:
        raise RuntimeError("No entropy quantile parameters found")
    optimizer = torch.optim.AdamW(
        main_parameters,
        lr=contract["training"]["joint"]["initial_lr"],
        weight_decay=contract["training"]["joint"]["weight_decay"],
    )
    aux_optimizer = torch.optim.Adam(aux_parameters, lr=contract["training"]["joint"]["aux_lr"])

    model.train()
    optimizer.zero_grad(set_to_none=True)
    with torch.no_grad():
        full_reference = model(image, active_ratio=1.0, q_index=1)["reconstruction"].detach()
    output = model(image, active_ratio=0.5, q_index=1)
    reconstruction = output["reconstruction"]
    l1 = F.l1_loss(reconstruction, image)
    mse = F.mse_loss(reconstruction, image)
    cross_rate = F.l1_loss(reconstruction, full_reference)
    rate = output["rate_proxy"]
    loss = l1 + mse + 0.01 * rate + 0.05 * cross_rate
    if not bool(torch.isfinite(loss)):
        raise RuntimeError("Non-finite main smoke loss")
    loss.backward()
    finite_gradients = [
        bool(torch.isfinite(parameter.grad).all())
        for parameter in main_parameters
        if parameter.grad is not None
    ]
    if not finite_gradients or not all(finite_gradients):
        raise RuntimeError("Missing or non-finite main gradients")
    grad_norm = torch.nn.utils.clip_grad_norm_(main_parameters, max_norm=1.0)
    optimizer.step()

    aux_optimizer.zero_grad(set_to_none=True)
    aux_loss = model.aux_loss()
    if not bool(torch.isfinite(aux_loss)):
        raise RuntimeError("Non-finite entropy auxiliary loss")
    aux_loss.backward()
    if not all(parameter.grad is not None and bool(torch.isfinite(parameter.grad).all()) for parameter in aux_parameters):
        raise RuntimeError("Missing or non-finite entropy auxiliary gradients")
    aux_optimizer.step()

    coverage: list[dict[str, float | int]] = []
    model.eval()
    with torch.no_grad():
        for q_index in contract["model"]["q_indices"]:
            for active_ratio in contract["model"]["active_ratios"]:
                result = model(image, active_ratio=float(active_ratio), q_index=int(q_index))
                reconstruction = result["reconstruction"]
                bpp = float(result["rate_proxy"].detach().cpu())
                if tuple(reconstruction.shape) != tuple(image.shape) or not math.isfinite(bpp):
                    raise RuntimeError(f"Invalid action output for q={q_index}, ratio={active_ratio}")
                coverage.append({"q_index": q_index, "active_ratio": active_ratio, "likelihood_bpp": bpp})
    if len(coverage) != 16:
        raise RuntimeError(f"Expected 16 actions, got {len(coverage)}")

    output_path = ROOT / "runs" / "core_v3_local" / "training_smoke.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": "pass",
        "contract_version": contract["contract_version"],
        "contract_sha256": contract_sha256(),
        "device": str(device),
        "input_shape": list(image.shape),
        "loss": float(loss.detach().cpu()),
        "l1": float(l1.detach().cpu()),
        "mse": float(mse.detach().cpu()),
        "rate": float(rate.detach().cpu()),
        "cross_rate": float(cross_rate.detach().cpu()),
        "aux_loss": float(aux_loss.detach().cpu()),
        "grad_norm": float(grad_norm.detach().cpu()),
        "actions_covered": len(coverage),
        "coverage": coverage,
    }
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"PASS: one backward step and {len(coverage)} core-v3 actions")


if __name__ == "__main__":
    main()
