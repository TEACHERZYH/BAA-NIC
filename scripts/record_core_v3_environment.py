from __future__ import annotations

import importlib.metadata
import json
import platform
from pathlib import Path
import shutil
import subprocess
import sys

from core_v3_contract import contract_sha256, load_contract


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "runs" / "core_v3_local"


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def command_output(command: list[str]) -> str | None:
    if shutil.which(command[0]) is None:
        return None
    completed = subprocess.run(command, check=False, capture_output=True, text=True, timeout=30)
    text = (completed.stdout or completed.stderr).strip()
    return text if text else None


def main() -> None:
    import torch

    contract = load_contract()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    packages = [
        "torch",
        "torchvision",
        "compressai",
        "numpy",
        "pillow",
        "opencv-python",
        "scikit-image",
        "lpips",
        "pytorch-msssim",
    ]
    metadata = {
        "contract_version": contract["contract_version"],
        "contract_sha256": contract_sha256(),
        "python_executable": sys.executable,
        "python_version": sys.version,
        "python_compiler": platform.python_compiler(),
        "platform": platform.platform(),
        "packages": {name: package_version(name) for name in packages},
        "torch_cuda_runtime": torch.version.cuda,
        "torch_cuda_available": torch.cuda.is_available(),
        "torch_cudnn_version": torch.backends.cudnn.version(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "gpu_capability": torch.cuda.get_device_capability(0) if torch.cuda.is_available() else None,
        "nvcc": command_output(["nvcc", "--version"]),
    }
    output = OUTPUT_DIR / "environment.json"
    output.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    freeze = subprocess.run(
        [sys.executable, "-m", "pip", "freeze"],
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    ).stdout
    (OUTPUT_DIR / "pip_freeze.txt").write_text(freeze, encoding="utf-8")
    print(f"PASS: wrote {output.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
