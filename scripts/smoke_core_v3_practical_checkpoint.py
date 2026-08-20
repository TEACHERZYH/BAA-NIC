"""Regression smoke for a selected core-v3 practical-bitstream checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models import BAANICModel
from models.baa_nic_bitstream import parse_stream, subset_stream
from core_v3_contract import contract_sha256, load_contract, sender_state_dict_sha256
from core_v3_data import CoreV3FixedTileDataset
from core_v3_training import configure_practical_determinism


def main() -> None:
    import torch

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    configure_practical_determinism(args.seed)
    contract = load_contract()
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if checkpoint.get("kind") != "core_v3_joint":
        raise ValueError("Expected a core-v3 joint checkpoint")
    if checkpoint["metadata"]["contract_sha256"] != contract_sha256():
        raise RuntimeError("Checkpoint belongs to another frozen contract")

    model = BAANICModel(use_entropy_model=True, entropy_model_type="group_hyperprior").to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    model.update_entropy_bottleneck(force=True)
    model.set_codec_metadata(sender_state_dict_sha256(model), contract_sha256())
    sample = CoreV3FixedTileDataset(args.manifest, limit=1, data_root=args.data_root)[0]
    image = sample["image"].unsqueeze(0).to(device)

    decoded_subsets = 0
    for q_index in contract["model"]["q_indices"]:
        encoded = model.compress(image, q_index=int(q_index))
        repeated = model.compress(image, q_index=int(q_index))
        if encoded["bitstream"] != repeated["bitstream"]:
            raise RuntimeError(f"Non-deterministic stream at q={q_index}")
        decoded_latent = model.decompress_latent(encoded["bitstream"])["masked_latent"]
        if not torch.equal(encoded["masked_latent"], decoded_latent):
            difference = float((encoded["masked_latent"] - decoded_latent).abs().max().item())
            raise RuntimeError(f"Entropy-decoded latent mismatch at q={q_index}: max_abs={difference}")
        _, segments = parse_stream(encoded["bitstream"])
        for mask_value in range(1, 16):
            group_ids = [index + 1 for index in range(4) if mask_value & (1 << index)]
            decoded = model.decompress(subset_stream(encoded["bitstream"], group_ids))
            if tuple(decoded["present_group_ids"]) != tuple(group_ids):
                raise RuntimeError(f"Subset decode mismatch at q={q_index}, mask={mask_value}")
            decoded_subsets += 1

    print(
        f"PASS: practical checkpoint seed={args.seed} q_points=4 "
        f"decoded_subsets={decoded_subsets} image={sample['metadata']['tile_id']}"
    )


if __name__ == "__main__":
    main()
