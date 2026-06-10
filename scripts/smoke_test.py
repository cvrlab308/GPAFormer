#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from gpaformer import SegFormer3D


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Construct GPAFormer and optionally run a forward smoke test.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--forward", action="store_true", help="Run a forward pass with a 96^3 dummy input.")
    parser.add_argument("--num_classes", type=int, default=14)
    return parser.parse_args()


def build_model(num_classes: int, device: torch.device) -> SegFormer3D:
    return SegFormer3D(
        in_channels=1,
        sr_ratios=[4, 2, 1],
        embed_dims=[32, 64, 160],
        patch_kernel_size=[(7, 5, 3), 3, 3],
        patch_stride=[2, 2, 2],
        patch_padding=[(3, 2, 1), 1, 1],
        mlp_ratios=[4, 4, 4],
        num_heads=[1, 2, 5],
        depths=[2, 2, 2],
        decoder_head_embedding_dim=160,
        num_classes=num_classes,
        decoder_dropout=0.0,
    ).to(device)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    model = build_model(args.num_classes, device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model_parameters={n_params}")

    if args.forward:
        model.eval()
        x = torch.randn(1, 1, 96, 96, 96, device=device)
        with torch.no_grad():
            y = model(x)
        print(f"input_shape={tuple(x.shape)}")
        print(f"output_shape={tuple(y.shape)}")


if __name__ == "__main__":
    main()
