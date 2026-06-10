#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
import shutil
from contextlib import nullcontext
from pathlib import Path
import sys

import numpy as np
import torch
import yaml
from monai.data import DataLoader, Dataset, PersistentDataset, list_data_collate, load_decathlon_datalist
from monai.losses import DiceCELoss
from monai.transforms import (
    Compose,
    CropForegroundd,
    EnsureChannelFirstd,
    LoadImaged,
    Orientationd,
    RandCropByPosNegLabeld,
    RandFlipd,
    RandRotate90d,
    RandShiftIntensityd,
    ScaleIntensityRanged,
    Spacingd,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from gpaformer import SegFormer3D


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GPAFormer training entry point.")
    parser.add_argument("--config", required=True, help="Path to a YAML training config.")
    parser.add_argument("--data_root", default="", help="Override data.root from the config.")
    parser.add_argument("--datalist", default="", help="Override data.datalist from the config.")
    parser.add_argument("--output_dir", default="", help="Override training.output_dir from the config.")
    parser.add_argument("--resume", action="store_true", help="Resume from output_dir/last.pt if present.")
    parser.add_argument("--preflight", action="store_true", help="Load one batch and run one forward pass, then exit.")
    return parser.parse_args()


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = True


def section(cfg: dict, key: str) -> dict:
    value = cfg.get(key, {})
    if not isinstance(value, dict):
        raise TypeError(f"Config section '{key}' must be a mapping.")
    return value


def to_plain_tensor(value):
    if hasattr(value, "as_tensor"):
        return value.as_tensor()
    return value if torch.is_tensor(value) else torch.as_tensor(value)


def autocast_context(enabled: bool, device: torch.device):
    if not enabled:
        return nullcontext()
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        return torch.amp.autocast(device_type=device.type, enabled=True, dtype=torch.float16)
    return torch.cuda.amp.autocast(enabled=True, dtype=torch.float16)


def build_grad_scaler(enabled: bool):
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        try:
            return torch.amp.GradScaler("cuda", enabled=enabled)
        except TypeError:
            return torch.amp.GradScaler(enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def build_transforms(cfg: dict) -> Compose:
    pp = section(cfg, "preprocessing")
    roi_size = tuple(int(v) for v in pp.get("roi_size", [96, 96, 96]))
    spacing = tuple(float(v) for v in pp.get("spacing", [1.5, 1.5, 2.0]))
    a_min, a_max = (float(v) for v in pp.get("intensity_window", [-175.0, 250.0]))
    train_num_samples = int(pp.get("train_num_samples", 4))

    return Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(keys=["image", "label"], pixdim=spacing, mode=("bilinear", "nearest")),
            ScaleIntensityRanged(
                keys=["image"],
                a_min=a_min,
                a_max=a_max,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            CropForegroundd(keys=["image", "label"], source_key="image"),
            RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=roi_size,
                pos=1,
                neg=1,
                num_samples=train_num_samples,
                image_key="image",
                image_threshold=0,
            ),
            RandFlipd(keys=["image", "label"], spatial_axis=[0], prob=0.10),
            RandFlipd(keys=["image", "label"], spatial_axis=[1], prob=0.10),
            RandFlipd(keys=["image", "label"], spatial_axis=[2], prob=0.10),
            RandRotate90d(keys=["image", "label"], prob=0.10, max_k=3),
            RandShiftIntensityd(keys=["image"], offsets=0.10, prob=0.50),
        ]
    )


def build_model(cfg: dict, device: torch.device) -> SegFormer3D:
    model_cfg = section(cfg, "model")
    return SegFormer3D(
        in_channels=int(model_cfg.get("in_channels", 1)),
        sr_ratios=list(model_cfg.get("sr_ratios", [4, 2, 1])),
        embed_dims=list(model_cfg.get("embed_dims", [32, 64, 160])),
        patch_kernel_size=[tuple(v) if isinstance(v, list) else v for v in model_cfg.get("patch_kernel_size", [[7, 5, 3], 3, 3])],
        patch_stride=list(model_cfg.get("patch_stride", [2, 2, 2])),
        patch_padding=[tuple(v) if isinstance(v, list) else v for v in model_cfg.get("patch_padding", [[3, 2, 1], 1, 1])],
        mlp_ratios=list(model_cfg.get("mlp_ratios", [4, 4, 4])),
        num_heads=list(model_cfg.get("num_heads", [1, 2, 5])),
        depths=list(model_cfg.get("depths", [2, 2, 2])),
        decoder_head_embedding_dim=int(model_cfg.get("decoder_head_embedding_dim", 160)),
        num_classes=int(model_cfg.get("num_classes", 14)),
        decoder_dropout=float(model_cfg.get("decoder_dropout", 0.0)),
    ).to(device)


def split_batch(x: torch.Tensor, y: torch.Tensor, grad_accum_steps: int):
    actual_steps = max(1, min(int(grad_accum_steps), int(x.shape[0])))
    return list(zip(torch.chunk(x, actual_steps, dim=0), torch.chunk(y, actual_steps, dim=0)))


def save_checkpoint(path: Path, model, optimizer, scaler, step: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict() if scaler is not None else None,
            "global_step": int(step),
        },
        path,
    )


def load_checkpoint(path: Path, model, optimizer, scaler, device: torch.device) -> int:
    if not path.exists():
        return 0
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    if scaler is not None and ckpt.get("scaler") is not None:
        scaler.load_state_dict(ckpt["scaler"])
    step = int(ckpt.get("global_step", 0))
    print(f"[Resume] {path} step={step}", flush=True)
    return step


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    data_cfg = section(cfg, "data")
    train_cfg = section(cfg, "training")

    if args.data_root:
        data_cfg["root"] = args.data_root
    if args.datalist:
        data_cfg["datalist"] = args.datalist
    if args.output_dir:
        train_cfg["output_dir"] = args.output_dir

    set_seed(int(cfg.get("seed", 42)))
    data_root = Path(data_cfg["root"]).expanduser().resolve()
    datalist = Path(data_cfg["datalist"]).expanduser().resolve()
    list_key = str(data_cfg.get("list_key", "training"))
    output_dir = Path(train_cfg.get("output_dir", "runs/gpaformer")).expanduser().resolve()
    cache_dir = train_cfg.get("cache_dir", "")
    cache_dir = Path(cache_dir).expanduser().resolve() if cache_dir else None
    output_dir.mkdir(parents=True, exist_ok=True)
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.config, output_dir / "config.yaml")

    train_files = load_decathlon_datalist(
        str(datalist),
        is_segmentation=True,
        data_list_key=list_key,
        base_dir=str(data_root),
    )
    transforms = build_transforms(cfg)
    dataset_cls = PersistentDataset if cache_dir is not None else Dataset
    dataset_kwargs = {"data": train_files, "transform": transforms}
    if cache_dir is not None:
        dataset_kwargs["cache_dir"] = str(cache_dir)
    train_ds = dataset_cls(**dataset_kwargs)
    train_loader = DataLoader(
        train_ds,
        batch_size=int(train_cfg.get("batch_size", 1)),
        shuffle=True,
        num_workers=int(train_cfg.get("num_workers", 4)),
        pin_memory=torch.cuda.is_available(),
        collate_fn=list_data_collate,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = bool(train_cfg.get("amp", True) and device.type == "cuda")
    model = build_model(cfg, device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg.get("lr", 1e-4)),
        weight_decay=float(train_cfg.get("weight_decay", 1e-5)),
    )
    scaler = build_grad_scaler(enabled=use_amp)
    loss_fn = DiceCELoss(to_onehot_y=True, softmax=True)

    print(f"[Data] root={data_root}", flush=True)
    print(f"[Data] datalist={datalist} key={list_key} cases={len(train_files)}", flush=True)
    print(f"[Run] output_dir={output_dir}", flush=True)
    print(f"[Run] device={device} amp={use_amp}", flush=True)

    sanity = next(iter(train_loader))
    sanity_x = to_plain_tensor(sanity["image"]).to(device).float()
    sanity_y = to_plain_tensor(sanity["label"]).to(device).long()
    with torch.no_grad():
        sanity_logits = model(sanity_x)
        sanity_loss = loss_fn(sanity_logits, sanity_y)
    print(f"[Preflight] image={tuple(sanity_x.shape)} label={tuple(sanity_y.shape)} logits={tuple(sanity_logits.shape)} loss={float(sanity_loss):.5f}", flush=True)
    del sanity, sanity_x, sanity_y, sanity_logits, sanity_loss
    if args.preflight:
        return

    last_path = output_dir / "last.pt"
    global_step = load_checkpoint(last_path, model, optimizer, scaler, device) if args.resume else 0
    max_iters = int(train_cfg.get("max_iters", 100000))
    print_every = int(train_cfg.get("print_every", 50))
    save_every = int(train_cfg.get("save_every", 1000))
    grad_accum_steps = int(train_cfg.get("grad_accum_steps", 1))
    running_loss = 0.0

    while global_step < max_iters:
        model.train()
        for batch in train_loader:
            x = to_plain_tensor(batch["image"]).to(device, non_blocking=True).float()
            y = to_plain_tensor(batch["label"]).to(device, non_blocking=True).long()
            micro_batches = split_batch(x, y, grad_accum_steps)
            total_samples = sum(int(x_mb.shape[0]) for x_mb, _ in micro_batches)
            optimizer.zero_grad(set_to_none=True)

            step_loss = 0.0
            for x_mb, y_mb in micro_batches:
                weight = float(x_mb.shape[0]) / float(total_samples)
                with autocast_context(enabled=use_amp, device=device):
                    logits = model(x_mb)
                    loss = loss_fn(logits, y_mb)
                scaler.scale(loss * weight).backward()
                step_loss += float(loss.detach().cpu().item()) * weight
                del x_mb, y_mb, logits, loss

            scaler.step(optimizer)
            scaler.update()
            global_step += 1
            running_loss += step_loss
            del x, y, micro_batches

            if global_step % print_every == 0:
                print(f"step={global_step} loss={running_loss / print_every:.5f}", flush=True)
                running_loss = 0.0
            if global_step % save_every == 0:
                save_checkpoint(last_path, model, optimizer, scaler, global_step)
                torch.save(model.state_dict(), output_dir / "model_state_dict.pth")
            if global_step >= max_iters:
                break

    save_checkpoint(last_path, model, optimizer, scaler, global_step)
    torch.save(model.state_dict(), output_dir / "model_state_dict.pth")
    print(f"[Done] step={global_step}", flush=True)


if __name__ == "__main__":
    main()
