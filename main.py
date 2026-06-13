import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast

from dataset.dataset_loader import ARADDataset
from model.MST_Plus_Plus import MST_Plus_Plus


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def crop_pair(rgb: torch.Tensor, hsi: torch.Tensor, patch_size: int, random_crop: bool) -> tuple[torch.Tensor, torch.Tensor]:
    """Crop RGB/HSI tensors with the same crop coordinates.

    Expected shapes:
        rgb: [B, 3, H, W]
        hsi: [B, 31, H, W]
    """
    if patch_size is None or patch_size <= 0:
        return rgb, hsi

    _, _, h, w = rgb.shape
    if patch_size >= min(h, w):
        return rgb, hsi

    if random_crop:
        top = random.randint(0, h - patch_size)
        left = random.randint(0, w - patch_size)
    else:
        top = (h - patch_size) // 2
        left = (w - patch_size) // 2

    rgb = rgb[:, :, top:top + patch_size, left:left + patch_size]
    hsi = hsi[:, :, top:top + patch_size, left:left + patch_size]
    return rgb, hsi


def mrae_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    return torch.mean(torch.abs(pred - target) / torch.clamp(torch.abs(target), min=eps))


def sam_metric(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Spectral angle mapper in degrees."""
    pred_f = pred.permute(0, 2, 3, 1).reshape(-1, pred.shape[1])
    target_f = target.permute(0, 2, 3, 1).reshape(-1, target.shape[1])
    numerator = torch.sum(pred_f * target_f, dim=1)
    denominator = torch.norm(pred_f, dim=1) * torch.norm(target_f, dim=1)
    cosine = torch.clamp(numerator / torch.clamp(denominator, min=eps), -1.0, 1.0)
    return torch.mean(torch.acos(cosine)) * (180.0 / torch.pi)


def psnr_metric(pred: torch.Tensor, target: torch.Tensor, max_val: float = 1.0, eps: float = 1e-10) -> torch.Tensor:
    mse = torch.mean((pred - target) ** 2)
    return 20.0 * torch.log10(torch.tensor(max_val, device=pred.device)) - 10.0 * torch.log10(torch.clamp(mse, min=eps))


def compute_loss(pred: torch.Tensor, target: torch.Tensor, args: argparse.Namespace) -> torch.Tensor:
    if args.loss == "l1":
        return F.l1_loss(pred, target)
    if args.loss == "mrae":
        return mrae_loss(pred, target, eps=args.mrae_eps)
    if args.loss == "l1_mrae":
        return F.l1_loss(pred, target) + args.mrae_weight * mrae_loss(pred, target, eps=args.mrae_eps)
    raise ValueError(f"Unknown loss: {args.loss}")


@torch.no_grad()
def validate(model: torch.nn.Module, loader: DataLoader, device: torch.device, args: argparse.Namespace) -> dict[str, float]:
    model.eval()

    total_loss = 0.0
    total_mrae = 0.0
    total_rmse = 0.0
    total_psnr = 0.0
    total_sam = 0.0
    total_samples = 0

    for rgb, hsi in loader:
        rgb = rgb.to(device, non_blocking=True)
        hsi = hsi.to(device, non_blocking=True)

        if not args.eval_full:
            rgb, hsi = crop_pair(rgb, hsi, args.patch_size, random_crop=False)

        pred = model(rgb)

        metric_pred = pred.clamp(0.0, args.data_range) if args.clamp_metrics else pred
        metric_hsi = hsi.clamp(0.0, args.data_range) if args.clamp_metrics else hsi

        batch_size = rgb.shape[0]
        loss = compute_loss(pred, hsi, args)
        mrae = mrae_loss(metric_pred, metric_hsi, eps=args.mrae_eps)
        rmse = torch.sqrt(torch.mean((metric_pred - metric_hsi) ** 2))
        psnr = psnr_metric(metric_pred, metric_hsi, max_val=args.data_range)
        sam = sam_metric(metric_pred, metric_hsi)

        total_loss += loss.item() * batch_size
        total_mrae += mrae.item() * batch_size
        total_rmse += rmse.item() * batch_size
        total_psnr += psnr.item() * batch_size
        total_sam += sam.item() * batch_size
        total_samples += batch_size

    total_samples = max(total_samples, 1)
    return {
        "loss": total_loss / total_samples,
        "mrae": total_mrae / total_samples,
        "rmse": total_rmse / total_samples,
        "psnr": total_psnr / total_samples,
        "sam": total_sam / total_samples,
    }


def train_one_epoch(model: torch.nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer, scaler: GradScaler, device: torch.device, args: argparse.Namespace) -> float:
    model.train()
    total_loss = 0.0
    total_samples = 0

    optimizer.zero_grad(set_to_none=True)

    for step, (rgb, hsi) in enumerate(loader, start=1):
        rgb = rgb.to(device, non_blocking=True)
        hsi = hsi.to(device, non_blocking=True)

        rgb, hsi = crop_pair(rgb, hsi, args.patch_size, random_crop=True)

        with autocast(enabled=args.amp and device.type == "cuda"):
            pred = model(rgb)
            loss = compute_loss(pred, hsi, args) / args.grad_accum_steps

        if args.amp and device.type == "cuda":
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if step % args.grad_accum_steps == 0 or step == len(loader):
            if args.grad_clip > 0:
                if args.amp and device.type == "cuda":
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            if args.amp and device.type == "cuda":
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            optimizer.zero_grad(set_to_none=True)

        batch_size = rgb.shape[0]
        total_loss += loss.item() * args.grad_accum_steps * batch_size
        total_samples += batch_size

    return total_loss / max(total_samples, 1)


def save_checkpoint(path: Path, epoch: int, model: torch.nn.Module, optimizer: torch.optim.Optimizer, scheduler, best_mrae: float, args: argparse.Namespace) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "best_mrae": best_mrae,
            "args": vars(args),
        },
        path,
    )


def main() -> None:
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_set = ARADDataset(
        root_dir=args.data_root,
        train=True,
        train_images=args.train_images,
        total_images=args.total_images,
        cube_key=args.cube_key,
        download=args.download,
    )
    val_set = ARADDataset(
        root_dir=args.data_root,
        train=False,
        train_images=args.train_images,
        total_images=args.total_images,
        cube_key=args.cube_key,
        download=False,
    )

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=1 if args.eval_full else args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    model = MST_Plus_Plus(in_channels=3, out_channels=31, n_feat=31, stage=args.model_stage).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 1e-2)
    scaler = GradScaler(enabled=args.amp and device.type == "cuda")

    start_epoch = 1
    best_mrae = float("inf")

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        if ckpt.get("scheduler") is not None:
            scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        best_mrae = float(ckpt.get("best_mrae", best_mrae))
        print(f"Resumed from {args.resume} at epoch {start_epoch}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(start_epoch, args.epochs + 1):
        start_time = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, scaler, device, args)
        val_stats = validate(model, val_loader, device, args)
        scheduler.step()

        lr_now = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - start_time

        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"Train Loss {train_loss:.6f} | "
            f"Val Loss {val_stats['loss']:.6f} | "
            f"Val MRAE {val_stats['mrae']:.6f} | "
            f"Val RMSE {val_stats['rmse']:.6f} | "
            f"Val PSNR {val_stats['psnr']:.4f} | "
            f"Val SAM {val_stats['sam']:.4f} | "
            f"LR {lr_now:.2e} | "
            f"Time {elapsed:.1f}s"
        )

        save_checkpoint(out_dir / "latest.pth", epoch, model, optimizer, scheduler, best_mrae, args)

        if val_stats["mrae"] < best_mrae:
            best_mrae = val_stats["mrae"]
            save_checkpoint(out_dir / "best.pth", epoch, model, optimizer, scheduler, best_mrae, args)
            print(f"Best model saved with Val MRAE {best_mrae:.6f}")

        if args.save_every > 0 and epoch % args.save_every == 0:
            save_checkpoint(out_dir / f"epoch_{epoch:03d}.pth", epoch, model, optimizer, scheduler, best_mrae, args)


if __name__ == "__main__":
    main()
