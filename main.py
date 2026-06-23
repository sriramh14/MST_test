import os
import time
import random

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler

from dataset.dataset_loader import ARADDataset

#Change file to MST_Plus_Plus or mst_plus_plus_cross_attn
from model.mst_plus_plus_cross_attn import MST_Plus_Plus
from loss import AverageMeter, Loss_MRAE, Loss_RMSE, Loss_PSNR, save_checkpoint


# ==========================================================
# Fixed training settings
# Edit these values here. Run with: python main.py
# ==========================================================

DATA_ROOT = "data"
DOWNLOAD_DATA = True

TOTAL_IMAGES = 230
TRAIN_IMAGES = 200
CUBE_KEY = "cube"

EPOCHS = 100
BATCH_SIZE = 2
NUM_WORKERS = 2
LR = 4e-4
WEIGHT_DECAY = 0.0

PATCH_SIZE = 128          # use 256 for full-image training
USE_RANDOM_CROP = True

MODEL_STAGE = 3
USE_AMP = True
GRAD_CLIP = 1.0
SEED = 42

SAVE_DIR = "checkpoints_mstpp"
RESUME_PATH = ""         # example: "checkpoints_mstpp/best.pth"
BEST_MODEL_NAME = "best.pth"
LATEST_MODEL_NAME = "latest.pth"


# ==========================================================
# Utilities
# ==========================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def crop_pair(rgb, hsi, patch_size, random_crop=True):
    if patch_size is None or patch_size <= 0:
        return rgb, hsi

    _, _, h, w = rgb.shape

    if patch_size >= h or patch_size >= w:
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


def save_training_state(path, epoch, model, optimizer, scheduler, best_mrae):
    torch.save(
        {
            "epoch": epoch,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_mrae": best_mrae,
        },
        path,
    )


def load_training_state(path, model, optimizer, scheduler, device):
    checkpoint = torch.load(path, map_location=device)

    if "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    elif "model" in checkpoint:
        model.load_state_dict(checkpoint["model"])
    else:
        model.load_state_dict(checkpoint)

    if isinstance(checkpoint, dict) and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])

    if isinstance(checkpoint, dict) and "scheduler" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler"])

    start_epoch = 1
    best_mrae = float("inf")

    if isinstance(checkpoint, dict):
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        best_mrae = float(checkpoint.get("best_mrae", float("inf")))

    return start_epoch, best_mrae


# ==========================================================
# Train / validation
# ==========================================================

def train_one_epoch(model, loader, criterion, optimizer, scaler, device):
    model.train()
    loss_meter = AverageMeter()

    for rgb, hsi in loader:
        rgb = rgb.to(device, non_blocking=True)
        hsi = hsi.to(device, non_blocking=True)

        if USE_RANDOM_CROP:
            rgb, hsi = crop_pair(
                rgb,
                hsi,
                PATCH_SIZE,
                random_crop=True,
            )

        optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=USE_AMP and device.type == "cuda"):
            pred = model(rgb)

            # loss.py uses .view(-1), which requires contiguous tensors.
            # Keep loss.py unchanged and make tensors contiguous here.
            loss = criterion(pred.contiguous(), hsi.contiguous())

        if USE_AMP and device.type == "cuda":
            scaler.scale(loss).backward()

            if GRAD_CLIP is not None and GRAD_CLIP > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)

            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()

            if GRAD_CLIP is not None and GRAD_CLIP > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)

            optimizer.step()

        loss_meter.update(loss.item(), rgb.size(0))

    return loss_meter.avg


@torch.no_grad()
def validate(model, loader, mrae_fn, rmse_fn, psnr_fn, device):
    model.eval()

    mrae_meter = AverageMeter()
    rmse_meter = AverageMeter()
    psnr_meter = AverageMeter()

    for rgb, hsi in loader:
        rgb = rgb.to(device, non_blocking=True)
        hsi = hsi.to(device, non_blocking=True)

        rgb, hsi = crop_pair(
            rgb,
            hsi,
            PATCH_SIZE,
            random_crop=False,
        )

        pred = model(rgb)
        batch_size = rgb.size(0)

        # Metrics also call the repo loss functions, so keep tensors contiguous.
        pred_for_loss = pred.contiguous()
        hsi_for_loss = hsi.contiguous()

        mrae = mrae_fn(pred_for_loss, hsi_for_loss)
        rmse = rmse_fn(pred_for_loss, hsi_for_loss)
        psnr = psnr_fn(hsi_for_loss.clone(), pred_for_loss.clone())

        mrae_meter.update(mrae.item(), batch_size)
        rmse_meter.update(rmse.item(), batch_size)
        psnr_meter.update(psnr.item(), batch_size)

    return {
        "mrae": mrae_meter.avg,
        "rmse": rmse_meter.avg,
        "psnr": psnr_meter.avg,
    }


# ==========================================================
# Main
# ==========================================================

def main():
    set_seed(SEED)
    os.makedirs(SAVE_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_dataset = ARADDataset(
        root_dir=DATA_ROOT,
        train=True,
        train_images=TRAIN_IMAGES,
        total_images=TOTAL_IMAGES,
        cube_key=CUBE_KEY,
        download=DOWNLOAD_DATA,
    )

    val_dataset = ARADDataset(
        root_dir=DATA_ROOT,
        train=False,
        train_images=TRAIN_IMAGES,
        total_images=TOTAL_IMAGES,
        cube_key=CUBE_KEY,
        download=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    model = MST_Plus_Plus(
        in_channels=3,
        out_channels=31,
        n_feat=31,
        stage=MODEL_STAGE,
    ).to(device)

    # This is the loss function already defined in loss.py.
    criterion = Loss_MRAE().to(device)
    rmse_fn = Loss_RMSE().to(device)
    psnr_fn = Loss_PSNR().to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=EPOCHS,
        eta_min=LR * 0.01,
    )

    scaler = GradScaler(enabled=USE_AMP and device.type == "cuda")

    start_epoch = 1
    best_mrae = float("inf")

    if RESUME_PATH:
        start_epoch, best_mrae = load_training_state(
            RESUME_PATH,
            model,
            optimizer,
            scheduler,
            device,
        )
        print(f"Resumed from epoch {start_epoch}")

    for epoch in range(start_epoch, EPOCHS + 1):
        start_time = time.time()

        train_mrae = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            scaler,
            device,
        )

        if len(val_dataset) > 0:
            val_stats = validate(
                model,
                val_loader,
                criterion,
                rmse_fn,
                psnr_fn,
                device,
            )
        else:
            val_stats = {
                "mrae": float("inf"),
                "rmse": 0.0,
                "psnr": 0.0,
            }

        scheduler.step()

        epoch_time = time.time() - start_time
        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch:03d}/{EPOCHS} | "
            f"Train MRAE {train_mrae:.6f} | "
            f"Val MRAE {val_stats['mrae']:.6f} | "
            f"Val RMSE {val_stats['rmse']:.6f} | "
            f"Val PSNR {val_stats['psnr']:.4f} | "
            f"LR {current_lr:.2e} | "
            f"Time {epoch_time:.1f}s"
        )

        # Uses the checkpoint utility already defined in loss.py.
        save_checkpoint(
            model_path=SAVE_DIR,
            epoch=epoch,
            iteration=len(train_loader),
            model=model,
            optimizer=optimizer,
        )

        latest_path = os.path.join(SAVE_DIR, LATEST_MODEL_NAME)
        save_training_state(
            latest_path,
            epoch,
            model,
            optimizer,
            scheduler,
            best_mrae,
        )

        if val_stats["mrae"] < best_mrae:
            best_mrae = val_stats["mrae"]
            best_path = os.path.join(SAVE_DIR, BEST_MODEL_NAME)
            save_training_state(
                best_path,
                epoch,
                model,
                optimizer,
                scheduler,
                best_mrae,
            )
            print(f"Best model saved: {best_path}")


if __name__ == "__main__":
    main()
