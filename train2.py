"""Full-resolution MST++ RGB-to-HSI training and inference.

Edit the configuration section below, then run only one of:

    python train_mst_plus_plus.py --mode train
    python train_mst_plus_plus.py --mode infer

The parser intentionally contains no argument other than ``--mode``.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from model.MST_Plus_Plus import MST_Plus_Plus

# Existing loss package supplied by the project.
from loss.mrae import mrae
from loss.rmse import rmse
from loss.sam import sam
from loss.psnr import psnr
from loss.ssim import ssim


# ============================================================
# Configuration: edit values here, not on the command line
# ============================================================

# Data paths
HSI_DATA_DIR = "/kaggle/input/datasets/sriramhari14/ntire-2022/Train_spectral/Train_spectral"
RGB_DATA_DIR = "/kaggle/input/datasets/sriramhari14/ntire-2022/Train_RGB/Train_RGB"
OUTPUT_DIR = "./mstpp_checkpoints"

# HSI data settings
HSI_CHANNELS = 31
HSI_KEY = "cube"
NORMALIZATION = "none"          # "none" or "minmax"

# Model settings
MODEL_STAGES = 3
MODEL_FEATURES = 31              # required by the supplied MST++ implementation

# Training settings
EPOCHS = 50
# Keep this at 1 when images have different spatial resolutions.
# Each sample is passed to MST++ at its original full resolution.
BATCH_SIZE = 1
NUM_WORKERS = 4
LEARNING_RATE = 4e-4
MIN_LEARNING_RATE = 1e-7
WEIGHT_DECAY = 1e-4
VALIDATION_FRACTION = 0.1
USE_AUGMENTATION = True
USE_AMP = True
GRADIENT_CLIP_NORM = 1.0
PRINT_EVERY = 30
SEED = 42

# Leave as None to start training from epoch 1.
RESUME_CHECKPOINT: Optional[str] = None
# Example:
# RESUME_CHECKPOINT = "./mstpp_checkpoints/last_mstpp.pth"

# Inference settings
INFERENCE_CHECKPOINT = "./mstpp_checkpoints/best_mstpp.pth"
INFERENCE_RGB_PATH = "./test_rgb.png"
# Set to None when ground-truth HSI is unavailable.
INFERENCE_HSI_PATH: Optional[str] = "./test_hsi.mat"
INFERENCE_OUTPUT_DIR = "./mstpp_results"
CLAMP_INFERENCE_OUTPUT = True
HEATMAP_REDUCTION = "mae"       # "mae" or "rmse"

# Supported file extensions
HSI_EXTENSIONS = {".mat", ".npy", ".npz", ".pt", ".pth"}
RGB_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".npy", ".pt", ".pth"
}


# ============================================================
# General utilities
# ============================================================

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def scalar_value(value: torch.Tensor | float) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().mean().item())
    return float(value)


# ============================================================
# File loading and pairing
# ============================================================

def extract_cube(data: dict, path: Path) -> np.ndarray:
    if HSI_KEY in data:
        value = data[HSI_KEY]
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
        if isinstance(value, np.ndarray) and value.ndim == 3:
            return value

    candidates = []
    for key, value in data.items():
        if str(key).startswith("__"):
            continue
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
        if isinstance(value, np.ndarray) and value.ndim == 3:
            candidates.append(value)

    if not candidates:
        raise ValueError(f"No three-dimensional HSI cube found in {path}")
    return max(candidates, key=lambda array: array.size)


def load_hdf5_mat(path: Path) -> np.ndarray:
    candidates = []
    with h5py.File(str(path), "r") as file:
        def visitor(_name, obj):
            if isinstance(obj, h5py.Dataset) and obj.ndim == 3:
                candidates.append(np.asarray(obj))
        file.visititems(visitor)

    if not candidates:
        raise ValueError(f"No three-dimensional HSI cube found in {path}")

    cube = max(candidates, key=lambda array: array.size)
    # MATLAB v7.3 arrays read through h5py commonly have reversed axes.
    return np.transpose(cube, tuple(range(cube.ndim - 1, -1, -1)))


def load_hsi(path: Path) -> np.ndarray:
    extension = path.suffix.lower()

    if extension == ".npy":
        cube = np.load(path)
    elif extension == ".npz":
        loaded = np.load(path)
        candidates = [loaded[key] for key in loaded.files if loaded[key].ndim == 3]
        if not candidates:
            raise ValueError(f"No three-dimensional HSI cube found in {path}")
        cube = max(candidates, key=lambda array: array.size)
    elif extension == ".mat":
        try:
            cube = extract_cube(sio.loadmat(path), path)
        except (NotImplementedError, ValueError):
            cube = load_hdf5_mat(path)
    elif extension in {".pt", ".pth"}:
        loaded = torch.load(path, map_location="cpu")
        if isinstance(loaded, torch.Tensor):
            cube = loaded.detach().cpu().numpy()
        elif isinstance(loaded, np.ndarray):
            cube = loaded
        elif isinstance(loaded, dict):
            cube = extract_cube(loaded, path)
        else:
            raise TypeError(f"Unsupported object in {path}: {type(loaded)}")
    else:
        raise ValueError(f"Unsupported HSI extension: {extension}")

    cube = np.asarray(cube, dtype=np.float32).squeeze()
    if cube.ndim != 3:
        raise ValueError(f"Expected a 3D HSI cube in {path}, found {cube.shape}")

    if cube.shape[0] == HSI_CHANNELS:
        pass
    elif cube.shape[-1] == HSI_CHANNELS:
        cube = cube.transpose(2, 0, 1)
    elif cube.shape[1] == HSI_CHANNELS:
        cube = cube.transpose(1, 0, 2)
    else:
        raise ValueError(
            f"Cannot identify the {HSI_CHANNELS}-band axis in {path}; shape={cube.shape}"
        )

    if not np.isfinite(cube).all():
        raise ValueError(f"NaN or Inf values found in {path}")

    return np.ascontiguousarray(cube)


def load_rgb(path: Path) -> np.ndarray:
    extension = path.suffix.lower()

    if extension in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}:
        image = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
        return np.ascontiguousarray(image.transpose(2, 0, 1))

    if extension == ".npy":
        image = np.load(path)
    elif extension in {".pt", ".pth"}:
        loaded = torch.load(path, map_location="cpu")
        if not isinstance(loaded, torch.Tensor):
            raise TypeError(f"Expected a tensor in RGB file {path}")
        image = loaded.detach().cpu().numpy()
    else:
        raise ValueError(f"Unsupported RGB extension: {extension}")

    image = np.asarray(image, dtype=np.float32).squeeze()
    if image.ndim != 3:
        raise ValueError(f"Expected a 3D RGB image in {path}, found {image.shape}")

    if image.shape[0] == 3:
        pass
    elif image.shape[-1] == 3:
        image = image.transpose(2, 0, 1)
    else:
        raise ValueError(f"Cannot identify RGB channels in {path}; shape={image.shape}")

    if image.max() > 1.0:
        image = image / 255.0
    return np.ascontiguousarray(image)


def normalize_hsi(cube: np.ndarray) -> np.ndarray:
    if NORMALIZATION == "none":
        return cube
    if NORMALIZATION == "minmax":
        minimum = cube.min()
        maximum = cube.max()
        return (cube - minimum) / (maximum - minimum + 1e-8)
    raise ValueError(f"Unknown NORMALIZATION value: {NORMALIZATION}")


def build_pairs() -> List[Tuple[Path, Path]]:
    hsi_root = Path(HSI_DATA_DIR)
    rgb_root = Path(RGB_DATA_DIR)

    if not hsi_root.exists():
        raise FileNotFoundError(f"HSI_DATA_DIR does not exist: {hsi_root}")
    if not rgb_root.exists():
        raise FileNotFoundError(f"RGB_DATA_DIR does not exist: {rgb_root}")

    rgb_by_stem = {
        path.stem: path
        for path in rgb_root.rglob("*")
        if path.is_file() and path.suffix.lower() in RGB_EXTENSIONS
    }

    pairs = []
    for hsi_path in sorted(hsi_root.rglob("*")):
        if hsi_path.is_file() and hsi_path.suffix.lower() in HSI_EXTENSIONS:
            rgb_path = rgb_by_stem.get(hsi_path.stem)
            if rgb_path is not None:
                pairs.append((hsi_path, rgb_path))

    if not pairs:
        raise RuntimeError(
            "No paired files found. RGB and HSI files must have matching filename stems."
        )

    return pairs


# ============================================================
# Dataset
# ============================================================

def augment_pair(
    rgb: torch.Tensor,
    hsi: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if random.random() < 0.5:
        rgb = torch.flip(rgb, dims=[1])
        hsi = torch.flip(hsi, dims=[1])
    if random.random() < 0.5:
        rgb = torch.flip(rgb, dims=[2])
        hsi = torch.flip(hsi, dims=[2])

    rotations = random.randint(0, 3)
    if rotations:
        rgb = torch.rot90(rgb, rotations, dims=[1, 2])
        hsi = torch.rot90(hsi, rotations, dims=[1, 2])

    return rgb.contiguous(), hsi.contiguous()


class RGBHSIDataset(Dataset):
    def __init__(self, pairs: List[Tuple[Path, Path]], training: bool):
        self.pairs = pairs
        self.training = training

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        hsi_path, rgb_path = self.pairs[index]

        rgb = torch.from_numpy(load_rgb(rgb_path)).float()
        hsi = torch.from_numpy(normalize_hsi(load_hsi(hsi_path))).float()

        if rgb.shape[1:] != hsi.shape[1:]:
            raise ValueError(
                f"Spatial mismatch for {hsi_path.stem}: RGB={rgb.shape}, HSI={hsi.shape}"
            )

        # No crop or resize: use the complete RGB/HSI pair.
        # MST++ pads internally to a multiple of 8 and crops its output back
        # to the original height and width.
        if self.training and USE_AUGMENTATION:
            rgb, hsi = augment_pair(rgb, hsi)

        return rgb, hsi


# ============================================================
# Losses, metrics, and residual heatmap
# ============================================================

def reconstruction_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Optimization loss. MRAE follows the target, prediction order used by the loss package."""
    value = mrae(target, prediction)
    if not isinstance(value, torch.Tensor):
        raise TypeError("loss.mrae.mrae must return a torch.Tensor during training")
    return value.mean()


@torch.no_grad()
def calculate_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> Dict[str, float]:
    prediction = prediction.detach().float()
    target = target.detach().float()

    return {
        "mrae": scalar_value(mrae(target, prediction)),
        "rmse": scalar_value(rmse(target, prediction)),
        "sam": scalar_value(sam(target, prediction)),
        "psnr": scalar_value(psnr(target, prediction)),
        "ssim": scalar_value(ssim(target, prediction)),
    }


class ResidualHeatmap(nn.Module):
    """Collapse the spectral residual into one spatial error map."""

    def __init__(self, reduction: str = "mae"):
        super().__init__()
        if reduction not in {"mae", "rmse"}:
            raise ValueError("HEATMAP_REDUCTION must be 'mae' or 'rmse'")
        self.reduction = reduction

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        residual = prediction - target
        if self.reduction == "rmse":
            return residual.square().mean(dim=1, keepdim=True).sqrt()
        return residual.abs().mean(dim=1, keepdim=True)


def save_heatmap(heatmap: torch.Tensor, path: Path, title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    array = heatmap[0, 0].detach().cpu().numpy()

    plt.figure(figsize=(7, 6))
    image = plt.imshow(array, cmap="inferno")
    plt.colorbar(image, label="Residual magnitude")
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()


# ============================================================
# Training and validation
# ============================================================

def empty_metric_sums() -> Dict[str, float]:
    return {
        "loss": 0.0,
        "mrae": 0.0,
        "rmse": 0.0,
        "sam": 0.0,
        "psnr": 0.0,
        "ssim": 0.0,
    }


def average_metric_sums(sums: Dict[str, float], count: int) -> Dict[str, float]:
    if count == 0:
        raise RuntimeError("No samples were processed")
    return {name: value / count for name, value in sums.items()}


def print_metrics(prefix: str, values: Dict[str, float]) -> None:
    print(
        f"{prefix} | "
        f"Loss: {values['loss']:.6f} | "
        f"MRAE: {values['mrae']:.6f} | "
        f"RMSE: {values['rmse']:.6f} | "
        f"SAM: {values['sam']:.6f} | "
        f"PSNR: {values['psnr']:.4f} | "
        f"SSIM: {values['ssim']:.4f}"
    )


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    use_amp: bool,
) -> Dict[str, float]:
    model.train()
    sums = empty_metric_sums()
    sample_count = 0

    for batch_index, (rgb, hsi) in enumerate(loader, start=1):
        rgb = rgb.to(device, non_blocking=True)
        hsi = hsi.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            prediction = model(rgb)
            loss = reconstruction_loss(prediction, hsi)

        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite training loss: {loss.item()}")

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP_NORM)
        scaler.step(optimizer)
        scaler.update()

        batch_size = rgb.size(0)
        batch_metrics = calculate_metrics(prediction, hsi)
        sums["loss"] += scalar_value(loss) * batch_size
        for name, value in batch_metrics.items():
            sums[name] += value * batch_size
        sample_count += batch_size

        if batch_index % PRINT_EVERY == 0 or batch_index == len(loader):
            print_metrics(
                f"  Train batch {batch_index:04d}/{len(loader):04d}",
                average_metric_sums(sums, sample_count),
            )

    return average_metric_sums(sums, sample_count)


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool,
) -> Dict[str, float]:
    model.eval()
    sums = empty_metric_sums()
    sample_count = 0

    for rgb, hsi in loader:
        rgb = rgb.to(device, non_blocking=True)
        hsi = hsi.to(device, non_blocking=True)

        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            prediction = model(rgb)
            loss = reconstruction_loss(prediction, hsi)

        batch_size = rgb.size(0)
        batch_metrics = calculate_metrics(prediction, hsi)
        sums["loss"] += scalar_value(loss) * batch_size
        for name, value in batch_metrics.items():
            sums[name] += value * batch_size
        sample_count += batch_size

    return average_metric_sums(sums, sample_count)


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    epoch: int,
    best_mrae: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_mrae": best_mrae,
            "normalization": NORMALIZATION,
            "model_config": {
                "in_channels": 3,
                "out_channels": HSI_CHANNELS,
                "n_feat": MODEL_FEATURES,
                "stage": MODEL_STAGES,
            },
        },
        path,
    )


def train() -> None:
    set_seed(SEED)
    device = get_device()
    use_amp = USE_AMP and device.type == "cuda"

    all_pairs = build_pairs()
    random.Random(SEED).shuffle(all_pairs)

    validation_size = max(1, int(len(all_pairs) * VALIDATION_FRACTION))
    validation_pairs = all_pairs[:validation_size]
    training_pairs = all_pairs[validation_size:]

    if not training_pairs:
        raise RuntimeError("No training pairs remain after the validation split")

    training_dataset = RGBHSIDataset(training_pairs, training=True)
    validation_dataset = RGBHSIDataset(validation_pairs, training=False)

    loader_options = {
        "batch_size": BATCH_SIZE,
        "num_workers": NUM_WORKERS,
        "pin_memory": device.type == "cuda",
        "persistent_workers": NUM_WORKERS > 0,
    }
    training_loader = DataLoader(
        training_dataset,
        shuffle=True,
        drop_last=False,
        **loader_options,
    )
    validation_loader = DataLoader(
        validation_dataset,
        shuffle=False,
        drop_last=False,
        **loader_options,
    )

    model = MST_Plus_Plus(
        in_channels=3,
        out_channels=HSI_CHANNELS,
        n_feat=MODEL_FEATURES,
        stage=MODEL_STAGES,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=EPOCHS,
        eta_min=MIN_LEARNING_RATE,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    start_epoch = 1
    best_mrae = float("inf")

    if RESUME_CHECKPOINT:
        checkpoint = torch.load(RESUME_CHECKPOINT, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_mrae = float(checkpoint.get("best_mrae", best_mrae))
        print(f"Resumed from epoch {start_epoch - 1}: {RESUME_CHECKPOINT}")

    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device: {device} | AMP: {use_amp}")
    print(f"Training pairs: {len(training_pairs)}")
    print(f"Validation pairs: {len(validation_pairs)}")
    print("Input mode: full resolution (no cropping or resizing)")

    for epoch in range(start_epoch, EPOCHS + 1):
        print(f"\nEpoch {epoch:03d}/{EPOCHS:03d}")

        training_metrics = train_one_epoch(
            model=model,
            loader=training_loader,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            use_amp=use_amp,
        )
        validation_metrics = validate(
            model=model,
            loader=validation_loader,
            device=device,
            use_amp=use_amp,
        )
        scheduler.step()

        print_metrics("Train", training_metrics)
        print_metrics("Validation", validation_metrics)
        print(f"Learning rate: {optimizer.param_groups[0]['lr']:.2e}")

        improved = validation_metrics["mrae"] < best_mrae
        if improved:
            best_mrae = validation_metrics["mrae"]

        save_checkpoint(
            output_dir / "last_mstpp.pth",
            model,
            optimizer,
            scheduler,
            epoch,
            best_mrae,
        )

        if improved:
            save_checkpoint(
                output_dir / "best_mstpp.pth",
                model,
                optimizer,
                scheduler,
                epoch,
                best_mrae,
            )
            print(f"Saved new best checkpoint with validation MRAE {best_mrae:.6f}")


# ============================================================
# Full-resolution inference and heatmap generation
# ============================================================

@torch.no_grad()
def infer() -> None:
    device = get_device()
    use_amp = USE_AMP and device.type == "cuda"

    checkpoint_path = Path(INFERENCE_CHECKPOINT)
    rgb_path = Path(INFERENCE_RGB_PATH)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Inference checkpoint does not exist: {checkpoint_path}")
    if not rgb_path.exists():
        raise FileNotFoundError(f"Inference RGB image does not exist: {rgb_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_config = checkpoint.get("model_config", {})

    model = MST_Plus_Plus(
        in_channels=model_config.get("in_channels", 3),
        out_channels=model_config.get("out_channels", HSI_CHANNELS),
        n_feat=model_config.get("n_feat", MODEL_FEATURES),
        stage=model_config.get("stage", MODEL_STAGES),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    rgb = torch.from_numpy(load_rgb(rgb_path)).unsqueeze(0).to(device)

    with torch.amp.autocast(device_type=device.type, enabled=use_amp):
        prediction = model(rgb)

    prediction = prediction.float()
    if CLAMP_INFERENCE_OUTPUT:
        prediction = prediction.clamp(0.0, 1.0)

    save_dir = Path(INFERENCE_OUTPUT_DIR)
    save_dir.mkdir(parents=True, exist_ok=True)
    stem = rgb_path.stem

    prediction_chw = prediction[0].cpu().numpy()
    np.save(save_dir / f"{stem}_predicted.npy", prediction_chw)
    sio.savemat(
        save_dir / f"{stem}_predicted.mat",
        {HSI_KEY: prediction_chw.transpose(1, 2, 0)},
    )
    print(f"Prediction saved to: {save_dir}")

    if INFERENCE_HSI_PATH is None or str(INFERENCE_HSI_PATH).strip() == "":
        print("INFERENCE_HSI_PATH is not set; metrics and residual heatmap were skipped.")
        return

    target_path = Path(INFERENCE_HSI_PATH)
    if not target_path.exists():
        raise FileNotFoundError(f"Inference ground-truth HSI does not exist: {target_path}")

    target = torch.from_numpy(normalize_hsi(load_hsi(target_path))).unsqueeze(0).to(device)
    if prediction.shape != target.shape:
        raise ValueError(
            f"Prediction and ground truth have different shapes: "
            f"prediction={prediction.shape}, target={target.shape}"
        )

    inference_metrics = calculate_metrics(prediction, target)
    print(
        "Inference | "
        f"MRAE: {inference_metrics['mrae']:.6f} | "
        f"RMSE: {inference_metrics['rmse']:.6f} | "
        f"SAM: {inference_metrics['sam']:.6f} | "
        f"PSNR: {inference_metrics['psnr']:.4f} | "
        f"SSIM: {inference_metrics['ssim']:.4f}"
    )

    heatmap_module = ResidualHeatmap(HEATMAP_REDUCTION).to(device)
    heatmap = heatmap_module(prediction, target)

    heatmap_png = save_dir / f"{stem}_{HEATMAP_REDUCTION}_residual_heatmap.png"
    heatmap_npy = save_dir / f"{stem}_{HEATMAP_REDUCTION}_residual.npy"

    save_heatmap(
        heatmap,
        heatmap_png,
        f"{HEATMAP_REDUCTION.upper()} spectral residual: {stem}",
    )
    np.save(heatmap_npy, heatmap[0, 0].cpu().numpy())

    print(f"Residual heatmap saved to: {heatmap_png}")
    print(f"Raw residual map saved to: {heatmap_npy}")


# ============================================================
# Parser: --mode is deliberately the only parser argument
# ============================================================

def parse_mode() -> str:
    parser = argparse.ArgumentParser("MST++ RGB-to-HSI training and inference")
    parser.add_argument(
        "--mode",
        choices=["train", "infer"],
        required=True,
        help="Run training or inference.",
    )
    return parser.parse_args().mode


def main() -> None:
    mode = parse_mode()
    if mode == "train":
        train()
    else:
        infer()


if __name__ == "__main__":
    main()
