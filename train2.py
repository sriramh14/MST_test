"""Full-resolution MST++ RGB-to-HSI training and inference.

Edit the configuration section below, then run only one of:

    python mstpp_full_resolution_fixed_losses.py --mode train
    python mstpp_full_resolution_fixed_losses.py --mode infer

The parser intentionally contains no argument other than ``--mode``.
"""

from __future__ import annotations

import argparse
import hashlib
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

# Losses and metrics are implemented directly below.
# This avoids argument-order, reduction, epsilon, and AMP differences between
# external metric implementations.


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
EPOCHS = 75
# Keep this at 1 when images have different spatial resolutions.
# Each sample is passed to MST++ at its original full resolution.
BATCH_SIZE = 1
NUM_WORKERS = 4
LEARNING_RATE = 1e-4
MIN_LEARNING_RATE = 1e-7
WEIGHT_DECAY = 1e-4
VALIDATION_FRACTION = 0.1
USE_AUGMENTATION = True
USE_AMP = True
AMP_INITIAL_SCALE = 1024.0
GRADIENT_CLIP_NORM = 1.0
PRINT_EVERY = 30
SEED = 42

# Stable loss/metric settings. The NTIRE spectral cubes are normally in [0, 1].
# Change METRIC_DATA_RANGE only when the target data uses another fixed range.
MRAE_EPSILON = 1e-3
SAM_EPSILON = 1e-8
METRIC_DATA_RANGE = 1.0
SSIM_WINDOW_SIZE = 11
SSIM_SIGMA = 1.5
REPORT_SAM_IN_DEGREES = False
WARN_ON_RANGE_MISMATCH = True

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

# Fast corrupt-file filtering. The cache is reused until a file path, size,
# or modification time changes. Set FORCE_REVALIDATE=True for a manual rescan.
VALIDATION_CACHE = Path(OUTPUT_DIR) / "hsi_validation_cache.pth"
INVALID_FILE_LOG = Path(OUTPUT_DIR) / "invalid_hsi_files.txt"
FORCE_REVALIDATE = False


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
# Fast HSI validation and cache
# ============================================================

def is_possible_hsi_shape(shape: Tuple[int, ...]) -> bool:
    return (
        len(shape) == 3
        and HSI_CHANNELS in shape
        and all(int(size) > 0 for size in shape)
    )


def inspect_hdf5_mat(path: Path) -> None:
    """Inspect only HDF5 metadata; opening also detects truncated v7.3 files."""
    candidates: List[Tuple[str, Tuple[int, ...]]] = []

    with h5py.File(str(path), "r") as file:
        if HSI_KEY in file and isinstance(file[HSI_KEY], h5py.Dataset):
            dataset = file[HSI_KEY]
            candidates.append((HSI_KEY, tuple(int(v) for v in dataset.shape)))

        def visitor(name, obj):
            if not isinstance(obj, h5py.Dataset) or obj.ndim != 3:
                return
            try:
                if np.issubdtype(obj.dtype, np.number):
                    record = (name, tuple(int(v) for v in obj.shape))
                    if record not in candidates:
                        candidates.append(record)
            except TypeError:
                return

        file.visititems(visitor)

    if not candidates:
        raise ValueError(f"No numerical 3D HDF5 dataset found in {path}")

    if not any(is_possible_hsi_shape(shape) for _, shape in candidates):
        raise ValueError(
            f"No {HSI_CHANNELS}-band cube found in {path}; "
            f"HDF5 datasets={candidates}"
        )


def inspect_standard_mat(path: Path) -> None:
    """Read the MAT directory only; do not load the full cube."""
    try:
        metadata = sio.whosmat(path)
    except (NotImplementedError, ValueError, OSError):
        # MATLAB v7.3 uses HDF5. h5py.File will also expose truncation errors.
        inspect_hdf5_mat(path)
        return

    candidates = [
        (name, tuple(int(v) for v in shape))
        for name, shape, _dtype in metadata
        if len(shape) == 3
    ]

    if not candidates:
        raise ValueError(f"No 3D array found in {path}")

    preferred = [item for item in candidates if item[0] == HSI_KEY]
    shapes_to_check = preferred if preferred else candidates
    if not any(is_possible_hsi_shape(shape) for _, shape in shapes_to_check):
        raise ValueError(
            f"No {HSI_CHANNELS}-band cube found in {path}; MAT arrays={candidates}"
        )


def inspect_hsi_metadata(path: Path) -> None:
    """Validate shape/readability with the cheapest available operation."""
    extension = path.suffix.lower()

    if extension == ".mat":
        inspect_standard_mat(path)
        return

    if extension == ".npy":
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        if not is_possible_hsi_shape(tuple(int(v) for v in array.shape)):
            raise ValueError(f"Invalid HSI shape {array.shape} in {path}")
        return

    if extension == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            shapes = [
                tuple(int(v) for v in archive[key].shape)
                for key in archive.files
                if archive[key].ndim == 3
            ]
        if not any(is_possible_hsi_shape(shape) for shape in shapes):
            raise ValueError(f"No valid {HSI_CHANNELS}-band cube in {path}; shapes={shapes}")
        return

    # Tensor checkpoints cannot be inspected reliably without loading the object.
    cube = load_hsi(path)
    if not is_possible_hsi_shape(tuple(int(v) for v in cube.shape)):
        raise ValueError(f"Invalid HSI shape {cube.shape} in {path}")


def make_pairs_fingerprint(pairs: List[Tuple[Path, Path]]) -> str:
    records = []
    for hsi_path, rgb_path in pairs:
        for path in (hsi_path, rgb_path):
            stat = path.stat()
            records.append(
                f"{path.resolve()}|{stat.st_size}|{stat.st_mtime_ns}"
            )
    return hashlib.sha256("\n".join(records).encode("utf-8")).hexdigest()


def load_validation_cache(path: Path) -> dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        # Compatibility with older PyTorch versions without weights_only.
        return torch.load(path, map_location="cpu")


def filter_valid_pairs(
    pairs: List[Tuple[Path, Path]],
) -> List[Tuple[Path, Path]]:
    """Skip unreadable HSI files before DataLoader workers are created."""
    VALIDATION_CACHE.parent.mkdir(parents=True, exist_ok=True)
    INVALID_FILE_LOG.parent.mkdir(parents=True, exist_ok=True)

    fingerprint = make_pairs_fingerprint(pairs)
    pair_lookup = {str(hsi.resolve()): (hsi, rgb) for hsi, rgb in pairs}

    if VALIDATION_CACHE.exists() and not FORCE_REVALIDATE:
        try:
            cached = load_validation_cache(VALIDATION_CACHE)
            if isinstance(cached, dict) and cached.get("fingerprint") == fingerprint:
                valid_pairs = [
                    pair_lookup[path]
                    for path in cached.get("valid_hsi_paths", [])
                    if path in pair_lookup
                ]
                invalid_records = cached.get("invalid_records", [])

                if valid_pairs:
                    print("Using cached HSI validation.")
                    print(f"Valid pairs: {len(valid_pairs)}")
                    print(f"Skipped corrupt/invalid files: {len(invalid_records)}")
                    for record in invalid_records[:10]:
                        print(f"  Skipped: {record['path']} | {record['error']}")
                    return valid_pairs
        except Exception as error:
            print(f"Validation cache could not be used; rescanning. Reason: {error}")

    print(f"Checking {len(pairs)} HSI files using metadata only...")
    valid_pairs: List[Tuple[Path, Path]] = []
    invalid_records = []

    for index, (hsi_path, rgb_path) in enumerate(pairs, start=1):
        try:
            inspect_hsi_metadata(hsi_path)
            valid_pairs.append((hsi_path, rgb_path))
        except Exception as error:
            invalid_records.append(
                {
                    "path": str(hsi_path.resolve()),
                    "error": f"{type(error).__name__}: {error}",
                }
            )
            print(f"  Skipping invalid HSI: {hsi_path.name} | {error}")

        if index % 100 == 0 or index == len(pairs):
            print(
                f"  Checked {index}/{len(pairs)} | "
                f"valid={len(valid_pairs)} | invalid={len(invalid_records)}"
            )

    if not valid_pairs:
        raise RuntimeError("No valid HSI/RGB pairs remain after validation")

    if invalid_records:
        with INVALID_FILE_LOG.open("w", encoding="utf-8") as file:
            for record in invalid_records:
                file.write(f"{record['path']} | {record['error']}\n")
        print(f"Invalid-file log saved to: {INVALID_FILE_LOG}")
    elif INVALID_FILE_LOG.exists():
        INVALID_FILE_LOG.unlink()

    torch.save(
        {
            "fingerprint": fingerprint,
            "valid_hsi_paths": [str(hsi.resolve()) for hsi, _ in valid_pairs],
            "invalid_records": invalid_records,
        },
        VALIDATION_CACHE,
    )
    print(f"Validation cache saved to: {VALIDATION_CACHE}")
    return valid_pairs


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

def unwrap_prediction(output: torch.Tensor | List[torch.Tensor] | Tuple[torch.Tensor, ...]) -> torch.Tensor:
    """Return the final reconstruction if a model returns intermediate outputs."""
    if isinstance(output, (list, tuple)):
        if not output:
            raise ValueError("The model returned an empty output sequence")
        output = output[-1]
    if not isinstance(output, torch.Tensor):
        raise TypeError(f"Expected model output to be a tensor, found {type(output)}")
    return output


def check_prediction_target_shapes(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> None:
    if prediction.shape != target.shape:
        raise ValueError(
            "Prediction and target must have identical BCHW shapes; "
            f"prediction={tuple(prediction.shape)}, target={tuple(target.shape)}"
        )
    if prediction.ndim != 4:
        raise ValueError(
            f"Expected BCHW prediction and target tensors, found {prediction.ndim} dimensions"
        )


def stable_mrae_per_sample(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """MRAE per image, computed in float32 with a non-underflowing denominator."""
    prediction = prediction.float()
    target = target.float()
    denominator = target.abs().clamp_min(MRAE_EPSILON)
    return ((prediction - target).abs() / denominator).mean(dim=(1, 2, 3))


def reconstruction_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Stable optimization loss; always evaluated outside mixed precision."""
    check_prediction_target_shapes(prediction, target)
    with torch.amp.autocast(device_type=prediction.device.type, enabled=False):
        return stable_mrae_per_sample(prediction, target).mean()


def _gaussian_kernel(
    channels: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    window_size = min(SSIM_WINDOW_SIZE, 11)
    if window_size % 2 == 0:
        window_size -= 1
    coordinates = torch.arange(window_size, device=device, dtype=dtype)
    coordinates = coordinates - (window_size - 1) / 2
    gaussian = torch.exp(-(coordinates.square()) / (2 * SSIM_SIGMA * SSIM_SIGMA))
    gaussian = gaussian / gaussian.sum()
    kernel_2d = torch.outer(gaussian, gaussian)
    return kernel_2d.view(1, 1, window_size, window_size).expand(
        channels, 1, window_size, window_size
    ).contiguous()


def _ssim_filter(image: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    padding = kernel.shape[-1] // 2
    height, width = image.shape[-2:]
    padding_mode = "reflect" if height > padding and width > padding else "replicate"
    image = F.pad(image, (padding, padding, padding, padding), mode=padding_mode)
    return F.conv2d(image, kernel, groups=image.shape[1])


def ssim_per_sample(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """Windowed SSIM averaged over all spectral bands and spatial positions."""
    channels = prediction.shape[1]
    kernel = _gaussian_kernel(channels, prediction.device, prediction.dtype)

    mu_prediction = _ssim_filter(prediction, kernel)
    mu_target = _ssim_filter(target, kernel)

    mu_prediction_sq = mu_prediction.square()
    mu_target_sq = mu_target.square()
    mu_cross = mu_prediction * mu_target

    sigma_prediction_sq = (
        _ssim_filter(prediction.square(), kernel) - mu_prediction_sq
    ).clamp_min(0.0)
    sigma_target_sq = (
        _ssim_filter(target.square(), kernel) - mu_target_sq
    ).clamp_min(0.0)
    sigma_cross = _ssim_filter(prediction * target, kernel) - mu_cross

    c1 = (0.01 * METRIC_DATA_RANGE) ** 2
    c2 = (0.03 * METRIC_DATA_RANGE) ** 2
    numerator = (2.0 * mu_cross + c1) * (2.0 * sigma_cross + c2)
    denominator = (
        (mu_prediction_sq + mu_target_sq + c1)
        * (sigma_prediction_sq + sigma_target_sq + c2)
    ).clamp_min(1e-12)

    return (numerator / denominator).mean(dim=(1, 2, 3))


@torch.no_grad()
def calculate_metric_tensors(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """Return one value per image for every metric."""
    check_prediction_target_shapes(prediction, target)

    prediction = prediction.detach().float()
    target = target.detach().float()
    error = prediction - target
    mse = error.square().mean(dim=(1, 2, 3))

    mrae_values = stable_mrae_per_sample(prediction, target)
    rmse_values = mse.sqrt()
    psnr_values = 10.0 * torch.log10(
        (METRIC_DATA_RANGE ** 2) / mse.clamp_min(1e-12)
    )

    dot_product = (prediction * target).sum(dim=1)
    prediction_norm = prediction.square().sum(dim=1).sqrt()
    target_norm = target.square().sum(dim=1).sqrt()
    norm_product = prediction_norm * target_norm
    valid_pixels = norm_product > SAM_EPSILON

    cosine = dot_product / norm_product.clamp_min(SAM_EPSILON)
    angle = torch.acos(cosine.clamp(-1.0, 1.0))
    angle = torch.where(valid_pixels, angle, torch.zeros_like(angle))
    valid_count = valid_pixels.sum(dim=(1, 2)).clamp_min(1)
    sam_values = angle.sum(dim=(1, 2)) / valid_count
    if REPORT_SAM_IN_DEGREES:
        sam_values = torch.rad2deg(sam_values)

    ssim_values = ssim_per_sample(prediction, target)

    metrics = {
        "mrae": mrae_values,
        "rmse": rmse_values,
        "sam": sam_values,
        "psnr": psnr_values,
        "ssim": ssim_values,
    }
    for name, values in metrics.items():
        if not torch.isfinite(values).all():
            raise FloatingPointError(
                f"Non-finite {name} values: {values.detach().cpu().tolist()}"
            )
    return metrics


@torch.no_grad()
def calculate_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> Dict[str, float]:
    """Return image-averaged metrics for inference reporting."""
    tensors = calculate_metric_tensors(prediction, target)
    return {name: float(values.mean().item()) for name, values in tensors.items()}


def print_range_diagnostics(
    rgb: torch.Tensor,
    hsi: torch.Tensor,
    prefix: str,
) -> None:
    rgb_min = float(rgb.detach().min().item())
    rgb_max = float(rgb.detach().max().item())
    hsi_min = float(hsi.detach().min().item())
    hsi_max = float(hsi.detach().max().item())
    near_zero_fraction = float(
        (hsi.detach().abs() < MRAE_EPSILON).float().mean().item()
    )
    print(
        f"{prefix} ranges | RGB=[{rgb_min:.6f}, {rgb_max:.6f}] | "
        f"HSI=[{hsi_min:.6f}, {hsi_max:.6f}] | "
        f"HSI |x|<{MRAE_EPSILON:g}: {100.0 * near_zero_fraction:.3f}%"
    )
    if WARN_ON_RANGE_MISMATCH and (
        hsi_min < -1e-4 or hsi_max > METRIC_DATA_RANGE + 1e-4
    ):
        print(
            "WARNING: HSI values are outside the expected metric range "
            f"[0, {METRIC_DATA_RANGE}]. Set NORMALIZATION or "
            "METRIC_DATA_RANGE consistently before interpreting PSNR/SSIM."
        )


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


def add_batch_metrics(
    sums: Dict[str, float],
    batch_metrics: Dict[str, torch.Tensor],
) -> None:
    # Sum per-image values. This avoids accidentally averaging a batch scalar
    # twice and remains correct when the final batch is smaller.
    for name, values in batch_metrics.items():
        sums[name] += float(values.detach().sum().item())
    sums["loss"] += float(batch_metrics["mrae"].detach().sum().item())


def average_metric_sums(sums: Dict[str, float], count: int) -> Dict[str, float]:
    if count == 0:
        raise RuntimeError("No samples were processed")
    return {name: value / count for name, value in sums.items()}


def print_metrics(prefix: str, values: Dict[str, float]) -> None:
    sam_label = "SAM(deg)" if REPORT_SAM_IN_DEGREES else "SAM(rad)"
    print(
        f"{prefix} | "
        f"Loss/MRAE: {values['loss']:.6f} | "
        f"RMSE: {values['rmse']:.6f} | "
        f"{sam_label}: {values['sam']:.6f} | "
        f"PSNR: {values['psnr']:.4f} dB | "
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

        if batch_index == 1:
            print_range_diagnostics(rgb, hsi, "  Train")

        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            prediction = unwrap_prediction(model(rgb))

        # MRAE is deliberately computed in float32, outside autocast.
        loss = reconstruction_loss(prediction, hsi)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite training loss: {loss.item()}")

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        
        gradients_are_finite = all(
            parameter.grad is None
            or torch.isfinite(parameter.grad).all().item()
            for parameter in model.parameters()
        )
        
        if not gradients_are_finite:
            if not use_amp:
                raise FloatingPointError(
                    f"Non-finite gradients at batch {batch_index} with AMP disabled."
                )
        
            previous_scale = scaler.get_scale()
        
            # Skip this optimizer update. unscale_() has already recorded
            # the overflow, so update() will lower the AMP scale.
            optimizer.zero_grad(set_to_none=True)
            scaler.update()
        
            print(
                f"WARNING: skipped batch {batch_index} because AMP gradients "
                f"overflowed; scale {previous_scale:g} -> {scaler.get_scale():g}"
            )
            continue
        
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            GRADIENT_CLIP_NORM,
            error_if_nonfinite=True,
        )
        
        scaler.step(optimizer)
        scaler.update()

        batch_size = rgb.size(0)
        batch_metrics = calculate_metric_tensors(prediction, hsi)
        add_batch_metrics(sums, batch_metrics)
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

    for batch_index, (rgb, hsi) in enumerate(loader, start=1):
        rgb = rgb.to(device, non_blocking=True)
        hsi = hsi.to(device, non_blocking=True)

        if batch_index == 1:
            print_range_diagnostics(rgb, hsi, "  Validation")

        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            prediction = unwrap_prediction(model(rgb))

        # Compute all metrics from the same float32, unclamped prediction.
        batch_size = rgb.size(0)
        batch_metrics = calculate_metric_tensors(prediction, hsi)
        add_batch_metrics(sums, batch_metrics)
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
            "metric_config": {
                "mrae_epsilon": MRAE_EPSILON,
                "sam_epsilon": SAM_EPSILON,
                "data_range": METRIC_DATA_RANGE,
                "sam_in_degrees": REPORT_SAM_IN_DEGREES,
            },
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
    all_pairs = filter_valid_pairs(all_pairs)
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
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp,init_scale=AMP_INITIAL_SCALE)

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
        prediction = unwrap_prediction(model(rgb))

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
