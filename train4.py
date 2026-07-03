"""Full-resolution training and inference for a frozen MST++ + ResShift residual diffusion pipeline.

The pretrained MST++ model produces a coarse HSI (y0) and remains completely frozen.
Only the ResShift denoiser (f_theta) is optimized. The diffusion process follows
"ResShift: Efficient Diffusion Model for Image Super-resolution by Residual Shifting"
(Yue, Wang & Loy, NeurIPS 2023), Eqs. (1)-(10): a short Markov chain shifts the
residual e0 = y0 - x0 between the coarse MST++ prediction (y0) and the ground-truth
HSI (x0), instead of corrupting toward Gaussian white noise.

Edit the configuration section, then run one of:

    python train_mstpp_residual_resshift.py --mode train
    python train_mstpp_residual_resshift.py --mode infer

The parser intentionally contains no argument other than ``--mode``.
"""

from __future__ import annotations

import argparse
import hashlib
import math
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

from model.MST_Plus_Plus import MST, MST_Plus_Plus


# ============================================================
# Configuration: edit values here, not on the command line
# ============================================================

# Data paths
HSI_DATA_DIR = "/kaggle/input/datasets/sriramhari14/ntire-2022/Train_spectral/Train_spectral"
RGB_DATA_DIR = "/kaggle/input/datasets/sriramhari14/ntire-2022/Train_RGB/Train_RGB"
OUTPUT_DIR = "./mstpp_resshift_checkpoints"

# Pretrained frozen MST++ checkpoint
MSTPP_CHECKPOINT = "./mstpp_checkpoints/mst_plus_plus.pth"

# HSI data settings
HSI_CHANNELS = 31
HSI_KEY = "cube"
NORMALIZATION = "none"          # "none" or "minmax"

# Frozen MST++ fallback settings. Values stored in its checkpoint take priority.
MSTPP_STAGES = 3
MSTPP_FEATURES = 31

# ResShift diffusion settings (paper notation: T, kappa, p; Eqs. 1-10)
RESSHIFT_TIMESTEPS = 15         # T
RESSHIFT_KAPPA = 2.0            # kappa: noise-strength hyper-parameter
RESSHIFT_P = 0.3                # p: shifting-speed hyper-parameter, Eq. (10)
RESSHIFT_MIN_NOISE_LEVEL = 0.04 # used to set eta_1, Sec. 2.2

# Denoiser (f_theta) settings
RESSHIFT_FEATURES = 31
RESSHIFT_BODY_DEPTH = 3
RESSHIFT_MST_STAGE = 2
RESSHIFT_NUM_BLOCKS = (1, 1, 1)

# Training settings
EPOCHS = 75
# Keep this at 1 when full-resolution images have different spatial sizes.
BATCH_SIZE = 1
NUM_WORKERS = 4
LEARNING_RATE = 2e-4
MIN_LEARNING_RATE = 1e-7
WEIGHT_DECAY = 1e-4
VALIDATION_FRACTION = 0.1
USE_AUGMENTATION = True
USE_AMP = True
AMP_INITIAL_SCALE = 1024.0
GRADIENT_CLIP_NORM = 1.0
PRINT_EVERY = 30
SEED = 42

# The loss is applied only to the ResShift denoiser output. MST++ receives no gradients.
RESIDUAL_LOSS_WEIGHT = 1.0
RECONSTRUCTION_LOSS_WEIGHT = 0.5
SPECTRAL_LOSS_WEIGHT = 0.1
SMOOTH_L1_BETA = 0.01

# "endpoint" validates with one denoiser pass at xt=coarse_hsi (deterministic,
# analogous to a single reverse step). "sample" runs the complete T-step reverse
# Markov chain of Eq. (4) and is much more expensive.
VALIDATION_MODE = "endpoint"    # "endpoint" or "sample"
VALIDATION_STOCHASTIC = False

# Stable metric settings. NTIRE spectral cubes are normally in [0, 1].
MRAE_EPSILON = 1e-3
SAM_EPSILON = 1e-8
METRIC_DATA_RANGE = 1.0
SSIM_WINDOW_SIZE = 11
SSIM_SIGMA = 1.5
REPORT_SAM_IN_DEGREES = False
WARN_ON_RANGE_MISMATCH = True

# Leave as None to start ResShift training from epoch 1.
RESUME_CHECKPOINT: Optional[str] = None
# Example:
# RESUME_CHECKPOINT = "./mstpp_resshift_checkpoints/last_resshift.pth"

# Inference settings
INFERENCE_RESSHIFT_CHECKPOINT = "./mstpp_resshift_checkpoints/best_resshift.pth"
INFERENCE_RGB_PATH = "./test_rgb.png"
INFERENCE_HSI_PATH: Optional[str] = "./test_hsi.mat"
INFERENCE_OUTPUT_DIR = "./mstpp_resshift_results"
CLAMP_INFERENCE_OUTPUT = True
INFERENCE_STOCHASTIC = True     # ResShift's reverse process is stochastic by design
HEATMAP_REDUCTION = "mae"       # "mae" or "rmse"

# Supported file extensions
HSI_EXTENSIONS = {".mat", ".npy", ".npz", ".pt", ".pth"}
RGB_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".npy", ".pt", ".pth"
}

# Fast corrupt-file filtering.
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
# ResShift noise schedule (Sec. 2.2, Eqs. 9-10) and forward/reverse process
# (Eqs. 1-8) -- ported inline from the ResShift model file so this script is
# self-contained, exactly as the original ResShift paper defines it, with y0
# playing the role of the frozen MST++ coarse prediction.
# ============================================================

def make_eta_schedule(T: int, kappa: float, p: float,
                       min_noise_level: float) -> torch.Tensor:
    """
    eta_1 -> min((min_noise_level/kappa)^2, 0.001), eta_T -> 0.999, and the
    intermediate steps follow the non-uniform geometric schedule of Eq. (9)-(10).
    Returns a tensor of shape (T,) with etas[t-1] == eta_t (1-indexed in the paper).
    """
    eta_1 = min((min_noise_level / kappa) ** 2, 0.001)
    eta_T = 0.999

    if T == 1:
        return torch.tensor([eta_T], dtype=torch.float64).float()

    sqrt_eta_1 = math.sqrt(eta_1)
    b0 = math.exp((1.0 / (2 * (T - 1))) * math.log(eta_T / eta_1))

    sqrt_etas = [sqrt_eta_1]
    for t in range(2, T):  # t = 2, ..., T-1
        beta_t = ((t - 1) / (T - 1)) ** p * (T - 1)
        sqrt_etas.append(sqrt_eta_1 * (b0 ** beta_t))
    sqrt_etas.append(math.sqrt(eta_T))  # t = T

    etas = torch.tensor(sqrt_etas, dtype=torch.float64) ** 2
    return etas.float()


class ResShiftSchedule(nn.Module):
    """Registered-buffer container for eta_t, alpha_t, and the posterior
    mean/variance coefficients of Eq. (6)-(8)."""

    def __init__(self, T: int, kappa: float, p: float, min_noise_level: float):
        super().__init__()
        self.T = T
        self.kappa = kappa
        self.p = p

        etas = make_eta_schedule(T, kappa, p, min_noise_level)          # eta_t, t=1..T
        etas_prev = torch.cat([torch.zeros(1), etas[:-1]])              # eta_0 := 0
        alphas = etas - etas_prev
        alphas[0] = etas[0]                                             # alpha_1 = eta_1

        self.register_buffer("etas", etas)
        self.register_buffer("etas_prev", etas_prev)
        self.register_buffer("alphas", alphas)

        self.register_buffer(
            "posterior_variance",
            (kappa ** 2) * etas_prev * alphas / etas.clamp(min=1e-12),
        )
        self.register_buffer("posterior_mean_coef_xt", etas_prev / etas.clamp(min=1e-12))
        self.register_buffer("posterior_mean_coef_x0", alphas / etas.clamp(min=1e-12))

        wt = alphas / (2 * (kappa ** 2) * etas * etas_prev.clamp(min=1e-12))
        wt[0] = alphas[0] / (2 * (kappa ** 2) * etas[0] * etas[0])
        self.register_buffer("loss_weight", wt)

    @staticmethod
    def _extract(arr: torch.Tensor, t: torch.Tensor, x_shape: torch.Size) -> torch.Tensor:
        out = arr.to(t.device).gather(0, t)
        return out.reshape(t.shape[0], *([1] * (len(x_shape) - 1)))

    def q_sample(self, x0: torch.Tensor, y0: torch.Tensor, t: torch.Tensor,
                 noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        """q(x_t|x0,y0) = N(x0 + eta_t*e0, kappa^2*eta_t*I), e0 = y0 - x0 (Eq. 2)."""
        if noise is None:
            noise = torch.randn_like(x0)
        e0 = y0 - x0
        eta_t = self._extract(self.etas, t, x0.shape)
        mean = x0 + eta_t * e0
        std = self.kappa * eta_t.sqrt()
        return mean + std * noise

    def q_posterior_mean_variance(self, x0: torch.Tensor, xt: torch.Tensor,
                                   t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """q(x_{t-1}|x_t,x0,y0) (Eq. 6)."""
        coef_xt = self._extract(self.posterior_mean_coef_xt, t, xt.shape)
        coef_x0 = self._extract(self.posterior_mean_coef_x0, t, xt.shape)
        mean = coef_xt * xt + coef_x0 * x0
        var = self._extract(self.posterior_variance, t, xt.shape)
        return mean, var


# ============================================================
# Frozen MST++ + ResShift residual-shifting diffusion model
# ============================================================

class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        if half <= 1:
            frequencies = torch.ones(1, device=t.device, dtype=t.dtype)
        else:
            frequencies = torch.exp(
                -math.log(10000.0)
                * torch.arange(half, device=t.device, dtype=t.dtype)
                / float(half - 1)
            )

        angles = t[:, None] * frequencies[None, :]
        embedding = torch.cat([angles.sin(), angles.cos()], dim=-1)
        if embedding.shape[-1] < self.dim:
            embedding = F.pad(embedding, (0, self.dim - embedding.shape[-1]))
        return embedding


class MSTResShiftDenoiser(nn.Module):
    """
    f_theta(x_t, y0, t): predicts x0 (Eq. 7-8 parameterization), expressed as a
    residual added to the frozen MST++ coarse prediction y0, i.e.
    x0_pred = y0 + predicted_residual, so predicted_residual approximates
    ground_truth_hsi - coarse_hsi (= -e0).
    """

    def __init__(
        self,
        hsi_channels: int = 31,
        rgb_channels: int = 3,
        n_feat: int = 31,
        body_depth: int = 3,
        mst_stage: int = 2,
        num_blocks: Tuple[int, ...] = (1, 1, 1),
        total_steps: int = 15,
    ):
        super().__init__()
        self.pad_multiple = 2 ** mst_stage
        self.total_steps = total_steps

        input_channels = hsi_channels + hsi_channels + rgb_channels
        self.conv_in = nn.Conv2d(
            input_channels, n_feat, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(n_feat),
            nn.Linear(n_feat, n_feat * 4),
            nn.GELU(),
            nn.Linear(n_feat * 4, n_feat),
        )
        self.body = nn.Sequential(
            *[
                MST(
                    in_dim=n_feat,
                    out_dim=n_feat,
                    dim=n_feat,
                    stage=mst_stage,
                    num_blocks=list(num_blocks),
                )
                for _ in range(body_depth)
            ]
        )
        # No sigmoid: an HSI residual must be allowed to be positive or negative.
        self.conv_out = nn.Conv2d(
            n_feat, hsi_channels, kernel_size=3, stride=1, padding=1, bias=False
        )

    def forward(
        self,
        x_t: torch.Tensor,
        coarse_hsi: torch.Tensor,
        rgb: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        if x_t.shape != coarse_hsi.shape:
            raise ValueError(
                "x_t and coarse_hsi must have the same shape; "
                f"received {tuple(x_t.shape)} and {tuple(coarse_hsi.shape)}"
            )
        if rgb.shape[0] != x_t.shape[0] or rgb.shape[-2:] != x_t.shape[-2:]:
            raise ValueError("RGB and HSI tensors must share batch and spatial dimensions")

        _, _, original_h, original_w = x_t.shape
        pad_h = (self.pad_multiple - original_h % self.pad_multiple) % self.pad_multiple
        pad_w = (self.pad_multiple - original_w % self.pad_multiple) % self.pad_multiple

        inputs = torch.cat([x_t, coarse_hsi, rgb], dim=1)
        if pad_h or pad_w:
            mode = (
                "reflect"
                if original_h > pad_h and original_w > pad_w
                else "replicate"
            )
            inputs = F.pad(inputs, (0, pad_w, 0, pad_h), mode=mode)

        features = self.conv_in(inputs)
        # t is 0-indexed here; normalize by T so the embedding spans [0, (T-1)/T].
        normalized_t = t.float() / float(self.total_steps)
        time_features = self.time_mlp(normalized_t).to(features.dtype)
        features = features + time_features[:, :, None, None]
        features = self.body(features)
        residual = self.conv_out(features)
        return residual[:, :, :original_h, :original_w]


class ResShiftDiffusion(nn.Module):
    """Wraps the schedule and denoiser and exposes training/sampling entry points."""

    def __init__(self, denoiser: nn.Module, schedule: ResShiftSchedule):
        super().__init__()
        self.denoiser = denoiser
        self.schedule = schedule
        self.T = schedule.T

    def training_predictions(
        self,
        rgb: torch.Tensor,
        coarse_hsi: torch.Tensor,
        ground_truth: torch.Tensor,
        t: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        batch_size = ground_truth.shape[0]
        if t is None:
            t = torch.randint(
                low=0,
                high=self.T,
                size=(batch_size,),
                device=ground_truth.device,
                dtype=torch.long,
            )

        x_t = self.schedule.q_sample(ground_truth, coarse_hsi, t)
        predicted_residual = self.denoiser(
            x_t=x_t,
            coarse_hsi=coarse_hsi,
            rgb=rgb,
            t=t,
        )
        target_residual = ground_truth - coarse_hsi
        reconstruction = coarse_hsi + predicted_residual  # x0_pred, Eq. (7)

        return {
            "t": t,
            "x_t": x_t,
            "target_residual": target_residual,
            "predicted_residual": predicted_residual,
            "reconstruction": reconstruction,
            "loss_weight": self.schedule.loss_weight.to(ground_truth.device).gather(0, t),
        }

    @torch.no_grad()
    def sample(
        self,
        rgb: torch.Tensor,
        coarse_hsi: torch.Tensor,
        clip_denoised: bool,
        stochastic: bool = True,
    ) -> torch.Tensor:
        """Full T-step reverse Markov chain (Eq. 4): x_T ~ N(y0, kappa^2 I) down to x_0."""
        batch_size = coarse_hsi.shape[0]
        x_t = coarse_hsi + self.schedule.kappa * torch.randn_like(coarse_hsi)

        for step in reversed(range(self.T)):
            t = torch.full((batch_size,), step, device=coarse_hsi.device, dtype=torch.long)
            predicted_residual = self.denoiser(
                x_t=x_t,
                coarse_hsi=coarse_hsi,
                rgb=rgb,
                t=t,
            )
            x0_hat = coarse_hsi + predicted_residual
            if clip_denoised:
                x0_hat = x0_hat.clamp(0.0, METRIC_DATA_RANGE)

            mean, var = self.schedule.q_posterior_mean_variance(x0_hat, x_t, t)
            if stochastic and step > 0:
                # posterior_variance is exactly 0 at step 0 (eta_0=0), so this
                # branch is skipped automatically at the last step regardless.
                x_t = mean + var.sqrt() * torch.randn_like(x_t)
            else:
                x_t = mean

        return x_t


class MSTPlusPlusResShift(nn.Module):
    def __init__(self, coarse_model: nn.Module, diffusion: ResShiftDiffusion):
        super().__init__()
        self.coarse_model = coarse_model
        self.diffusion = diffusion
        self.freeze_coarse_model()

    def freeze_coarse_model(self) -> None:
        self.coarse_model.requires_grad_(False)
        self.coarse_model.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        # super().train() changes every child module; immediately restore MST++.
        self.coarse_model.eval()
        return self

    def get_coarse(self, rgb: torch.Tensor) -> torch.Tensor:
        self.coarse_model.eval()
        with torch.no_grad():
            return unwrap_prediction(self.coarse_model(rgb))

    def forward(
        self,
        rgb: torch.Tensor,
        ground_truth: torch.Tensor,
        t: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        coarse_hsi = self.get_coarse(rgb)
        outputs = self.diffusion.training_predictions(
            rgb=rgb,
            coarse_hsi=coarse_hsi,
            ground_truth=ground_truth,
            t=t,
        )
        outputs["coarse_hsi"] = coarse_hsi
        return outputs

    @torch.no_grad()
    def reconstruct(
        self,
        rgb: torch.Tensor,
        clip_denoised: bool,
        stochastic: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        coarse_hsi = self.get_coarse(rgb)
        refined_hsi = self.diffusion.sample(
            rgb=rgb,
            coarse_hsi=coarse_hsi,
            clip_denoised=clip_denoised,
            stochastic=stochastic,
        )
        return coarse_hsi, refined_hsi


def load_checkpoint_file(path: Path, map_location) -> dict:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def extract_model_state_dict(checkpoint) -> Dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict):
        for key in ("model_state_dict", "state_dict", "params", "model"):
            value = checkpoint.get(key)
            if isinstance(value, dict) and value:
                return value
        if checkpoint and all(isinstance(value, torch.Tensor) for value in checkpoint.values()):
            return checkpoint
    raise KeyError(
        "Could not find a model state dictionary. Expected model_state_dict, "
        "state_dict, params, model, or a direct tensor dictionary."
    )


def remove_prefix_if_present(
    state_dict: Dict[str, torch.Tensor],
    prefix: str,
) -> Dict[str, torch.Tensor]:
    if state_dict and all(key.startswith(prefix) for key in state_dict):
        return {key[len(prefix):]: value for key, value in state_dict.items()}
    return state_dict


def load_frozen_mstpp(
    checkpoint_path: Path,
    device: torch.device,
) -> MST_Plus_Plus:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"MST++ checkpoint does not exist: {checkpoint_path}")

    checkpoint = load_checkpoint_file(checkpoint_path, map_location="cpu")
    checkpoint_config = checkpoint.get("model_config", {}) if isinstance(checkpoint, dict) else {}

    model = MST_Plus_Plus(
        in_channels=int(checkpoint_config.get("in_channels", 3)),
        out_channels=int(checkpoint_config.get("out_channels", HSI_CHANNELS)),
        n_feat=int(checkpoint_config.get("n_feat", MSTPP_FEATURES)),
        stage=int(checkpoint_config.get("stage", MSTPP_STAGES)),
    )

    state_dict = extract_model_state_dict(checkpoint)
    state_dict = remove_prefix_if_present(state_dict, "module.")
    state_dict = remove_prefix_if_present(state_dict, "coarse_model.")
    model.load_state_dict(state_dict, strict=True)

    model.requires_grad_(False)
    model.eval()
    return model.to(device)


def current_resshift_config() -> dict:
    return {
        "hsi_channels": HSI_CHANNELS,
        "rgb_channels": 3,
        "n_feat": RESSHIFT_FEATURES,
        "body_depth": RESSHIFT_BODY_DEPTH,
        "mst_stage": RESSHIFT_MST_STAGE,
        "num_blocks": list(RESSHIFT_NUM_BLOCKS),
        "T": RESSHIFT_TIMESTEPS,
        "kappa": RESSHIFT_KAPPA,
        "p": RESSHIFT_P,
        "min_noise_level": RESSHIFT_MIN_NOISE_LEVEL,
    }


def build_pipeline(
    device: torch.device,
    resshift_config: Optional[dict] = None,
) -> MSTPlusPlusResShift:
    config = current_resshift_config() if resshift_config is None else resshift_config
    coarse_model = load_frozen_mstpp(Path(MSTPP_CHECKPOINT), device)

    denoiser = MSTResShiftDenoiser(
        hsi_channels=int(config.get("hsi_channels", HSI_CHANNELS)),
        rgb_channels=int(config.get("rgb_channels", 3)),
        n_feat=int(config.get("n_feat", RESSHIFT_FEATURES)),
        body_depth=int(config.get("body_depth", RESSHIFT_BODY_DEPTH)),
        mst_stage=int(config.get("mst_stage", RESSHIFT_MST_STAGE)),
        num_blocks=tuple(config.get("num_blocks", RESSHIFT_NUM_BLOCKS)),
        total_steps=int(config.get("T", RESSHIFT_TIMESTEPS)),
    )
    schedule = ResShiftSchedule(
        T=int(config.get("T", RESSHIFT_TIMESTEPS)),
        kappa=float(config.get("kappa", RESSHIFT_KAPPA)),
        p=float(config.get("p", RESSHIFT_P)),
        min_noise_level=float(config.get("min_noise_level", RESSHIFT_MIN_NOISE_LEVEL)),
    )
    diffusion = ResShiftDiffusion(denoiser=denoiser, schedule=schedule)
    return MSTPlusPlusResShift(coarse_model, diffusion).to(device)


def assert_mstpp_is_frozen(model: MSTPlusPlusResShift) -> None:
    trainable = [name for name, p in model.coarse_model.named_parameters() if p.requires_grad]
    if trainable:
        raise RuntimeError(f"MST++ contains trainable parameters: {trainable[:5]}")
    if model.coarse_model.training:
        raise RuntimeError("Frozen MST++ must remain in evaluation mode")


# ============================================================
# ResShift losses, training, validation, and checkpoints
# ============================================================

def spectral_cosine_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Stable differentiable spectral-direction loss: mean(1 - cosine)."""
    prediction = prediction.float()
    target = target.float()
    dot_product = (prediction * target).sum(dim=1)
    prediction_norm = prediction.square().sum(dim=1).sqrt()
    target_norm = target.square().sum(dim=1).sqrt()
    cosine = dot_product / (prediction_norm * target_norm).clamp_min(eps)
    return (1.0 - cosine.clamp(-1.0, 1.0)).mean()


def calculate_resshift_losses(
    predicted_residual: torch.Tensor,
    target_residual: torch.Tensor,
    reconstruction: torch.Tensor,
    target_hsi: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    predicted_residual = predicted_residual.float()
    target_residual = target_residual.float()
    reconstruction = reconstruction.float()
    target_hsi = target_hsi.float()

    # Reproduces the spirit of Eq. (8) (min ||f_theta(x_t,y0,t) - x0||^2), applied
    # to the residual parameterization; the paper notes an unweighted objective
    # (dropping wt) performs best in practice, so wt is not applied here either.
    residual_loss = F.smooth_l1_loss(
        predicted_residual,
        target_residual,
        beta=SMOOTH_L1_BETA,
    )
    reconstruction_value = F.l1_loss(reconstruction, target_hsi)
    spectral_value = spectral_cosine_loss(reconstruction, target_hsi)
    total = (
        RESIDUAL_LOSS_WEIGHT * residual_loss
        + RECONSTRUCTION_LOSS_WEIGHT * reconstruction_value
        + SPECTRAL_LOSS_WEIGHT * spectral_value
    )
    return {
        "total_loss": total,
        "residual_loss": residual_loss,
        "reconstruction_loss": reconstruction_value,
        "spectral_loss": spectral_value,
    }


def empty_epoch_sums() -> Dict[str, float]:
    return {
        "total_loss": 0.0,
        "residual_loss": 0.0,
        "reconstruction_loss": 0.0,
        "spectral_loss": 0.0,
        "coarse_mrae": 0.0,
        "mrae": 0.0,
        "rmse": 0.0,
        "sam": 0.0,
        "psnr": 0.0,
        "ssim": 0.0,
    }


def add_batch_results(
    sums: Dict[str, float],
    losses: Dict[str, torch.Tensor],
    coarse_metrics: Dict[str, torch.Tensor],
    refined_metrics: Dict[str, torch.Tensor],
    batch_size: int,
) -> None:
    for name, value in losses.items():
        sums[name] += float(value.detach().item()) * batch_size
    sums["coarse_mrae"] += float(coarse_metrics["mrae"].detach().sum().item())
    for name, values in refined_metrics.items():
        sums[name] += float(values.detach().sum().item())


def average_epoch_sums(sums: Dict[str, float], count: int) -> Dict[str, float]:
    if count == 0:
        raise RuntimeError("No samples were processed")
    return {name: value / count for name, value in sums.items()}


def print_resshift_metrics(prefix: str, values: Dict[str, float]) -> None:
    sam_label = "SAM(deg)" if REPORT_SAM_IN_DEGREES else "SAM(rad)"
    print(
        f"{prefix} | "
        f"Total: {values['total_loss']:.6f} | "
        f"Residual: {values['residual_loss']:.6f} | "
        f"Recon-L1: {values['reconstruction_loss']:.6f} | "
        f"Spectral: {values['spectral_loss']:.6f} | "
        f"Coarse MRAE: {values['coarse_mrae']:.6f} | "
        f"Refined MRAE: {values['mrae']:.6f} | "
        f"RMSE: {values['rmse']:.6f} | "
        f"{sam_label}: {values['sam']:.6f} | "
        f"PSNR: {values['psnr']:.4f} dB | "
        f"SSIM: {values['ssim']:.4f}"
    )


def train_one_epoch(
    model: MSTPlusPlusResShift,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    use_amp: bool,
) -> Dict[str, float]:
    model.train()
    assert_mstpp_is_frozen(model)

    sums = empty_epoch_sums()
    sample_count = 0
    trainable_parameters = [
        parameter for parameter in model.diffusion.parameters() if parameter.requires_grad
    ]

    for batch_index, (rgb, hsi) in enumerate(loader, start=1):
        rgb = rgb.to(device, non_blocking=True)
        hsi = hsi.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        if batch_index == 1:
            print_range_diagnostics(rgb, hsi, "  Train")

        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            outputs = model(rgb=rgb, ground_truth=hsi)

        # Losses are always evaluated in float32 outside autocast.
        with torch.amp.autocast(device_type=device.type, enabled=False):
            losses = calculate_resshift_losses(
                predicted_residual=outputs["predicted_residual"],
                target_residual=outputs["target_residual"],
                reconstruction=outputs["reconstruction"],
                target_hsi=hsi,
            )
        total_loss = losses["total_loss"]
        if not torch.isfinite(total_loss):
            raise FloatingPointError(
                f"Non-finite ResShift training loss at batch {batch_index}: "
                f"{total_loss.item()}"
            )

        scaler.scale(total_loss).backward()
        scaler.unscale_(optimizer)

        gradients_are_finite = all(
            parameter.grad is None or torch.isfinite(parameter.grad).all().item()
            for parameter in trainable_parameters
        )
        if not gradients_are_finite:
            if not use_amp:
                raise FloatingPointError(
                    f"Non-finite ResShift gradients at batch {batch_index} with AMP disabled"
                )
            previous_scale = scaler.get_scale()
            optimizer.zero_grad(set_to_none=True)
            scaler.update()
            print(
                f"WARNING: skipped batch {batch_index} because AMP gradients "
                f"overflowed; scale {previous_scale:g} -> {scaler.get_scale():g}"
            )
            continue

        gradient_norm = torch.nn.utils.clip_grad_norm_(
            trainable_parameters,
            GRADIENT_CLIP_NORM,
            error_if_nonfinite=True,
        )
        if not torch.isfinite(gradient_norm):
            raise FloatingPointError(
                f"Non-finite gradient norm at batch {batch_index}: {gradient_norm.item()}"
            )

        scaler.step(optimizer)
        scaler.update()

        batch_size = rgb.shape[0]
        coarse_metrics = calculate_metric_tensors(outputs["coarse_hsi"], hsi)
        refined_metrics = calculate_metric_tensors(outputs["reconstruction"], hsi)
        add_batch_results(sums, losses, coarse_metrics, refined_metrics, batch_size)
        sample_count += batch_size

        if batch_index % PRINT_EVERY == 0 or batch_index == len(loader):
            print_resshift_metrics(
                f"  Train batch {batch_index:04d}/{len(loader):04d}",
                average_epoch_sums(sums, sample_count),
            )

    return average_epoch_sums(sums, sample_count)


@torch.no_grad()
def validate(
    model: MSTPlusPlusResShift,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool,
) -> Dict[str, float]:
    if VALIDATION_MODE not in {"endpoint", "sample"}:
        raise ValueError("VALIDATION_MODE must be 'endpoint' or 'sample'")

    model.eval()
    assert_mstpp_is_frozen(model)
    sums = empty_epoch_sums()
    sample_count = 0

    for batch_index, (rgb, hsi) in enumerate(loader, start=1):
        rgb = rgb.to(device, non_blocking=True)
        hsi = hsi.to(device, non_blocking=True)

        if batch_index == 1:
            print_range_diagnostics(rgb, hsi, "  Validation")

        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            if VALIDATION_MODE == "sample":
                coarse_hsi, prediction = model.reconstruct(
                    rgb=rgb,
                    clip_denoised=CLAMP_INFERENCE_OUTPUT,
                    stochastic=VALIDATION_STOCHASTIC,
                )
                predicted_residual = prediction - coarse_hsi
                target_residual = hsi - coarse_hsi
            else:
                # Fast, deterministic single-pass check: run the denoiser once
                # with x_t == coarse_hsi (a stand-in for the noisy last state).
                coarse_hsi = model.get_coarse(rgb)
                endpoint_t = torch.full(
                    (rgb.shape[0],),
                    model.diffusion.T - 1,
                    device=device,
                    dtype=torch.long,
                )
                predicted_residual = model.diffusion.denoiser(
                    x_t=coarse_hsi,
                    coarse_hsi=coarse_hsi,
                    rgb=rgb,
                    t=endpoint_t,
                )
                prediction = coarse_hsi + predicted_residual
                target_residual = hsi - coarse_hsi

        with torch.amp.autocast(device_type=device.type, enabled=False):
            losses = calculate_resshift_losses(
                predicted_residual=predicted_residual,
                target_residual=target_residual,
                reconstruction=prediction,
                target_hsi=hsi,
            )

        batch_size = rgb.shape[0]
        coarse_metrics = calculate_metric_tensors(coarse_hsi, hsi)
        refined_metrics = calculate_metric_tensors(prediction, hsi)
        add_batch_results(sums, losses, coarse_metrics, refined_metrics, batch_size)
        sample_count += batch_size

    return average_epoch_sums(sums, sample_count)


def save_checkpoint(
    path: Path,
    model: MSTPlusPlusResShift,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler,
    epoch: int,
    best_mrae: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            # Intentionally save only the denoiser weights; frozen MST++ stays external.
            "diffusion_state_dict": model.diffusion.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "best_mrae": best_mrae,
            "mstpp_checkpoint": str(MSTPP_CHECKPOINT),
            "normalization": NORMALIZATION,
            "validation_mode": VALIDATION_MODE,
            "resshift_config": current_resshift_config(),
            "loss_config": {
                "residual_weight": RESIDUAL_LOSS_WEIGHT,
                "reconstruction_weight": RECONSTRUCTION_LOSS_WEIGHT,
                "spectral_weight": SPECTRAL_LOSS_WEIGHT,
                "smooth_l1_beta": SMOOTH_L1_BETA,
            },
            "metric_config": {
                "mrae_epsilon": MRAE_EPSILON,
                "sam_epsilon": SAM_EPSILON,
                "data_range": METRIC_DATA_RANGE,
                "sam_in_degrees": REPORT_SAM_IN_DEGREES,
            },
        },
        path,
    )


def load_diffusion_checkpoint(
    model: MSTPlusPlusResShift,
    checkpoint: dict,
) -> None:
    state_dict = checkpoint.get("diffusion_state_dict")
    if not isinstance(state_dict, dict):
        raise KeyError("ResShift checkpoint does not contain diffusion_state_dict")
    state_dict = remove_prefix_if_present(state_dict, "module.")
    state_dict = remove_prefix_if_present(state_dict, "diffusion.")
    model.diffusion.load_state_dict(state_dict, strict=True)


def train() -> None:
    set_seed(SEED)
    device = get_device()
    use_amp = USE_AMP and device.type == "cuda"

    all_pairs = filter_valid_pairs(build_pairs())
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

    resume_data = None
    resshift_config = current_resshift_config()
    if RESUME_CHECKPOINT:
        resume_path = Path(RESUME_CHECKPOINT)
        if not resume_path.exists():
            raise FileNotFoundError(f"Resume checkpoint does not exist: {resume_path}")
        resume_data = load_checkpoint_file(resume_path, map_location="cpu")
        resshift_config = resume_data.get("resshift_config", resshift_config)

    model = build_pipeline(device=device, resshift_config=resshift_config)
    assert_mstpp_is_frozen(model)

    trainable_parameters = [
        parameter for parameter in model.diffusion.parameters() if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=EPOCHS,
        eta_min=MIN_LEARNING_RATE,
    )
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=use_amp,
        init_scale=AMP_INITIAL_SCALE,
    )

    start_epoch = 1
    best_mrae = float("inf")
    if resume_data is not None:
        load_diffusion_checkpoint(model, resume_data)
        optimizer.load_state_dict(resume_data["optimizer_state_dict"])
        scheduler.load_state_dict(resume_data["scheduler_state_dict"])
        if "scaler_state_dict" in resume_data:
            scaler.load_state_dict(resume_data["scaler_state_dict"])
        start_epoch = int(resume_data["epoch"]) + 1
        best_mrae = float(resume_data.get("best_mrae", best_mrae))
        print(f"Resumed ResShift from epoch {start_epoch - 1}: {RESUME_CHECKPOINT}")

    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    frozen_count = sum(parameter.numel() for parameter in model.coarse_model.parameters())
    trainable_count = sum(parameter.numel() for parameter in trainable_parameters)
    print(f"Device: {device} | AMP: {use_amp}")
    print(f"Frozen MST++ checkpoint: {MSTPP_CHECKPOINT}")
    print(f"Frozen MST++ parameters: {frozen_count:,}")
    print(f"Trainable ResShift parameters: {trainable_count:,}")
    print(f"ResShift schedule: T={RESSHIFT_TIMESTEPS}, kappa={RESSHIFT_KAPPA}, p={RESSHIFT_P}")
    print(f"Training pairs: {len(training_pairs)}")
    print(f"Validation pairs: {len(validation_pairs)}")
    print(f"Validation mode: {VALIDATION_MODE}")
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

        print_resshift_metrics("Train", training_metrics)
        print_resshift_metrics("Validation", validation_metrics)
        print(f"Learning rate: {optimizer.param_groups[0]['lr']:.2e}")

        improved = validation_metrics["mrae"] < best_mrae
        if improved:
            best_mrae = validation_metrics["mrae"]

        save_checkpoint(
            output_dir / "last_resshift.pth",
            model,
            optimizer,
            scheduler,
            scaler,
            epoch,
            best_mrae,
        )
        if improved:
            save_checkpoint(
                output_dir / "best_resshift.pth",
                model,
                optimizer,
                scheduler,
                scaler,
                epoch,
                best_mrae,
            )
            print(
                "Saved new best ResShift checkpoint with validation refined MRAE "
                f"{best_mrae:.6f}"
            )


# ============================================================
# Full-resolution ResShift inference
# ============================================================

@torch.no_grad()
def infer() -> None:
    device = get_device()
    use_amp = USE_AMP and device.type == "cuda"

    resshift_checkpoint_path = Path(INFERENCE_RESSHIFT_CHECKPOINT)
    rgb_path = Path(INFERENCE_RGB_PATH)
    if not resshift_checkpoint_path.exists():
        raise FileNotFoundError(
            f"ResShift inference checkpoint does not exist: {resshift_checkpoint_path}"
        )
    if not rgb_path.exists():
        raise FileNotFoundError(f"Inference RGB image does not exist: {rgb_path}")

    checkpoint = load_checkpoint_file(resshift_checkpoint_path, map_location="cpu")
    model = build_pipeline(
        device=device,
        resshift_config=checkpoint.get("resshift_config", current_resshift_config()),
    )
    load_diffusion_checkpoint(model, checkpoint)
    model.eval()
    assert_mstpp_is_frozen(model)

    rgb = torch.from_numpy(load_rgb(rgb_path)).unsqueeze(0).to(device)
    with torch.amp.autocast(device_type=device.type, enabled=use_amp):
        coarse_hsi, refined_hsi = model.reconstruct(
            rgb=rgb,
            clip_denoised=CLAMP_INFERENCE_OUTPUT,
            stochastic=INFERENCE_STOCHASTIC,
        )

    coarse_hsi = coarse_hsi.float()
    refined_hsi = refined_hsi.float()
    predicted_residual = refined_hsi - coarse_hsi

    save_dir = Path(INFERENCE_OUTPUT_DIR)
    save_dir.mkdir(parents=True, exist_ok=True)
    stem = rgb_path.stem

    outputs = {
        "coarse": coarse_hsi[0].cpu().numpy(),
        "refined": refined_hsi[0].cpu().numpy(),
        "predicted_residual": predicted_residual[0].cpu().numpy(),
    }
    for name, array_chw in outputs.items():
        np.save(save_dir / f"{stem}_{name}.npy", array_chw)
        sio.savemat(
            save_dir / f"{stem}_{name}.mat",
            {HSI_KEY: array_chw.transpose(1, 2, 0)},
        )
    print(f"Coarse, refined, and residual predictions saved to: {save_dir}")

    if INFERENCE_HSI_PATH is None or str(INFERENCE_HSI_PATH).strip() == "":
        print("INFERENCE_HSI_PATH is not set; metrics and heatmaps were skipped")
        return

    target_path = Path(INFERENCE_HSI_PATH)
    if not target_path.exists():
        raise FileNotFoundError(
            f"Inference ground-truth HSI does not exist: {target_path}"
        )
    target = torch.from_numpy(normalize_hsi(load_hsi(target_path))).unsqueeze(0).to(device)
    if refined_hsi.shape != target.shape:
        raise ValueError(
            "Prediction and ground truth have different shapes: "
            f"prediction={tuple(refined_hsi.shape)}, target={tuple(target.shape)}"
        )

    coarse_metrics = calculate_metrics(coarse_hsi, target)
    refined_metrics = calculate_metrics(refined_hsi, target)
    print(
        "Coarse MST++ | "
        f"MRAE: {coarse_metrics['mrae']:.6f} | "
        f"RMSE: {coarse_metrics['rmse']:.6f} | "
        f"SAM: {coarse_metrics['sam']:.6f} | "
        f"PSNR: {coarse_metrics['psnr']:.4f} | "
        f"SSIM: {coarse_metrics['ssim']:.4f}"
    )
    print(
        "Refined ResShift | "
        f"MRAE: {refined_metrics['mrae']:.6f} | "
        f"RMSE: {refined_metrics['rmse']:.6f} | "
        f"SAM: {refined_metrics['sam']:.6f} | "
        f"PSNR: {refined_metrics['psnr']:.4f} | "
        f"SSIM: {refined_metrics['ssim']:.4f}"
    )

    heatmap_module = ResidualHeatmap(HEATMAP_REDUCTION).to(device)
    for name, prediction in (("coarse", coarse_hsi), ("refined", refined_hsi)):
        heatmap = heatmap_module(prediction, target)
        png_path = save_dir / f"{stem}_{name}_{HEATMAP_REDUCTION}_heatmap.png"
        npy_path = save_dir / f"{stem}_{name}_{HEATMAP_REDUCTION}_heatmap.npy"
        save_heatmap(
            heatmap,
            png_path,
            f"{name.capitalize()} {HEATMAP_REDUCTION.upper()} spectral residual: {stem}",
        )
        np.save(npy_path, heatmap[0, 0].cpu().numpy())


# ============================================================
# Parser: --mode is deliberately the only parser argument
# ============================================================

def parse_mode() -> str:
    parser = argparse.ArgumentParser(
        "Frozen MST++ with ResShift residual-shifting diffusion training and inference"
    )
    parser.add_argument(
        "--mode",
        choices=["train", "infer"],
        required=True,
        help="Run ResShift training or full reverse-diffusion inference",
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
