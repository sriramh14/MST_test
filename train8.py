"""
Train and visualize the frozen-MST++ Residual Diffusion Bridge Model (RDBM)
for RGB-to-HSI reconstruction.

Only the execution mode is controlled from the command line:

    python train_rdbm_hsi.py --mode train
    python train_rdbm_hsi.py --mode visualize
    python train_rdbm_hsi.py --mode train_visualize

All dataset, model, optimization, metric, and visualization settings remain as
constants in the configuration section below.

Expected model interface
------------------------
This script imports the self-contained model created previously:

    RDBMHSI

The model uses:
    ground-truth HSI x0 <-> frozen MST++ coarse HSI mu

Training uses the corrected residual-modulated forward bridge:

    pi  = x0 - mu
    x_t = mu + pi * Theta_t + pi * Sigma_t * epsilon

The U-Net predicts x0 as mu plus a learned residual, and the configured L1/L2
loss is applied between the predicted and true x0. Validation uses one random
timestep per image and reports the same objective loss plus x_t MRAE.
Visualization and full validation inference retain MRAE, RMSE, SAM, PSNR, and
SSIM.
"""

from __future__ import annotations

import argparse
import hashlib
import random
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import h5py
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset

# -----------------------------------------------------------------------------
# Change this import path to match your project structure.
# -----------------------------------------------------------------------------
from model.RDBM_rgb2hsi import RDBMHSI, extract

# REVISION: all HSI metrics, including one-step MRAE, use project loss modules.
# Use the project metric implementations exactly as in the supplied scripts.
from loss.mrae import mrae
from loss.psnr import psnr
from loss.rmse import rmse
from loss.sam import sam
from loss.ssim import ssim


# =============================================================================
# Configuration: edit values here
# =============================================================================

TRAIN_HSI_DIR = (
    "/kaggle/input/datasets/sriramhari14/ntire-2022/"
    "Train_spectral/Train_spectral"
)
TRAIN_RGB_DIR = (
    "/kaggle/input/datasets/sriramhari14/ntire-2022/"
    "Train_RGB/Train_RGB"
)
VALIDATION_HSI_DIR = (
    "/kaggle/input/datasets/sriramhari14/ntire-2022/"
    "Valid_spectral/Valid_spectral"
)
VALIDATION_RGB_DIR = (
    "/kaggle/input/datasets/sriramhari14/ntire-2022/"
    "Valid_RGB/Valid_RGB"
)

MST_CHECKPOINT = "./mst_checkpoints/mst_plus_plus.pth"

# Set this to match the MST++ checkpoint architecture.
# For a trained single-stage MST++ checkpoint, keep this as 1.
# For the common three-stage MST++ model, set this to 3.
MST_NUM_STAGES = 1

# Set this to the constructor keyword used by your MST++ implementation.
# Set it to None only when no explicit stage-count argument is required:
#   MST_STAGE_PARAMETER_NAME = "stage"
#   MST_STAGE_PARAMETER_NAME = "num_stages"
#   MST_STAGE_PARAMETER_NAME = "stages"
#   MST_STAGE_PARAMETER_NAME = "n_stages"
MST_STAGE_PARAMETER_NAME: Optional[str] = "stage"

# Other arguments passed to MST_Plus_Plus. Do not duplicate the stage-count
# argument here unless you intentionally want to override MST_NUM_STAGES.
MST_MODEL_KWARGS: Dict[str, Any] = {}
STRICT_MST_CHECKPOINT = True
# RDBMHSI expects MST++ to return its HSI prediction directly as a tensor.

OUTPUT_DIR = Path("./mstpp_rdbm_checkpoints")
BEST_CHECKPOINT = OUTPUT_DIR / "best_mstpp_rdbm.pth"
LAST_CHECKPOINT = OUTPUT_DIR / "last_mstpp_rdbm.pth"
RESUME_CHECKPOINT: Optional[str] = None

VISUALIZATION_CHECKPOINT = BEST_CHECKPOINT
VISUALIZATION_DIR = Path("./mstpp_rdbm_visualizations")
VISUALIZATION_FILE = VISUALIZATION_DIR / "random_validation_visualization.png"

HSI_KEY = "cube"
HSI_CHANNELS = 31
SUPPORTED_HSI_EXTENSIONS = {".npy", ".npz", ".mat", ".pt", ".pth"}
SUPPORTED_RGB_EXTENSIONS = {".png", ".jpg", ".jpeg", ".npy", ".pt", ".pth"}

# "none", "minmax", or "band_minmax".
# NTIRE spectral reconstruction cubes are normally already in a common range,
# so "none" is usually appropriate.
HSI_NORMALIZATION = "none"

TRAIN_PAIR_VALIDATION_CACHE = OUTPUT_DIR / "training_pair_validation_cache.pth"
VALIDATION_PAIR_VALIDATION_CACHE = OUTPUT_DIR / "validation_pair_validation_cache.pth"

# Residual Diffusion Bridge settings.
NUM_TIMESTEPS = 100
SAMPLE_STEPS = 20
RDBM_LAMBDA = 10.0 / 255.0
LOSS_TYPE = "l1"
OBJECTIVE = "pred_x_start"
SAMPLING_TYPE = "pred_x_start"

# RDBM U-Net settings. The corrected model file exposes the base width and
# channel multipliers directly; the remaining U-Net details live in that file.
TRAIN_CROP_SIZE = 256
VALIDATION_CROP_SIZE: Optional[int] = TRAIN_CROP_SIZE
UNET_MODEL_CHANNELS = 32
UNET_CHANNEL_MULT = (1, 2, 4, 4)

MODEL_DOWNSAMPLE_FACTOR = 2 ** (len(UNET_CHANNEL_MULT) - 1)

# Dataset and augmentation settings.
PATCHES_PER_IMAGE = 2
USE_AUGMENTATION = True

# Training settings.
BATCH_SIZE = 2
VALIDATION_BATCH_SIZE = 2
NUM_EPOCHS = 40
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
MIN_LEARNING_RATE = 1e-7
GRADIENT_CLIP_NORM = 1.0
NUM_WORKERS = 4
PRINT_EVERY = 30
SEED = 42

# AMP settings. Autocast is used only for the training forward pass. RDBM
# reverse sampling and all validation metrics are deliberately computed in
# float32 for numerical stability.
USE_AMP = True
PREFER_BFLOAT16 = True
FP16_INITIAL_SCALE = 1024.0
FP16_GROWTH_INTERVAL = 2000

# Prediction setting. Set to None when HSI values are not bounded to [0, 1].
# MRAE, RMSE, SAM, PSNR, and SSIM are imported from the project's loss folder.
PREDICTION_CLAMP_RANGE: Optional[Tuple[float, float]] = (0.0, 1.0)

# Checkpoint selection. Lower validation objective loss is better.
BEST_METRIC_NAME = "loss"

# Visualization settings.
NUM_VISUALIZATION_IMAGES = 5
VISUALIZATION_BANDS = (20, 10, 2)
FIGURE_DPI = 180


# =============================================================================
# MST++ constructor arguments
# =============================================================================

def build_mst_kwargs() -> Dict[str, Any]:
    """Build the keyword arguments passed to MST_Plus_Plus by RDBMHSI."""
    kwargs = dict(MST_MODEL_KWARGS)
    if MST_STAGE_PARAMETER_NAME is not None:
        kwargs.setdefault(MST_STAGE_PARAMETER_NAME, MST_NUM_STAGES)
    return kwargs


# =============================================================================
# Reproducibility and safe AMP helpers
# =============================================================================

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def get_amp_dtype(device: torch.device) -> torch.dtype:
    if (
        device.type == "cuda"
        and PREFER_BFLOAT16
        and hasattr(torch.cuda, "is_bf16_supported")
        and torch.cuda.is_bf16_supported()
    ):
        return torch.bfloat16
    return torch.float16


def autocast_context(device: torch.device, enabled: bool):
    if not enabled or device.type != "cuda":
        return nullcontext()
    return torch.autocast(
        device_type="cuda",
        dtype=get_amp_dtype(device),
        enabled=True,
    )


def make_grad_scaler(device: torch.device, use_amp: bool):
    scaler_enabled = (
        use_amp
        and device.type == "cuda"
        and get_amp_dtype(device) == torch.float16
    )

    # New API first, with a fallback for older PyTorch versions.
    try:
        return torch.amp.GradScaler(
            "cuda",
            enabled=scaler_enabled,
            init_scale=FP16_INITIAL_SCALE,
            growth_interval=FP16_GROWTH_INTERVAL,
        )
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(
            enabled=scaler_enabled,
            init_scale=FP16_INITIAL_SCALE,
            growth_interval=FP16_GROWTH_INTERVAL,
        )


# =============================================================================
# HSI/RGB loading
# =============================================================================

def _extract_3d_array_from_mapping(
    data: dict,
    file_path: Path,
    preferred_key: Optional[str] = None,
) -> np.ndarray:
    if preferred_key is not None and preferred_key in data:
        value = data[preferred_key]
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
        if isinstance(value, np.ndarray) and value.ndim == 3:
            return value

    candidates: List[np.ndarray] = []
    for key, value in data.items():
        if str(key).startswith("__"):
            continue
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
        if (
            isinstance(value, np.ndarray)
            and value.ndim == 3
            and np.issubdtype(value.dtype, np.number)
        ):
            candidates.append(value)

    if not candidates:
        raise ValueError(
            f"No numeric three-dimensional array was found in {file_path}."
        )
    return max(candidates, key=lambda array: array.size)


def load_mat_v73(
    file_path: Path,
    preferred_key: Optional[str] = None,
) -> np.ndarray:
    candidates: List[Tuple[str, np.ndarray]] = []

    with h5py.File(str(file_path), "r") as h5_file:
        if (
            preferred_key is not None
            and preferred_key in h5_file
            and isinstance(h5_file[preferred_key], h5py.Dataset)
            and h5_file[preferred_key].ndim == 3
        ):
            candidates.append(
                (preferred_key, np.asarray(h5_file[preferred_key]))
            )

        if not candidates:

            def visitor(name, obj):
                if not isinstance(obj, h5py.Dataset) or obj.ndim != 3:
                    return
                try:
                    if np.issubdtype(obj.dtype, np.number):
                        candidates.append((name, np.asarray(obj)))
                except TypeError:
                    return

            h5_file.visititems(visitor)

    if not candidates:
        raise ValueError(
            f"No numeric three-dimensional HSI dataset was found in {file_path}."
        )

    _, cube = max(candidates, key=lambda item: item[1].size)
    # MATLAB v7.3/HDF5 arrays are commonly stored with reversed dimensions.
    return np.transpose(cube, axes=tuple(range(cube.ndim - 1, -1, -1)))


def load_hsi_file(file_path: Path) -> np.ndarray:
    extension = file_path.suffix.lower()

    if extension == ".npy":
        cube = np.load(file_path)
    elif extension == ".npz":
        with np.load(file_path) as loaded:
            candidates = [
                loaded[key]
                for key in loaded.files
                if loaded[key].ndim == 3
            ]
            if not candidates:
                raise ValueError(
                    f"No three-dimensional array was found in {file_path}."
                )
            cube = max(candidates, key=lambda array: array.size)
    elif extension == ".mat":
        try:
            loaded = sio.loadmat(file_path)
            cube = _extract_3d_array_from_mapping(
                loaded,
                file_path=file_path,
                preferred_key=HSI_KEY,
            )
        except (NotImplementedError, ValueError):
            cube = load_mat_v73(
                file_path=file_path,
                preferred_key=HSI_KEY,
            )
    elif extension in {".pt", ".pth"}:
        try:
            loaded = torch.load(
                file_path,
                map_location="cpu",
                weights_only=False,
            )
        except TypeError:
            loaded = torch.load(file_path, map_location="cpu")

        if isinstance(loaded, torch.Tensor):
            cube = loaded.detach().cpu().numpy()
        elif isinstance(loaded, np.ndarray):
            cube = loaded
        elif isinstance(loaded, dict):
            cube = _extract_3d_array_from_mapping(
                loaded,
                file_path=file_path,
                preferred_key=HSI_KEY,
            )
        else:
            raise TypeError(
                f"Unsupported object type in {file_path}: {type(loaded)}"
            )
    else:
        raise ValueError(f"Unsupported HSI extension: {extension}")

    cube = np.asarray(cube, dtype=np.float32)
    cube = np.squeeze(cube)
    if cube.ndim != 3:
        raise ValueError(
            f"Expected a three-dimensional HSI cube in {file_path}, "
            f"but found shape {cube.shape}."
        )
    return cube


def convert_hsi_to_chw(
    cube: np.ndarray,
    hsi_channels: int,
    file_path: Path,
) -> np.ndarray:
    if cube.shape[0] == hsi_channels:
        return np.ascontiguousarray(cube)
    if cube.shape[-1] == hsi_channels:
        return np.ascontiguousarray(np.transpose(cube, (2, 0, 1)))
    if cube.shape[1] == hsi_channels:
        return np.ascontiguousarray(np.transpose(cube, (1, 0, 2)))
    raise ValueError(
        f"Could not identify the spectral axis in {file_path}. "
        f"Found shape {cube.shape}; expected {hsi_channels} bands."
    )


def load_rgb_file(file_path: Path) -> np.ndarray:
    extension = file_path.suffix.lower()

    if extension in {".png", ".jpg", ".jpeg"}:
        image = Image.open(file_path).convert("RGB")
        array = np.asarray(image, dtype=np.float32) / 255.0
        return np.ascontiguousarray(np.transpose(array, (2, 0, 1)))

    if extension == ".npy":
        array = np.load(file_path).astype(np.float32)
    elif extension in {".pt", ".pth"}:
        try:
            loaded = torch.load(
                file_path,
                map_location="cpu",
                weights_only=False,
            )
        except TypeError:
            loaded = torch.load(file_path, map_location="cpu")

        if isinstance(loaded, torch.Tensor):
            array = loaded.detach().cpu().float().numpy()
        elif isinstance(loaded, np.ndarray):
            array = loaded.astype(np.float32)
        else:
            raise TypeError(
                f"Unsupported RGB object in {file_path}: {type(loaded)}"
            )
    else:
        raise ValueError(f"Unsupported RGB extension: {extension}")

    array = np.squeeze(array)
    if array.ndim == 2:
        array = np.stack([array, array, array], axis=0)
    elif array.ndim == 3 and array.shape[0] == 3:
        pass
    elif array.ndim == 3 and array.shape[-1] == 3:
        array = np.transpose(array, (2, 0, 1))
    else:
        raise ValueError(
            f"Could not convert RGB file {file_path} to CHW. "
            f"Found shape {array.shape}."
        )

    array = np.asarray(array, dtype=np.float32)
    if np.nanmax(array) > 1.5:
        array = array / 255.0
    return np.ascontiguousarray(array)


def normalize_hsi_cube(cube: np.ndarray, mode: str) -> np.ndarray:
    if mode == "none":
        return cube
    if mode == "minmax":
        minimum = float(cube.min())
        maximum = float(cube.max())
        return (cube - minimum) / (maximum - minimum + 1e-8)
    if mode == "band_minmax":
        minimum = cube.min(axis=(1, 2), keepdims=True)
        maximum = cube.max(axis=(1, 2), keepdims=True)
        return (cube - minimum) / (maximum - minimum + 1e-8)
    raise ValueError(f"Unknown HSI normalization mode: {mode}")


# =============================================================================
# File discovery, pairing, metadata checking, and cache
# =============================================================================

def find_files(
    directory: str,
    extensions: Sequence[str],
    kind: str,
) -> List[Path]:
    root = Path(directory)
    if not root.exists():
        raise FileNotFoundError(f"{kind} directory does not exist: {root}")

    files = sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in extensions
    )
    if not files:
        raise RuntimeError(f"No supported {kind} files were found in {root}.")
    return files


def _index_unique_stems(
    files: Sequence[Path],
    kind: str,
) -> Dict[str, Path]:
    index: Dict[str, Path] = {}
    for path in files:
        if path.stem in index:
            raise RuntimeError(
                f"Duplicate {kind} filename stem '{path.stem}'.\n"
                f"First:  {index[path.stem]}\n"
                f"Second: {path}"
            )
        index[path.stem] = path
    return index


def pair_hsi_rgb_files(
    hsi_directory: str,
    rgb_directory: str,
) -> List[Tuple[Path, Path]]:
    hsi_files = find_files(
        hsi_directory,
        SUPPORTED_HSI_EXTENSIONS,
        "HSI",
    )
    rgb_files = find_files(
        rgb_directory,
        SUPPORTED_RGB_EXTENSIONS,
        "RGB",
    )

    hsi_by_stem = _index_unique_stems(hsi_files, "HSI")
    rgb_by_stem = _index_unique_stems(rgb_files, "RGB")

    shared_stems = sorted(set(hsi_by_stem) & set(rgb_by_stem))
    missing_rgb = sorted(set(hsi_by_stem) - set(rgb_by_stem))
    missing_hsi = sorted(set(rgb_by_stem) - set(hsi_by_stem))

    if missing_rgb:
        print(
            f"Warning: {len(missing_rgb)} HSI files have no matching RGB file."
        )
    if missing_hsi:
        print(
            f"Warning: {len(missing_hsi)} RGB files have no matching HSI file."
        )
    if not shared_stems:
        raise RuntimeError(
            "No paired HSI/RGB files were found. "
            "Paired files must have identical stems."
        )

    pairs = [
        (hsi_by_stem[stem], rgb_by_stem[stem])
        for stem in shared_stems
    ]
    print(
        f"Found {len(pairs)} paired files in:\n"
        f"  HSI: {hsi_directory}\n"
        f"  RGB: {rgb_directory}"
    )
    return pairs


def make_files_fingerprint(files: Sequence[Path]) -> str:
    records = []
    for file_path in files:
        stat = file_path.stat()
        records.append(
            f"{file_path.resolve()}|{stat.st_size}|{stat.st_mtime_ns}"
        )
    return hashlib.sha256(
        "\n".join(records).encode("utf-8")
    ).hexdigest()


def is_possible_hsi_shape(
    shape: Sequence[int],
    hsi_channels: int,
) -> bool:
    return (
        len(shape) == 3
        and hsi_channels in shape
        and all(int(size) > 0 for size in shape)
    )


def inspect_hdf5_mat_file(
    file_path: Path,
    hsi_channels: int,
    hsi_key: str,
) -> None:
    candidates: List[Tuple[str, Tuple[int, ...]]] = []

    with h5py.File(str(file_path), "r") as h5_file:
        if (
            hsi_key in h5_file
            and isinstance(h5_file[hsi_key], h5py.Dataset)
        ):
            dataset = h5_file[hsi_key]
            candidates.append(
                (
                    hsi_key,
                    tuple(int(value) for value in dataset.shape),
                )
            )
        else:

            def visitor(name, obj):
                if not isinstance(obj, h5py.Dataset) or obj.ndim != 3:
                    return
                try:
                    if np.issubdtype(obj.dtype, np.number):
                        candidates.append(
                            (
                                name,
                                tuple(int(value) for value in obj.shape),
                            )
                        )
                except TypeError:
                    return

            h5_file.visititems(visitor)

    if not candidates:
        raise ValueError(
            f"No numerical three-dimensional dataset was found in {file_path}."
        )
    if not any(
        is_possible_hsi_shape(shape, hsi_channels)
        for _, shape in candidates
    ):
        raise ValueError(
            f"No {hsi_channels}-band cube was found in {file_path}. "
            f"HDF5 datasets: {candidates}"
        )


def inspect_standard_mat_file(
    file_path: Path,
    hsi_channels: int,
    hsi_key: str,
) -> None:
    try:
        metadata = sio.whosmat(file_path)
    except (NotImplementedError, ValueError, OSError):
        inspect_hdf5_mat_file(
            file_path=file_path,
            hsi_channels=hsi_channels,
            hsi_key=hsi_key,
        )
        return

    candidates = [
        (name, tuple(int(value) for value in shape))
        for name, shape, _ in metadata
        if len(shape) == 3
    ]
    if not candidates:
        raise ValueError(
            f"No three-dimensional array was found in {file_path}."
        )

    preferred = [
        candidate
        for candidate in candidates
        if candidate[0] == hsi_key
    ]
    arrays_to_check = preferred if preferred else candidates
    if not any(
        is_possible_hsi_shape(shape, hsi_channels)
        for _, shape in arrays_to_check
    ):
        raise ValueError(
            f"No {hsi_channels}-band cube was found in {file_path}. "
            f"MATLAB arrays: {candidates}"
        )


def inspect_hsi_file_metadata(
    file_path: Path,
    hsi_channels: int,
    hsi_key: str,
) -> None:
    if file_path.suffix.lower() == ".mat":
        inspect_standard_mat_file(
            file_path=file_path,
            hsi_channels=hsi_channels,
            hsi_key=hsi_key,
        )
        return

    cube = load_hsi_file(file_path)
    if not is_possible_hsi_shape(cube.shape, hsi_channels):
        raise ValueError(
            f"Invalid HSI shape {cube.shape} in {file_path}."
        )


def filter_valid_pairs(
    pairs: Sequence[Tuple[Path, Path]],
    hsi_channels: int,
    log_path: Path,
    cache_path: Path,
) -> List[Tuple[Path, Path]]:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    pairs = list(pairs)
    hsi_files = [hsi_path for hsi_path, _ in pairs]
    fingerprint = make_files_fingerprint(hsi_files)
    pair_lookup = {
        str(hsi_path.resolve()): (hsi_path, rgb_path)
        for hsi_path, rgb_path in pairs
    }

    if cache_path.exists():
        try:
            try:
                cached = torch.load(
                    cache_path,
                    map_location="cpu",
                    weights_only=False,
                )
            except TypeError:
                cached = torch.load(cache_path, map_location="cpu")

            if (
                isinstance(cached, dict)
                and cached.get("fingerprint") == fingerprint
            ):
                valid_paths = cached.get("valid_hsi_paths", [])
                invalid_records = cached.get("invalid_records", [])
                valid_pairs = [
                    pair_lookup[path]
                    for path in valid_paths
                    if path in pair_lookup
                ]
                print(f"\nUsing cached pair validation: {cache_path}")
                print(
                    f"Valid pairs: {len(valid_pairs)} | "
                    f"Invalid: {len(invalid_records)}"
                )
                if valid_pairs:
                    return valid_pairs
        except Exception as error:
            print(
                "\nCould not use the validation cache. "
                "The dataset will be checked again. "
                f"Reason: {error}"
            )

    print("\nChecking HSI file metadata before use...")
    valid_pairs: List[Tuple[Path, Path]] = []
    invalid_records: List[dict] = []

    for index, (hsi_path, rgb_path) in enumerate(pairs, start=1):
        try:
            inspect_hsi_file_metadata(
                file_path=hsi_path,
                hsi_channels=hsi_channels,
                hsi_key=HSI_KEY,
            )
            valid_pairs.append((hsi_path, rgb_path))
        except Exception as error:
            invalid_records.append(
                {
                    "path": str(hsi_path.resolve()),
                    "error": f"{type(error).__name__}: {error}",
                }
            )
            print(
                "\nSkipping invalid HSI file:\n"
                f"  File: {hsi_path}\n"
                f"  Error: {error}"
            )

        if index % 100 == 0 or index == len(pairs):
            print(
                f"Checked {index}/{len(pairs)} | "
                f"Valid: {len(valid_pairs)} | "
                f"Invalid: {len(invalid_records)}"
            )

    if not valid_pairs:
        raise RuntimeError(
            "No valid HSI/RGB pairs remain after metadata validation."
        )

    if invalid_records:
        with log_path.open("w", encoding="utf-8") as log_file:
            for record in invalid_records:
                log_file.write(
                    f"{record['path']} | {record['error']}\n"
                )
        print(f"Invalid-file log saved to: {log_path}")

    torch.save(
        {
            "fingerprint": fingerprint,
            "valid_hsi_paths": [
                str(hsi_path.resolve())
                for hsi_path, _ in valid_pairs
            ],
            "invalid_records": invalid_records,
        },
        cache_path,
    )
    print(f"Validation cache saved to: {cache_path}")
    return valid_pairs


# =============================================================================
# Paired spatial transforms
# =============================================================================

def _pad_tensor_to_minimum_size(
    tensor: torch.Tensor,
    minimum_height: int,
    minimum_width: int,
) -> torch.Tensor:
    _, height, width = tensor.shape
    pad_height = max(0, minimum_height - height)
    pad_width = max(0, minimum_width - width)
    if pad_height == 0 and pad_width == 0:
        return tensor
    return F.pad(
        tensor,
        (0, pad_width, 0, pad_height),
        mode="replicate",
    )


def random_crop_pair(
    hsi: torch.Tensor,
    rgb: torch.Tensor,
    crop_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    hsi = _pad_tensor_to_minimum_size(
        hsi,
        crop_size,
        crop_size,
    )
    rgb = _pad_tensor_to_minimum_size(
        rgb,
        crop_size,
        crop_size,
    )
    _, height, width = hsi.shape
    top = random.randint(0, height - crop_size)
    left = random.randint(0, width - crop_size)
    return (
        hsi[:, top:top + crop_size, left:left + crop_size],
        rgb[:, top:top + crop_size, left:left + crop_size],
    )


def center_crop_pair(
    hsi: torch.Tensor,
    rgb: torch.Tensor,
    crop_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    hsi = _pad_tensor_to_minimum_size(
        hsi,
        crop_size,
        crop_size,
    )
    rgb = _pad_tensor_to_minimum_size(
        rgb,
        crop_size,
        crop_size,
    )
    _, height, width = hsi.shape
    top = (height - crop_size) // 2
    left = (width - crop_size) // 2
    return (
        hsi[:, top:top + crop_size, left:left + crop_size],
        rgb[:, top:top + crop_size, left:left + crop_size],
    )


def augment_pair(
    hsi: torch.Tensor,
    rgb: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if random.random() < 0.5:
        hsi = torch.flip(hsi, dims=[1])
        rgb = torch.flip(rgb, dims=[1])
    if random.random() < 0.5:
        hsi = torch.flip(hsi, dims=[2])
        rgb = torch.flip(rgb, dims=[2])

    rotations = random.randint(0, 3)
    if rotations:
        hsi = torch.rot90(hsi, k=rotations, dims=(1, 2))
        rgb = torch.rot90(rgb, k=rotations, dims=(1, 2))

    return hsi.contiguous(), rgb.contiguous()


def pad_pair_to_multiple(
    hsi: torch.Tensor,
    rgb: torch.Tensor,
    multiple: int,
) -> Tuple[torch.Tensor, torch.Tensor, int, int]:
    _, original_height, original_width = hsi.shape
    pad_height = (multiple - original_height % multiple) % multiple
    pad_width = (multiple - original_width % multiple) % multiple

    if pad_height == 0 and pad_width == 0:
        return hsi, rgb, original_height, original_width

    hsi = F.pad(
        hsi,
        (0, pad_width, 0, pad_height),
        mode="replicate",
    )
    rgb = F.pad(
        rgb,
        (0, pad_width, 0, pad_height),
        mode="replicate",
    )
    return hsi, rgb, original_height, original_width


# =============================================================================
# Dataset and DataLoader
# =============================================================================

class HSIRGBPairDataset(Dataset):
    def __init__(
        self,
        pairs: Sequence[Tuple[Path, Path]],
        hsi_channels: int,
        crop_size: Optional[int],
        patches_per_image: int,
        training: bool,
        normalization: str,
        augment: bool,
        return_paths: bool = False,
    ) -> None:
        self.pairs = list(pairs)
        self.hsi_channels = hsi_channels
        self.crop_size = crop_size
        self.patches_per_image = patches_per_image
        self.training = training
        self.normalization = normalization
        self.augment = augment
        self.return_paths = return_paths

        if training and crop_size is None:
            raise ValueError("Training requires a finite crop_size.")
        if patches_per_image < 1:
            raise ValueError(
                "patches_per_image must be at least 1."
            )

    def __len__(self) -> int:
        multiplier = self.patches_per_image if self.training else 1
        return len(self.pairs) * multiplier

    def _load_pair(
        self,
        pair_index: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, Path, Path]:
        hsi_path, rgb_path = self.pairs[pair_index]

        hsi_array = convert_hsi_to_chw(
            load_hsi_file(hsi_path),
            hsi_channels=self.hsi_channels,
            file_path=hsi_path,
        )
        hsi_array = normalize_hsi_cube(
            hsi_array,
            mode=self.normalization,
        )
        rgb_array = load_rgb_file(rgb_path)

        if hsi_array.shape[1:] != rgb_array.shape[1:]:
            raise ValueError(
                f"Spatial mismatch for pair {hsi_path.stem}: "
                f"HSI={hsi_array.shape[1:]}, "
                f"RGB={rgb_array.shape[1:]}."
            )
        if not np.isfinite(hsi_array).all():
            raise ValueError(f"HSI contains NaN/Inf: {hsi_path}")
        if not np.isfinite(rgb_array).all():
            raise ValueError(f"RGB contains NaN/Inf: {rgb_path}")

        hsi = torch.from_numpy(hsi_array.copy()).float()
        rgb = torch.from_numpy(rgb_array.copy()).float()
        return hsi, rgb, hsi_path, rgb_path

    def __getitem__(self, index: int):
        pair_index = (
            index // self.patches_per_image
            if self.training
            else index
        )
        hsi, rgb, hsi_path, rgb_path = self._load_pair(pair_index)

        if self.crop_size is not None:
            if self.training:
                hsi, rgb = random_crop_pair(
                    hsi,
                    rgb,
                    self.crop_size,
                )
            else:
                hsi, rgb = center_crop_pair(
                    hsi,
                    rgb,
                    self.crop_size,
                )

        if self.training and self.augment:
            hsi, rgb = augment_pair(hsi, rgb)

        if self.return_paths:
            return hsi, rgb, str(hsi_path), str(rgb_path)
        return hsi, rgb


def make_loader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    drop_last: bool,
    device: torch.device,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(SEED)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=NUM_WORKERS,
        pin_memory=(device.type == "cuda"),
        drop_last=drop_last,
        persistent_workers=(NUM_WORKERS > 0),
        worker_init_fn=seed_worker,
        generator=generator,
    )


# =============================================================================
# Project metric functions
# =============================================================================

def _metric_to_float(value: Any, name: str) -> float:
    """Convert a project metric result to one Python scalar."""
    if not torch.is_tensor(value):
        value = torch.as_tensor(value)

    value = value.detach().float()
    if value.numel() != 1:
        value = value.mean()

    result = float(value.item())
    if not np.isfinite(result) and not (
        name == "PSNR" and result == float("inf")
    ):
        raise FloatingPointError(
            f"{name} returned a non-finite value: {result}"
        )
    return result


@torch.no_grad()
def calculate_validation_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> Dict[str, float]:
    """Evaluate a batch using the existing project metric functions.

    The argument order follows the supplied project code:
    ``metric(target, reconstruction)``. Each image is evaluated separately so
    dataset averages are independent of the final DataLoader batch size.
    Returned values are sums across the input batch.
    """
    prediction = prediction.detach().float()
    target = target.detach().float()

    if prediction.shape != target.shape:
        raise ValueError(
            f"Metric shape mismatch: prediction={tuple(prediction.shape)}, "
            f"target={tuple(target.shape)}"
        )
    if prediction.ndim != 4:
        raise ValueError(
            "Project metrics expect prediction and target in BCHW format."
        )
    if not torch.isfinite(prediction).all():
        raise FloatingPointError(
            "Validation prediction contains NaN or Inf."
        )
    if not torch.isfinite(target).all():
        raise FloatingPointError(
            "Validation target contains NaN or Inf."
        )

    metric_sums = {
        "mrae": 0.0,
        "psnr": 0.0,
        "rmse": 0.0,
        "sam": 0.0,
        "ssim": 0.0,
    }

    for sample_index in range(prediction.shape[0]):
        sample_prediction = prediction[
            sample_index:sample_index + 1
        ]
        sample_target = target[sample_index:sample_index + 1]

        metric_sums["mrae"] += _metric_to_float(
            mrae(sample_target, sample_prediction), "MRAE"
        )
        metric_sums["psnr"] += _metric_to_float(
            psnr(sample_target, sample_prediction), "PSNR"
        )
        metric_sums["rmse"] += _metric_to_float(
            rmse(sample_target, sample_prediction), "RMSE"
        )
        metric_sums["sam"] += _metric_to_float(
            sam(sample_target, sample_prediction), "SAM"
        )
        metric_sums["ssim"] += _metric_to_float(
            ssim(sample_target, sample_prediction), "SSIM"
        )

    return metric_sums


@torch.no_grad()
def calculate_single_image_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> Dict[str, float]:
    """Evaluate one BCHW image with the imported project metrics."""
    if prediction.shape[0] != 1 or target.shape[0] != 1:
        raise ValueError(
            "calculate_single_image_metrics expects batch size 1."
        )
    return calculate_validation_metrics(
        prediction=prediction,
        target=target,
    )


@torch.no_grad()
def calculate_batch_metric_sums(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> Dict[str, float]:
    """Return per-metric sums for a BCHW batch."""
    return calculate_validation_metrics(
        prediction=prediction,
        target=target,
    )


# =============================================================================
# Pair preparation
# =============================================================================

def prepare_training_and_validation_pairs() -> Tuple[
    List[Tuple[Path, Path]],
    List[Tuple[Path, Path]],
]:
    train_pairs = pair_hsi_rgb_files(
        TRAIN_HSI_DIR,
        TRAIN_RGB_DIR,
    )
    validation_pairs = pair_hsi_rgb_files(
        VALIDATION_HSI_DIR,
        VALIDATION_RGB_DIR,
    )

    train_pairs = filter_valid_pairs(
        pairs=train_pairs,
        hsi_channels=HSI_CHANNELS,
        log_path=OUTPUT_DIR / "invalid_training_pairs.txt",
        cache_path=TRAIN_PAIR_VALIDATION_CACHE,
    )
    validation_pairs = filter_valid_pairs(
        pairs=validation_pairs,
        hsi_channels=HSI_CHANNELS,
        log_path=OUTPUT_DIR / "invalid_validation_pairs.txt",
        cache_path=VALIDATION_PAIR_VALIDATION_CACHE,
    )
    return train_pairs, validation_pairs


def prepare_validation_pairs() -> List[Tuple[Path, Path]]:
    validation_pairs = pair_hsi_rgb_files(
        VALIDATION_HSI_DIR,
        VALIDATION_RGB_DIR,
    )
    return filter_valid_pairs(
        pairs=validation_pairs,
        hsi_channels=HSI_CHANNELS,
        log_path=OUTPUT_DIR / "invalid_validation_pairs.txt",
        cache_path=VALIDATION_PAIR_VALIDATION_CACHE,
    )


# =============================================================================
# Model and checkpoint helpers
# =============================================================================

def load_torch_checkpoint(
    path: str | Path,
    device: str | torch.device = "cpu",
):
    try:
        return torch.load(
            path,
            map_location=device,
            weights_only=False,
        )
    except TypeError:
        return torch.load(path, map_location=device)


def build_model(device: torch.device) -> RDBMHSI:
    model = RDBMHSI(
        mst_ckpt=MST_CHECKPOINT,
        hsi_channels=HSI_CHANNELS,
        unet_dim=UNET_MODEL_CHANNELS,
        dim_mults=UNET_CHANNEL_MULT,
        timesteps=NUM_TIMESTEPS,
        sampling_timesteps=SAMPLE_STEPS,
        image_size=TRAIN_CROP_SIZE,
        objective=OBJECTIVE,
        sampling_type=SAMPLING_TYPE,
        lamb=RDBM_LAMBDA,
        loss_type=LOSS_TYPE,
        mst_kwargs=build_mst_kwargs(),
        mst_strict=STRICT_MST_CHECKPOINT,
        check_finite=True,
    )
    return model.to(device)


def rdbm_state_dict(
    model: RDBMHSI,
) -> Dict[str, torch.Tensor]:
    """Save the trainable RDBM and its fixed analytical schedule buffers."""
    return {
        key: value.detach().cpu()
        for key, value in model.rdbm.state_dict().items()
    }


def load_rdbm_state_dict(
    model: RDBMHSI,
    state_dict: Dict[str, torch.Tensor],
) -> None:
    if state_dict and all(
        key.startswith("module.")
        for key in state_dict
    ):
        state_dict = {
            key[len("module."):]: value
            for key, value in state_dict.items()
        }

    incompatible = model.rdbm.load_state_dict(
        state_dict,
        strict=True,
    )
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            "RDBM checkpoint mismatch. "
            f"Missing keys: {incompatible.missing_keys}; "
            f"unexpected keys: {incompatible.unexpected_keys}"
        )


def model_config_dictionary() -> dict:
    return {
        "hsi_channels": HSI_CHANNELS,
        "train_crop_size": TRAIN_CROP_SIZE,
        "num_timesteps": NUM_TIMESTEPS,
        "sample_steps": SAMPLE_STEPS,
        "rdbm_lambda": RDBM_LAMBDA,
        "loss_type": LOSS_TYPE,
        "objective": OBJECTIVE,
        "sampling_type": SAMPLING_TYPE,
        "unet_model_channels": UNET_MODEL_CHANNELS,
        "unet_channel_mult": UNET_CHANNEL_MULT,
        "mst_checkpoint": MST_CHECKPOINT,
        "mst_num_stages": MST_NUM_STAGES,
        "mst_stage_parameter_name": MST_STAGE_PARAMETER_NAME,
        "mst_model_kwargs": MST_MODEL_KWARGS,
    }


def save_training_checkpoint(
    path: Path,
    model: RDBMHSI,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: Any,
    epoch: int,
    best_metric: float,
    training_metrics: dict,
    validation_metrics: dict,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "rdbm_state_dict": rdbm_state_dict(model),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "best_metric": best_metric,
            "best_metric_name": BEST_METRIC_NAME,
            "training_metrics": training_metrics,
            "validation_metrics": validation_metrics,
            "model_config": model_config_dictionary(),
            "mst_checkpoint": MST_CHECKPOINT,
        },
        path,
    )


def load_model_for_visualization(
    checkpoint_path: str | Path,
    device: torch.device,
) -> RDBMHSI:
    model = build_model(device)
    checkpoint = load_torch_checkpoint(
        checkpoint_path,
        device="cpu",
    )

    if (
        isinstance(checkpoint, dict)
        and isinstance(checkpoint.get("rdbm_state_dict"), dict)
    ):
        state_dict = checkpoint["rdbm_state_dict"]
    elif (
        isinstance(checkpoint, dict)
        and isinstance(checkpoint.get("bridge_state_dict"), dict)
    ):
        # Backward-compatible name used by the attached training template.
        state_dict = checkpoint["bridge_state_dict"]
    elif isinstance(checkpoint, dict) and all(
        torch.is_tensor(value)
        for value in checkpoint.values()
    ):
        state_dict = checkpoint
    else:
        raise KeyError(
            f"Could not find rdbm_state_dict in {checkpoint_path}."
        )

    load_rdbm_state_dict(model, state_dict)
    model.eval()
    print(f"Loaded RDBM checkpoint: {checkpoint_path}")
    return model


def clamp_prediction(prediction: torch.Tensor) -> torch.Tensor:
    if PREDICTION_CLAMP_RANGE is None:
        return prediction
    minimum, maximum = PREDICTION_CLAMP_RANGE
    return prediction.clamp(minimum, maximum)


def coarse_estimate(model: RDBMHSI, rgb: torch.Tensor) -> torch.Tensor:
    """Return the frozen MST++ endpoint mu."""
    return model._frozen_mst_predict(rgb)


def one_step_rdbm_forward(
    model: RDBMHSI,
    rgb: torch.Tensor,
    hsi: torch.Tensor,
    *,
    timesteps: Optional[torch.Tensor] = None,
    noise: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Run one residual-modulated bridge training/validation step.

    This is the same objective implemented by ``RDBMHSI.forward``, but it also
    returns the intermediate tensors needed for the unchanged one-step metrics.
    """
    mu = coarse_estimate(model, rgb)
    if mu.shape != hsi.shape:
        raise ValueError(
            f"MST++ output shape {tuple(mu.shape)} does not match "
            f"ground-truth HSI shape {tuple(hsi.shape)}."
        )

    batch_size = hsi.shape[0]
    if timesteps is None:
        timesteps = torch.randint(
            0,
            model.rdbm.num_timesteps,
            (batch_size,),
            device=hsi.device,
            dtype=torch.long,
        )
    if noise is None:
        noise = torch.randn_like(hsi)

    true_xt = model.rdbm.q_sample(
        x_start=hsi,
        mu=mu,
        t=timesteps,
        noise=noise,
    )
    predicted_x0 = model.unet(true_xt, mu, timesteps)

    if model.rdbm.loss_type == "l1":
        loss = F.l1_loss(predicted_x0, hsi)
    elif model.rdbm.loss_type == "l2":
        loss = F.mse_loss(predicted_x0, hsi)
    else:
        raise NotImplementedError(
            f"Unsupported RDBM loss type: {model.rdbm.loss_type}"
        )

    theta_t = extract(model.rdbm.Theta, timesteps, hsi.shape)
    sigma_t = extract(model.rdbm.Sigma, timesteps, hsi.shape)
    predicted_residual = predicted_x0 - mu
    predicted_xt = (
        mu
        + predicted_residual * theta_t
        + predicted_residual * sigma_t * noise
    )

    return loss, {
        "mu": mu,
        "timesteps": timesteps,
        "noise": noise,
        "true_xt": true_xt,
        "predicted_xt": predicted_xt,
        "x0_recon": predicted_x0,
    }


# =============================================================================
# Training and validation
# =============================================================================

def train_one_epoch(
    model: RDBMHSI,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: Any,
    device: torch.device,
    use_amp: bool,
) -> dict:
    model.train()
    trainable_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]

    loss_sum = 0.0
    reconstructed_mrae_sum = 0.0
    sample_count = 0
    skipped_batches = 0

    for batch_index, (hsi, rgb) in enumerate(loader, start=1):
        hsi = hsi.to(
            device,
            non_blocking=True,
            dtype=torch.float32,
        )
        rgb = rgb.to(
            device,
            non_blocking=True,
            dtype=torch.float32,
        )
        optimizer.zero_grad(set_to_none=True)

        with autocast_context(device, use_amp):
            loss, outputs = one_step_rdbm_forward(
                model=model,
                rgb=rgb,
                hsi=hsi,
            )

        if not torch.isfinite(loss):
            skipped_batches += 1
            print(
                f"  Warning: skipped batch {batch_index} because the "
                f"training loss was non-finite: {float(loss.detach())}"
            )
            optimizer.zero_grad(set_to_none=True)
            continue

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)

        gradient_norm = nn.utils.clip_grad_norm_(
            trainable_parameters,
            max_norm=GRADIENT_CLIP_NORM,
            error_if_nonfinite=False,
        )

        if not torch.isfinite(torch.as_tensor(gradient_norm)):
            skipped_batches += 1
            optimizer.zero_grad(set_to_none=True)
            scaler.update()
            print(
                f"  Warning: skipped batch {batch_index} because the "
                "unscaled gradient norm was non-finite."
            )
            continue

        scaler.step(optimizer)
        scaler.update()

        batch_size = hsi.shape[0]
        loss_sum += float(loss.detach().float()) * batch_size

        x0_recon = outputs["x0_recon"].detach().float()
        reconstructed_mrae = _metric_to_float(
            mrae(hsi.detach().float(), x0_recon),
            "MRAE",
        )
        reconstructed_mrae_sum += reconstructed_mrae * batch_size
        sample_count += batch_size

        if batch_index % PRINT_EVERY == 0 or batch_index == len(loader):
            denominator = max(sample_count, 1)
            print(
                f"  Batch {batch_index:04d}/{len(loader):04d} | "
                f"loss={loss_sum / denominator:.6f} | "
                f"one-step MRAE={reconstructed_mrae_sum / denominator:.6f} | "
                f"grad={float(gradient_norm):.4f} | "
                f"AMP-skipped={skipped_batches}"
            )

    if sample_count == 0:
        raise RuntimeError(
            "Every training batch was skipped. Check input scaling, the "
            "learning rate, and the selected AMP dtype."
        )

    return {
        "loss": loss_sum / sample_count,
        "one_step_mrae": reconstructed_mrae_sum / sample_count,
        "evaluated_samples": sample_count,
        "skipped_batches": skipped_batches,
    }


@torch.no_grad()
def validate_one_epoch(
    model: RDBMHSI,
    loader: DataLoader,
    device: torch.device,
) -> dict:
    """One-step random-timestep validation matching RDBM training.

    For every validation batch this function keeps the paper's residual
    modulation in the forward state, predicts x0 with one U-Net call, and
    reports the configured x0 objective loss and MRAE between the rebuilt
    predicted x_t and the true x_t.
    """
    model.eval()

    objective_loss_sum = 0.0
    xt_mrae_sum = 0.0
    sample_count = 0

    try:
        timestep_generator = torch.Generator(device=device)
        noise_generator = torch.Generator(device=device)
    except TypeError:
        timestep_generator = torch.Generator(device=device.type)
        noise_generator = torch.Generator(device=device.type)

    timestep_generator.manual_seed(SEED + 10_000)
    noise_generator.manual_seed(SEED + 20_000)

    for batch_index, (hsi, rgb) in enumerate(loader, start=1):
        hsi = hsi.to(
            device,
            non_blocking=True,
            dtype=torch.float32,
        )
        rgb = rgb.to(
            device,
            non_blocking=True,
            dtype=torch.float32,
        )
        batch_size = hsi.shape[0]

        timesteps = torch.randint(
            low=0,
            high=model.rdbm.num_timesteps,
            size=(batch_size,),
            device=device,
            generator=timestep_generator,
            dtype=torch.long,
        )
        noise = torch.randn(
            hsi.shape,
            generator=noise_generator,
            device=device,
            dtype=torch.float32,
        )

        objective_loss, outputs = one_step_rdbm_forward(
            model=model,
            rgb=rgb,
            hsi=hsi,
            timesteps=timesteps,
            noise=noise,
        )

        true_xt = outputs["true_xt"].float()
        predicted_xt = outputs["predicted_xt"].float()
        xt_mrae = _metric_to_float(
            mrae(true_xt, predicted_xt),
            "MRAE",
        )

        if not torch.isfinite(objective_loss):
            raise FloatingPointError(
                "Validation objective loss is non-finite at "
                f"batch {batch_index}."
            )
        if not np.isfinite(xt_mrae):
            raise FloatingPointError(
                f"Validation x_t MRAE is non-finite at batch {batch_index}."
            )

        objective_loss_sum += float(objective_loss.detach()) * batch_size
        xt_mrae_sum += xt_mrae * batch_size
        sample_count += batch_size

        print(
            f"  Validation batch {batch_index:04d}/{len(loader):04d} | "
            f"objective loss={objective_loss_sum / sample_count:.6f} | "
            f"x_t MRAE={xt_mrae_sum / sample_count:.6f}"
        )

    if sample_count == 0:
        raise RuntimeError("The validation DataLoader produced no samples.")

    return {
        "loss": objective_loss_sum / sample_count,
        "objective_loss": objective_loss_sum / sample_count,
        # Kept as an alias so existing logging/analysis code does not break.
        "noise_loss": objective_loss_sum / sample_count,
        "mrae": xt_mrae_sum / sample_count,
        "xt_mrae": xt_mrae_sum / sample_count,
        "evaluated_images": sample_count,
    }


def run_training(
    train_pairs: Sequence[Tuple[Path, Path]],
    validation_pairs: Sequence[Tuple[Path, Path]],
    device: torch.device,
    use_amp: bool,
) -> None:
    if TRAIN_CROP_SIZE % MODEL_DOWNSAMPLE_FACTOR != 0:
        raise ValueError(
            f"TRAIN_CROP_SIZE={TRAIN_CROP_SIZE} must be divisible by "
            f"MODEL_DOWNSAMPLE_FACTOR={MODEL_DOWNSAMPLE_FACTOR}."
        )
    if (
        VALIDATION_CROP_SIZE is not None
        and VALIDATION_CROP_SIZE != TRAIN_CROP_SIZE
    ):
        raise ValueError(
            "For directly comparable one-step train/validation losses, set "
            "VALIDATION_CROP_SIZE equal to TRAIN_CROP_SIZE."
        )

    train_dataset = HSIRGBPairDataset(
        pairs=train_pairs,
        hsi_channels=HSI_CHANNELS,
        crop_size=TRAIN_CROP_SIZE,
        patches_per_image=PATCHES_PER_IMAGE,
        training=True,
        normalization=HSI_NORMALIZATION,
        augment=USE_AUGMENTATION,
    )
    validation_dataset = HSIRGBPairDataset(
        pairs=validation_pairs,
        hsi_channels=HSI_CHANNELS,
        crop_size=VALIDATION_CROP_SIZE,
        patches_per_image=1,
        training=False,
        normalization=HSI_NORMALIZATION,
        augment=False,
    )

    train_loader = make_loader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=(len(train_dataset) >= BATCH_SIZE),
        device=device,
    )
    validation_loader = make_loader(
        dataset=validation_dataset,
        batch_size=VALIDATION_BATCH_SIZE,
        shuffle=False,
        drop_last=False,
        device=device,
    )

    model = build_model(device)
    trainable_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]
    if not trainable_parameters:
        raise RuntimeError(
            "The RDBM U-Net has no trainable parameters."
        )

    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=NUM_EPOCHS,
        eta_min=MIN_LEARNING_RATE,
    )
    scaler = make_grad_scaler(device, use_amp)

    start_epoch = 1
    best_metric = float("inf")

    if RESUME_CHECKPOINT is not None:
        checkpoint = load_torch_checkpoint(
            RESUME_CHECKPOINT,
            device="cpu",
        )
        resume_state = checkpoint.get(
            "rdbm_state_dict",
            checkpoint.get("bridge_state_dict"),
        )
        if not isinstance(resume_state, dict):
            raise KeyError(
                f"No rdbm_state_dict was found in {RESUME_CHECKPOINT}."
            )
        load_rdbm_state_dict(model, resume_state)
        optimizer.load_state_dict(
            checkpoint["optimizer_state_dict"]
        )
        scheduler.load_state_dict(
            checkpoint["scheduler_state_dict"]
        )
        if "scaler_state_dict" in checkpoint:
            try:
                scaler.load_state_dict(
                    checkpoint["scaler_state_dict"]
                )
            except Exception as error:
                print(
                    "Warning: AMP scaler state could not be restored and "
                    f"will be reinitialized. Reason: {error}"
                )
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        best_metric = float(
            checkpoint.get("best_metric", float("inf"))
        )
        print(
            f"Resumed training from {RESUME_CHECKPOINT} "
            f"at epoch {start_epoch}."
        )

    frozen_parameters = sum(
        parameter.numel()
        for parameter in model.mst.parameters()
    )
    trainable_count = sum(
        parameter.numel()
        for parameter in trainable_parameters
    )
    amp_dtype = get_amp_dtype(device) if use_amp else torch.float32

    print(
        f"\nDevice: {device}\n"
        f"Training autocast: {use_amp} ({amp_dtype})\n"
        "Validation sampling/metrics: float32\n"
        f"Training pairs: {len(train_pairs)}\n"
        f"Validation pairs: {len(validation_pairs)}\n"
        f"Training samples per epoch: {len(train_dataset)}\n"
        f"Frozen MST++ parameters: {frozen_parameters:,}\n"
        f"MST++ stages: {MST_NUM_STAGES} "
        f"({MST_STAGE_PARAMETER_NAME or 'auto keyword'})\n"
        f"Trainable RDBM parameters: {trainable_count:,}\n"
        f"RDBM timesteps: {NUM_TIMESTEPS}\n"
        f"Reverse sampling steps: {SAMPLE_STEPS}\n"
        f"Residual-noise lambda: {RDBM_LAMBDA:.8f}"
    )

    for epoch in range(start_epoch, NUM_EPOCHS + 1):
        print(
            f"\n{'=' * 80}\n"
            f"Epoch {epoch}/{NUM_EPOCHS}\n"
            f"{'=' * 80}"
        )

        training_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            use_amp=use_amp,
        )
        validation_metrics = validate_one_epoch(
            model=model,
            loader=validation_loader,
            device=device,
        )
        scheduler.step()

        print(
            f"Epoch {epoch:03d} | "
            f"LR={optimizer.param_groups[0]['lr']:.2e} | "
            f"train loss={training_metrics['loss']:.6f} | "
            f"val loss={validation_metrics['loss']:.6f}"
        )
        print(
            "Validation one-step random-timestep metrics "
            f"({validation_metrics['evaluated_images']} images) | "
            f"objective loss={validation_metrics['objective_loss']:.6f} | "
            f"x_t MRAE={validation_metrics['mrae']:.6f}"
        )

        current_metric = float(
            validation_metrics[BEST_METRIC_NAME]
        )
        improved = current_metric < best_metric
        if improved:
            best_metric = current_metric

        save_training_checkpoint(
            path=LAST_CHECKPOINT,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            epoch=epoch,
            best_metric=best_metric,
            training_metrics=training_metrics,
            validation_metrics=validation_metrics,
        )

        if improved:
            save_training_checkpoint(
                path=BEST_CHECKPOINT,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                epoch=epoch,
                best_metric=best_metric,
                training_metrics=training_metrics,
                validation_metrics=validation_metrics,
            )
            print(
                f"Saved new best checkpoint: {BEST_CHECKPOINT} | "
                f"{BEST_METRIC_NAME.upper()}={best_metric:.6f}"
            )


# =============================================================================
# Full-resolution five-image visualization
# =============================================================================

def rgb_tensor_to_display(rgb: torch.Tensor) -> np.ndarray:
    array = (
        rgb.detach()
        .float()
        .cpu()
        .numpy()
        .transpose(1, 2, 0)
    )
    return np.clip(array, 0.0, 1.0)


def hsi_triplet_to_display(
    target: torch.Tensor,
    coarse_prediction: torch.Tensor,
    bridge_prediction: torch.Tensor,
    bands: Tuple[int, int, int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    target_np = target.detach().float().cpu().numpy()
    coarse_np = (
        coarse_prediction.detach().float().cpu().numpy()
    )
    bridge_np = (
        bridge_prediction.detach().float().cpu().numpy()
    )

    for band in bands:
        if not 0 <= band < target_np.shape[0]:
            raise ValueError(
                f"Visualization band {band} is outside "
                f"[0, {target_np.shape[0] - 1}]."
            )

    def select(cube: np.ndarray) -> np.ndarray:
        return np.stack(
            [cube[band] for band in bands],
            axis=-1,
        )

    target_rgb = select(target_np)
    coarse_rgb = select(coarse_np)
    bridge_rgb = select(bridge_np)

    # One target-derived scale is shared by all HSI panels.
    minimum = target_rgb.min(
        axis=(0, 1),
        keepdims=True,
    )
    maximum = target_rgb.max(
        axis=(0, 1),
        keepdims=True,
    )
    scale = maximum - minimum + 1e-8

    return (
        np.clip((target_rgb - minimum) / scale, 0.0, 1.0),
        np.clip((coarse_rgb - minimum) / scale, 0.0, 1.0),
        np.clip((bridge_rgb - minimum) / scale, 0.0, 1.0),
    )


def format_visual_metrics(metrics: Dict[str, float]) -> str:
    return (
        f"MRAE {metrics['mrae']:.4f} | "
        f"RMSE {metrics['rmse']:.4f}\n"
        f"SAM {metrics['sam']:.4f} rad | "
        f"PSNR {metrics['psnr']:.2f} dB | "
        f"SSIM {metrics['ssim']:.4f}"
    )


@torch.no_grad()
def run_visualization(
    model: RDBMHSI,
    validation_pairs: Sequence[Tuple[Path, Path]],
    device: torch.device,
) -> Path:
    model.eval()
    if not validation_pairs:
        raise RuntimeError(
            "The validation pair list is empty."
        )

    number_to_select = min(
        NUM_VISUALIZATION_IMAGES,
        len(validation_pairs),
    )
    selected_indices = random.sample(
        range(len(validation_pairs)),
        k=number_to_select,
    )

    dataset = HSIRGBPairDataset(
        pairs=validation_pairs,
        hsi_channels=HSI_CHANNELS,
        crop_size=None,
        patches_per_image=1,
        training=False,
        normalization=HSI_NORMALIZATION,
        augment=False,
        return_paths=True,
    )

    column_titles = (
        "Input RGB",
        "Ground-truth HSI\npseudo-RGB",
        "Frozen MST++ coarse HSI\npseudo-RGB",
        "RDBM refined HSI\npseudo-RGB",
    )
    figure, axes = plt.subplots(
        number_to_select,
        4,
        figsize=(19, 5.2 * number_to_select),
        squeeze=False,
    )

    for row, dataset_index in enumerate(selected_indices):
        hsi, rgb, hsi_path_string, _ = dataset[dataset_index]
        (
            padded_hsi,
            padded_rgb,
            original_height,
            original_width,
        ) = pad_pair_to_multiple(
            hsi=hsi,
            rgb=rgb,
            multiple=MODEL_DOWNSAMPLE_FACTOR,
        )

        rgb_batch = (
            padded_rgb.unsqueeze(0)
            .to(device, dtype=torch.float32)
        )
        coarse_prediction = coarse_estimate(
            model,
            rgb_batch,
        ).float()
        bridge_prediction = model.reconstruct(
            rgb_batch,
            last=True,
        )
        if not isinstance(bridge_prediction, torch.Tensor):
            raise TypeError(
                "Expected the final RDBM reconstruction to be a tensor."
            )
        bridge_prediction = clamp_prediction(
            bridge_prediction.float()
        )

        coarse_prediction = coarse_prediction[
            0,
            :,
            :original_height,
            :original_width,
        ].cpu()
        bridge_prediction = bridge_prediction[
            0,
            :,
            :original_height,
            :original_width,
        ].cpu()
        target = hsi[
            :,
            :original_height,
            :original_width,
        ].float().cpu()

        rgb_display = rgb_tensor_to_display(
            rgb[
                :,
                :original_height,
                :original_width,
            ]
        )
        (
            target_display,
            coarse_display,
            bridge_display,
        ) = hsi_triplet_to_display(
            target=target,
            coarse_prediction=coarse_prediction,
            bridge_prediction=bridge_prediction,
            bands=VISUALIZATION_BANDS,
        )

        coarse_metrics = calculate_single_image_metrics(
            coarse_prediction.unsqueeze(0),
            target.unsqueeze(0),
        )
        bridge_metrics = calculate_single_image_metrics(
            bridge_prediction.unsqueeze(0),
            target.unsqueeze(0),
        )

        panels = (
            rgb_display,
            target_display,
            coarse_display,
            bridge_display,
        )
        for column, panel in enumerate(panels):
            axis = axes[row, column]
            axis.imshow(panel)
            axis.axis("off")
            if row == 0:
                axis.set_title(
                    column_titles[column],
                    fontsize=12,
                    fontweight="bold",
                    pad=12,
                )

        stem = Path(hsi_path_string).stem
        axes[row, 0].text(
            0.5,
            -0.08,
            stem,
            transform=axes[row, 0].transAxes,
            ha="center",
            va="top",
            fontsize=10,
            fontweight="bold",
        )
        axes[row, 2].text(
            0.5,
            -0.08,
            format_visual_metrics(coarse_metrics),
            transform=axes[row, 2].transAxes,
            ha="center",
            va="top",
            fontsize=9,
        )
        axes[row, 3].text(
            0.5,
            -0.08,
            format_visual_metrics(bridge_metrics),
            transform=axes[row, 3].transAxes,
            ha="center",
            va="top",
            fontsize=9,
        )

    figure.suptitle(
        "Random full-resolution validation examples: "
        "MST++ coarse estimate and RDBM refinement",
        fontsize=16,
        fontweight="bold",
        y=0.997,
    )
    figure.tight_layout(
        rect=(0.01, 0.01, 0.99, 0.985),
        h_pad=4.0,
    )

    VISUALIZATION_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )
    figure.savefig(
        VISUALIZATION_FILE,
        dpi=FIGURE_DPI,
        bbox_inches="tight",
    )
    plt.close(figure)
    print(
        f"Saved labelled visualization to: {VISUALIZATION_FILE}"
    )
    return VISUALIZATION_FILE


# =============================================================================
# Full validation-set inference (full reverse RDBM sampling)
# =============================================================================

@torch.no_grad()
def evaluate_full_validation_inference(
    model: RDBMHSI,
    validation_pairs: Sequence[Tuple[Path, Path]],
    device: torch.device,
) -> Dict[str, float]:
    """
    Run full inference (complete RDBM reverse sampling, not the
    one-step random-timestep objective used during training/validation) over
    every image in the validation set, at full resolution.

    For each validation pair this mirrors run_visualization():
      1. load the RGB/HSI pair via the existing dataset class;
      2. pad to a multiple of MODEL_DOWNSAMPLE_FACTOR via pad_pair_to_multiple();
      3. compute the frozen MST++ coarse estimate;
      4. run the full reverse sampler via model.reconstruct(...);
      5. clamp the prediction via the existing clamp_prediction() helper;
      6. crop back to the original resolution;
      7. compute MRAE/RMSE/SAM/PSNR/SSIM via calculate_single_image_metrics().

    Metrics are accumulated and averaged over the full validation dataset,
    then printed as a summary.
    """
    model.eval()
    if not validation_pairs:
        raise RuntimeError(
            "The validation pair list is empty."
        )

    dataset = HSIRGBPairDataset(
        pairs=validation_pairs,
        hsi_channels=HSI_CHANNELS,
        crop_size=None,
        patches_per_image=1,
        training=False,
        normalization=HSI_NORMALIZATION,
        augment=False,
        return_paths=True,
    )

    metric_sums = {
        "mrae": 0.0,
        "rmse": 0.0,
        "sam": 0.0,
        "psnr": 0.0,
        "ssim": 0.0,
    }
    evaluated_images = 0

    print(
        f"\nRunning full-resolution inference over {len(dataset)} "
        "validation images (complete RDBM reverse sampling)..."
    )

    for dataset_index in range(len(dataset)):
        hsi, rgb, hsi_path_string, _ = dataset[dataset_index]

        (
            padded_hsi,
            padded_rgb,
            original_height,
            original_width,
        ) = pad_pair_to_multiple(
            hsi=hsi,
            rgb=rgb,
            multiple=MODEL_DOWNSAMPLE_FACTOR,
        )

        rgb_batch = (
            padded_rgb.unsqueeze(0)
            .to(device, dtype=torch.float32)
        )
        coarse_prediction = coarse_estimate(
            model,
            rgb_batch,
        ).float()
        bridge_prediction = model.reconstruct(
            rgb_batch,
            last=True,
        )
        if not isinstance(bridge_prediction, torch.Tensor):
            raise TypeError(
                "Expected the final RDBM reconstruction to be a tensor."
            )
        bridge_prediction = clamp_prediction(
            bridge_prediction.float()
        )

        bridge_prediction = bridge_prediction[
            0,
            :,
            :original_height,
            :original_width,
        ].cpu()
        target = hsi[
            :,
            :original_height,
            :original_width,
        ].float().cpu()

        image_metrics = calculate_single_image_metrics(
            bridge_prediction.unsqueeze(0),
            target.unsqueeze(0),
        )

        for key in metric_sums:
            metric_sums[key] += image_metrics[key]
        evaluated_images += 1

        stem = Path(hsi_path_string).stem
        print(
            f"  [{evaluated_images:04d}/{len(dataset):04d}] {stem} | "
            f"MRAE={image_metrics['mrae']:.6f} | "
            f"RMSE={image_metrics['rmse']:.6f} | "
            f"SAM={image_metrics['sam']:.4f} | "
            f"PSNR={image_metrics['psnr']:.4f} | "
            f"SSIM={image_metrics['ssim']:.6f}"
        )

    if evaluated_images == 0:
        raise RuntimeError(
            "No validation images were evaluated during full inference."
        )

    average_metrics = {
        key: value / evaluated_images
        for key, value in metric_sums.items()
    }

    print(
        "\nFull validation-set inference summary "
        f"({evaluated_images} images, complete reverse sampling):\n"
        f"  Average MRAE: {average_metrics['mrae']:.6f}\n"
        f"  Average RMSE: {average_metrics['rmse']:.6f}\n"
        f"  Average SAM:  {average_metrics['sam']:.6f} rad\n"
        f"  Average PSNR: {average_metrics['psnr']:.4f} dB\n"
        f"  Average SSIM: {average_metrics['ssim']:.6f}"
    )

    return {
        "evaluated_images": evaluated_images,
        **average_metrics,
    }


# =============================================================================
# Mode parser and main
# =============================================================================

def parse_mode() -> str:
    parser = argparse.ArgumentParser(
        description=(
            "Train or visualize the frozen-MST++ RDBM model."
        )
    )
    parser.add_argument(
        "--mode",
        required=True,
        choices=(
            "train",
            "visualize",
            "train_visualize",
        ),
        help=(
            "train: train only; "
            "visualize: load VISUALIZATION_CHECKPOINT and render five "
            "random validation images; "
            "train_visualize: train and then visualize the best checkpoint."
        ),
    )
    return parser.parse_args().mode


def main() -> None:
    mode = parse_mode()
    set_seed(SEED)
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    use_amp = USE_AMP and device.type == "cuda"

    validation_pairs: List[Tuple[Path, Path]]

    if mode in {"train", "train_visualize"}:
        (
            train_pairs,
            validation_pairs,
        ) = prepare_training_and_validation_pairs()
        run_training(
            train_pairs=train_pairs,
            validation_pairs=validation_pairs,
            device=device,
            use_amp=use_amp,
        )
    else:
        validation_pairs = prepare_validation_pairs()

    if mode in {"visualize", "train_visualize"}:
        checkpoint_path = (
            BEST_CHECKPOINT
            if mode == "train_visualize"
            else VISUALIZATION_CHECKPOINT
        )
        model = load_model_for_visualization(
            checkpoint_path=checkpoint_path,
            device=device,
        )
        run_visualization(
            model=model,
            validation_pairs=validation_pairs,
            device=device,
        )
        evaluate_full_validation_inference(
            model=model,
            validation_pairs=validation_pairs,
            device=device,
        )


if __name__ == "__main__":
    main()
