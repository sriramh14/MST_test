"""Train and visualize MST++ for paired RGB-to-HSI reconstruction.

This version follows the organization of the supplied reference training script:

* separate training and validation folders;
* cached HSI metadata validation;
* aligned random training crops and deterministic validation crops;
* multiple patches per training image;
* reproducible DataLoaders;
* FP32 / FP16 / BF16 automatic mixed precision;
* resumable best and last checkpoints;
* validation MRAE, PSNR, RMSE, SAM, and SSIM reporting;
* random full-resolution validation visualization.

MRAE is the only optimization loss.

Usage
-----
All settings except the execution mode are configured below.

    python train_mstpp_reference_style.py --mode train
    python train_mstpp_reference_style.py --mode visualize
    python train_mstpp_reference_style.py --mode train_visualize
"""

from __future__ import annotations

import argparse
import hashlib
import random
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset

from model.MST_Plus_Plus import MST_Plus_Plus

# Project metric functions. The training objective remains MRAE only.
from loss.mrae import mrae
from loss.psnr import psnr
from loss.rmse import rmse
from loss.sam import sam
from loss.ssim import ssim


# =============================================================================
# Configuration: edit values here; argparse is used only for --mode
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

OUTPUT_DIR = Path("./mstpp_checkpoints")
BEST_CHECKPOINT = OUTPUT_DIR / "best_mstpp.pth"
LAST_CHECKPOINT = OUTPUT_DIR / "last_mstpp.pth"
RESUME_CHECKPOINT: Optional[str] = None

VISUALIZATION_CHECKPOINT = BEST_CHECKPOINT
VISUALIZATION_DIR = Path("./mstpp_visualizations")
VISUALIZATION_FILE = VISUALIZATION_DIR / "random_validation_visualization.png"

# HSI/RGB settings.
HSI_KEY = "cube"
HSI_CHANNELS = 31
HSI_NORMALIZATION = "none"  # "none", "minmax", or "band_minmax"
SUPPORTED_HSI_EXTENSIONS = {".npy", ".npz", ".mat", ".pt", ".pth"}
SUPPORTED_RGB_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".tif",
    ".tiff",
    ".npy",
    ".pt",
    ".pth",
}

# MST++ settings.
# Change this variable to select the number of MST++ stages.
MST_NUM_STAGES = 1
MST_NUM_FEATURES = 31
MST_INPUT_CHANNELS = 3
MST_OUTPUT_CHANNELS = HSI_CHANNELS
MST_SPATIAL_FACTOR = 8

# Dataset and training settings.
TRAIN_CROP_SIZE = 256
VALIDATION_CROP_SIZE: Optional[int] = 256
PATCHES_PER_IMAGE = 2
USE_AUGMENTATION = True

BATCH_SIZE = 2
VALIDATION_BATCH_SIZE = 2
NUM_EPOCHS = 30
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
MIN_LEARNING_RATE = 1e-7
GRADIENT_CLIP_NORM = 1.0
NUM_WORKERS = 4
DROP_LAST_TRAIN_BATCH = True
PRINT_EVERY = 30
SEED = 42

# AMP settings.
USE_AMP = False  # MST++ depthwise pos_emb Conv2d may lack a CUDA AMP engine
PREFER_BFLOAT16 = False
FP16_INITIAL_SCALE = 1024.0
FP16_GROWTH_INTERVAL = 2000

# Metric/loss safety settings.
# This exact definition is used for BOTH optimization and validation MRAE.
# Keeping it internal avoids a silent mismatch with loss/mrae.py.
MRAE_EPSILON = 1e-3

# Metadata validation caches.
TRAIN_PAIR_VALIDATION_CACHE = OUTPUT_DIR / "training_pair_validation_cache.pth"
VALIDATION_PAIR_VALIDATION_CACHE = OUTPUT_DIR / "validation_pair_validation_cache.pth"
FORCE_REVALIDATE_FILES = False

# Visualization settings.
NUM_VISUALIZATION_IMAGES = 5
VISUALIZATION_BANDS = (20, 10, 2)
FIGURE_DPI = 180
CLAMP_VISUALIZATION_OUTPUT: Optional[Tuple[float, float]] = (0.0, 1.0)


# =============================================================================
# Reproducibility and AMP helpers
# =============================================================================


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
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
        and torch.cuda.is_bf16_supported()
    ):
        return torch.bfloat16
    return torch.float16


def autocast_context(device: torch.device, enabled: bool):
    if not enabled:
        return nullcontext()
    return torch.autocast(
        device_type=device.type,
        dtype=get_amp_dtype(device),
        enabled=True,
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
        raise ValueError(f"No numeric 3D array was found in {file_path}.")
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
            candidates.append((preferred_key, np.asarray(h5_file[preferred_key])))

        if not candidates:

            def visitor(name: str, obj: Any) -> None:
                if not isinstance(obj, h5py.Dataset) or obj.ndim != 3:
                    return
                try:
                    if np.issubdtype(obj.dtype, np.number):
                        candidates.append((name, np.asarray(obj)))
                except TypeError:
                    return

            h5_file.visititems(visitor)

    if not candidates:
        raise ValueError(f"No numeric 3D HSI dataset was found in {file_path}.")

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
                raise ValueError(f"No 3D array was found in {file_path}.")
            cube = max(candidates, key=lambda array: array.size)
    elif extension == ".mat":
        try:
            cube = _extract_3d_array_from_mapping(
                sio.loadmat(file_path),
                file_path=file_path,
                preferred_key=HSI_KEY,
            )
        except (NotImplementedError, ValueError, OSError):
            cube = load_mat_v73(file_path, preferred_key=HSI_KEY)
    elif extension in {".pt", ".pth"}:
        try:
            loaded = torch.load(file_path, map_location="cpu", weights_only=False)
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
            raise TypeError(f"Unsupported HSI object in {file_path}: {type(loaded)}")
    else:
        raise ValueError(f"Unsupported HSI extension: {extension}")

    cube = np.asarray(cube, dtype=np.float32).squeeze()
    if cube.ndim != 3:
        raise ValueError(
            f"Expected a 3D HSI cube in {file_path}, found shape {cube.shape}."
        )
    return cube


def convert_hsi_to_chw(
    cube: np.ndarray,
    hsi_channels: int,
    file_path: Path,
) -> np.ndarray:
    if cube.shape[0] == hsi_channels:
        output = cube
    elif cube.shape[-1] == hsi_channels:
        output = np.transpose(cube, (2, 0, 1))
    elif cube.shape[1] == hsi_channels:
        output = np.transpose(cube, (1, 0, 2))
    else:
        raise ValueError(
            f"Could not identify the spectral axis in {file_path}. "
            f"Found shape {cube.shape}; expected {hsi_channels} bands."
        )
    return np.ascontiguousarray(output, dtype=np.float32)


def load_rgb_file(file_path: Path) -> np.ndarray:
    extension = file_path.suffix.lower()

    if extension in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}:
        image = Image.open(file_path).convert("RGB")
        array = np.asarray(image, dtype=np.float32) / 255.0
        return np.ascontiguousarray(np.transpose(array, (2, 0, 1)))

    if extension == ".npy":
        array = np.load(file_path).astype(np.float32)
    elif extension in {".pt", ".pth"}:
        try:
            loaded = torch.load(file_path, map_location="cpu", weights_only=False)
        except TypeError:
            loaded = torch.load(file_path, map_location="cpu")

        if isinstance(loaded, torch.Tensor):
            array = loaded.detach().cpu().float().numpy()
        elif isinstance(loaded, np.ndarray):
            array = loaded.astype(np.float32)
        else:
            raise TypeError(f"Unsupported RGB object in {file_path}: {type(loaded)}")
    else:
        raise ValueError(f"Unsupported RGB extension: {extension}")

    array = np.asarray(array, dtype=np.float32).squeeze()
    if array.ndim == 2:
        array = np.stack([array, array, array], axis=0)
    elif array.ndim == 3 and array.shape[0] == 3:
        pass
    elif array.ndim == 3 and array.shape[-1] == 3:
        array = np.transpose(array, (2, 0, 1))
    else:
        raise ValueError(
            f"Could not convert RGB file {file_path} to CHW. Found {array.shape}."
        )

    if np.nanmax(array) > 1.5:
        array = array / 255.0
    return np.ascontiguousarray(array, dtype=np.float32)


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
# File discovery, pairing, metadata validation, and cache
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


def _index_unique_stems(files: Sequence[Path], kind: str) -> Dict[str, Path]:
    index: Dict[str, Path] = {}
    for path in files:
        if path.stem in index:
            raise RuntimeError(
                f"Duplicate {kind} filename stem '{path.stem}'.\n"
                f"First: {index[path.stem]}\nSecond: {path}"
            )
        index[path.stem] = path
    return index


def pair_hsi_rgb_files(
    hsi_directory: str,
    rgb_directory: str,
) -> List[Tuple[Path, Path]]:
    hsi_files = find_files(hsi_directory, SUPPORTED_HSI_EXTENSIONS, "HSI")
    rgb_files = find_files(rgb_directory, SUPPORTED_RGB_EXTENSIONS, "RGB")

    hsi_by_stem = _index_unique_stems(hsi_files, "HSI")
    rgb_by_stem = _index_unique_stems(rgb_files, "RGB")

    shared_stems = sorted(set(hsi_by_stem) & set(rgb_by_stem))
    missing_rgb = sorted(set(hsi_by_stem) - set(rgb_by_stem))
    missing_hsi = sorted(set(rgb_by_stem) - set(hsi_by_stem))

    if missing_rgb:
        print(f"Warning: {len(missing_rgb)} HSI files have no matching RGB file.")
    if missing_hsi:
        print(f"Warning: {len(missing_hsi)} RGB files have no matching HSI file.")
    if not shared_stems:
        raise RuntimeError(
            "No paired HSI/RGB files were found. Paired files need identical stems."
        )

    pairs = [(hsi_by_stem[stem], rgb_by_stem[stem]) for stem in shared_stems]
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
    return hashlib.sha256("\n".join(records).encode("utf-8")).hexdigest()


def is_possible_hsi_shape(shape: Sequence[int], hsi_channels: int) -> bool:
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
        if hsi_key in h5_file and isinstance(h5_file[hsi_key], h5py.Dataset):
            dataset = h5_file[hsi_key]
            candidates.append((hsi_key, tuple(int(v) for v in dataset.shape)))

        def visitor(name: str, obj: Any) -> None:
            if not isinstance(obj, h5py.Dataset) or obj.ndim != 3:
                return
            try:
                if np.issubdtype(obj.dtype, np.number):
                    record = (name, tuple(int(v) for v in obj.shape))
                    if record not in candidates:
                        candidates.append(record)
            except TypeError:
                return

        h5_file.visititems(visitor)

    if not candidates:
        raise ValueError(f"No numerical 3D dataset was found in {file_path}.")
    if not any(is_possible_hsi_shape(shape, hsi_channels) for _, shape in candidates):
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
        inspect_hdf5_mat_file(file_path, hsi_channels, hsi_key)
        return

    candidates = [
        (name, tuple(int(value) for value in shape))
        for name, shape, _ in metadata
        if len(shape) == 3
    ]
    if not candidates:
        raise ValueError(f"No 3D array was found in {file_path}.")

    preferred = [candidate for candidate in candidates if candidate[0] == hsi_key]
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
    extension = file_path.suffix.lower()

    if extension == ".mat":
        inspect_standard_mat_file(file_path, hsi_channels, hsi_key)
        return

    if extension == ".npy":
        cube = np.load(file_path, mmap_mode="r", allow_pickle=False)
        if not is_possible_hsi_shape(cube.shape, hsi_channels):
            raise ValueError(f"Invalid HSI shape {cube.shape} in {file_path}.")
        return

    if extension == ".npz":
        with np.load(file_path, allow_pickle=False) as archive:
            shapes = [
                archive[key].shape
                for key in archive.files
                if archive[key].ndim == 3
            ]
        if not any(is_possible_hsi_shape(shape, hsi_channels) for shape in shapes):
            raise ValueError(
                f"No valid {hsi_channels}-band cube in {file_path}; shapes={shapes}."
            )
        return

    cube = load_hsi_file(file_path)
    if not is_possible_hsi_shape(cube.shape, hsi_channels):
        raise ValueError(f"Invalid HSI shape {cube.shape} in {file_path}.")


def load_torch_checkpoint(
    path: str | Path,
    device: str | torch.device = "cpu",
):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def filter_valid_pairs(
    pairs: Sequence[Tuple[Path, Path]],
    hsi_channels: int,
    log_path: Path,
    cache_path: Path,
) -> List[Tuple[Path, Path]]:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    pairs = list(pairs)
    fingerprint = make_files_fingerprint(
        [path for pair in pairs for path in pair]
    )
    pair_lookup = {
        str(hsi_path.resolve()): (hsi_path, rgb_path)
        for hsi_path, rgb_path in pairs
    }

    if cache_path.exists() and not FORCE_REVALIDATE_FILES:
        try:
            cached = load_torch_checkpoint(cache_path, device="cpu")
            if isinstance(cached, dict) and cached.get("fingerprint") == fingerprint:
                valid_pairs = [
                    pair_lookup[path]
                    for path in cached.get("valid_hsi_paths", [])
                    if path in pair_lookup
                ]
                invalid_records = cached.get("invalid_records", [])
                print(f"\nUsing cached pair validation: {cache_path}")
                print(
                    f"Valid pairs: {len(valid_pairs)} | "
                    f"Invalid: {len(invalid_records)}"
                )
                if valid_pairs:
                    return valid_pairs
        except Exception as error:
            print(
                "\nCould not use the validation cache; rescanning. "
                f"Reason: {error}"
            )

    print("\nChecking HSI file metadata before use...")
    valid_pairs: List[Tuple[Path, Path]] = []
    invalid_records: List[dict] = []

    for index, (hsi_path, rgb_path) in enumerate(pairs, start=1):
        try:
            # Keep the pre-scan cheap, like the reference script: validate HSI
            # readability and spectral shape from metadata where possible. Full
            # RGB/HSI loading and spatial matching are checked in the Dataset.
            inspect_hsi_file_metadata(hsi_path, hsi_channels, HSI_KEY)
            valid_pairs.append((hsi_path, rgb_path))
        except Exception as error:
            invalid_records.append(
                {
                    "path": str(hsi_path.resolve()),
                    "error": f"{type(error).__name__}: {error}",
                }
            )
            print(
                "\nSkipping invalid pair:\n"
                f"  HSI: {hsi_path}\n"
                f"  RGB: {rgb_path}\n"
                f"  Error: {error}"
            )

        if index % 100 == 0 or index == len(pairs):
            print(
                f"Checked {index}/{len(pairs)} | "
                f"Valid: {len(valid_pairs)} | Invalid: {len(invalid_records)}"
            )

    if not valid_pairs:
        raise RuntimeError("No valid HSI/RGB pairs remain after validation.")

    if invalid_records:
        with log_path.open("w", encoding="utf-8") as log_file:
            for record in invalid_records:
                log_file.write(f"{record['path']} | {record['error']}\n")
        print(f"Invalid-pair log saved to: {log_path}")

    torch.save(
        {
            "fingerprint": fingerprint,
            "valid_hsi_paths": [
                str(hsi_path.resolve()) for hsi_path, _ in valid_pairs
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


def _safe_pad_chw(
    tensor: torch.Tensor,
    pad_left: int,
    pad_right: int,
    pad_top: int,
    pad_bottom: int,
) -> torch.Tensor:
    if not any((pad_left, pad_right, pad_top, pad_bottom)):
        return tensor

    height, width = tensor.shape[-2:]
    # Reflection padding requires each pad amount to be smaller than the source
    # dimension. Replication works for very small images as well.
    use_reflect = (
        height > max(pad_top, pad_bottom)
        and width > max(pad_left, pad_right)
    )
    mode = "reflect" if use_reflect else "replicate"
    return F.pad(
        tensor,
        (pad_left, pad_right, pad_top, pad_bottom),
        mode=mode,
    )


def _pad_tensor_to_minimum_size(
    tensor: torch.Tensor,
    minimum_height: int,
    minimum_width: int,
) -> torch.Tensor:
    _, height, width = tensor.shape
    pad_height = max(0, minimum_height - height)
    pad_width = max(0, minimum_width - width)
    return _safe_pad_chw(tensor, 0, pad_width, 0, pad_height)


def random_crop_pair(
    hsi: torch.Tensor,
    rgb: torch.Tensor,
    crop_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    hsi = _pad_tensor_to_minimum_size(hsi, crop_size, crop_size)
    rgb = _pad_tensor_to_minimum_size(rgb, crop_size, crop_size)
    _, height, width = hsi.shape
    top = random.randint(0, height - crop_size)
    left = random.randint(0, width - crop_size)
    return (
        hsi[:, top : top + crop_size, left : left + crop_size],
        rgb[:, top : top + crop_size, left : left + crop_size],
    )


def center_crop_pair(
    hsi: torch.Tensor,
    rgb: torch.Tensor,
    crop_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    hsi = _pad_tensor_to_minimum_size(hsi, crop_size, crop_size)
    rgb = _pad_tensor_to_minimum_size(rgb, crop_size, crop_size)
    _, height, width = hsi.shape
    top = (height - crop_size) // 2
    left = (width - crop_size) // 2
    return (
        hsi[:, top : top + crop_size, left : left + crop_size],
        rgb[:, top : top + crop_size, left : left + crop_size],
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

    hsi = _safe_pad_chw(hsi, 0, pad_width, 0, pad_height)
    rgb = _safe_pad_chw(rgb, 0, pad_width, 0, pad_height)
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
            raise ValueError("patches_per_image must be at least 1.")

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
        hsi_array = normalize_hsi_cube(hsi_array, mode=self.normalization)
        rgb_array = load_rgb_file(rgb_path)

        if hsi_array.shape[1:] != rgb_array.shape[1:]:
            raise ValueError(
                f"Spatial mismatch for {hsi_path.stem}: "
                f"HSI={hsi_array.shape[1:]}, RGB={rgb_array.shape[1:]}."
            )
        if not np.isfinite(hsi_array).all():
            raise ValueError(f"HSI contains NaN/Inf: {hsi_path}")
        if not np.isfinite(rgb_array).all():
            raise ValueError(f"RGB contains NaN/Inf: {rgb_path}")

        hsi = torch.from_numpy(hsi_array.copy()).float()
        rgb = torch.from_numpy(rgb_array.copy()).float()
        return hsi, rgb, hsi_path, rgb_path

    def __getitem__(self, index: int):
        pair_index = index // self.patches_per_image if self.training else index
        hsi, rgb, hsi_path, rgb_path = self._load_pair(pair_index)

        if self.crop_size is not None:
            if self.training:
                hsi, rgb = random_crop_pair(hsi, rgb, self.crop_size)
            else:
                hsi, rgb = center_crop_pair(hsi, rgb, self.crop_size)

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
# Model, loss, metrics, and checkpoint helpers
# =============================================================================


def unwrap_prediction(output: Any) -> torch.Tensor:
    """Use the final reconstruction if MST++ returns intermediate outputs."""
    if isinstance(output, (list, tuple)):
        if not output:
            raise ValueError("MST++ returned an empty output sequence.")
        output = output[-1]
    if isinstance(output, dict):
        for key in ("prediction", "reconstruction", "output", "out"):
            value = output.get(key)
            if torch.is_tensor(value):
                output = value
                break
    if not torch.is_tensor(output):
        raise TypeError(f"Expected MST++ output tensor, found {type(output)}")
    return output


def build_model(device: torch.device) -> MST_Plus_Plus:
    model = MST_Plus_Plus(
        in_channels=MST_INPUT_CHANNELS,
        out_channels=MST_OUTPUT_CHANNELS,
        n_feat=MST_NUM_FEATURES,
        stage=MST_NUM_STAGES,
    )
    return model.to(device)


def model_config_dict() -> dict:
    return {
        "in_channels": MST_INPUT_CHANNELS,
        "out_channels": MST_OUTPUT_CHANNELS,
        "n_feat": MST_NUM_FEATURES,
        "stage": MST_NUM_STAGES,
        "spatial_factor": MST_SPATIAL_FACTOR,
    }


def mrae_per_sample(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """Return one stable MRAE value per BCHW image.

    MRAE = mean(|prediction - target| / max(|target|, epsilon)).
    The calculation is always performed in float32.
    """
    if prediction.shape != target.shape:
        raise ValueError(
            f"MRAE shape mismatch: prediction={tuple(prediction.shape)}, "
            f"target={tuple(target.shape)}"
        )
    if prediction.ndim != 4:
        raise ValueError(
            f"MRAE expects BCHW tensors, found {prediction.ndim} dimensions."
        )

    prediction = prediction.float()
    target = target.float()
    denominator = target.abs().clamp_min(MRAE_EPSILON)
    relative_error = (prediction - target).abs() / denominator
    return relative_error.mean(dim=(1, 2, 3))


def stable_mrae(
    target: torch.Tensor,
    prediction: torch.Tensor,
) -> torch.Tensor:
    """Backward-compatible scalar MRAE wrapper."""
    return mrae_per_sample(prediction=prediction, target=target).mean()

def _scalar_metric(value: Any, name: str) -> torch.Tensor:
    if not torch.is_tensor(value):
        value = torch.as_tensor(value)
    value = value.float()
    if value.numel() != 1:
        value = value.mean()
    if not torch.isfinite(value):
        # A perfect reconstruction can legitimately produce +inf PSNR.
        if not (name == "PSNR" and torch.isposinf(value)):
            raise FloatingPointError(
                f"{name} returned a non-finite value: {value}"
            )
    return value


def calculate_mrae_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """Compute the exact MRAE training objective in float32."""
    with torch.autocast(
        device_type=prediction.device.type,
        enabled=False,
    ):
        loss = mrae_per_sample(
            prediction=prediction.float(),
            target=target.float(),
        ).mean()

    if not torch.isfinite(loss):
        raise FloatingPointError(f"MRAE returned a non-finite value: {loss}")
    if prediction.requires_grad and not loss.requires_grad:
        raise RuntimeError(
            "MRAE loss is detached from the prediction graph and cannot train the model."
        )
    return loss


@torch.no_grad()
def calculate_validation_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> Dict[str, float]:
    """Return sums of per-image metrics for a batch.

    Metrics are evaluated one image at a time so the epoch average is not
    affected by the reduction behavior inside the project metric functions.
    """
    prediction = prediction.detach().float()
    target = target.detach().float()

    if prediction.shape != target.shape:
        raise ValueError(
            f"Metric shape mismatch: prediction={tuple(prediction.shape)}, "
            f"target={tuple(target.shape)}"
        )
    if not torch.isfinite(prediction).all():
        raise FloatingPointError("Prediction contains NaN or Inf.")
    if not torch.isfinite(target).all():
        raise FloatingPointError("Target contains NaN or Inf.")

    metric_sums = {
        "mrae": 0.0,
        "psnr": 0.0,
        "rmse": 0.0,
        "sam": 0.0,
        "ssim": 0.0,
    }

    for sample_index in range(prediction.shape[0]):
        sample_prediction = prediction[sample_index : sample_index + 1]
        sample_target = target[sample_index : sample_index + 1]

        mrae_value = mrae_per_sample(
            prediction=sample_prediction,
            target=sample_target,
        ).mean()

        metric_sums["mrae"] += float(mrae_value.detach().item())
        metric_sums["psnr"] += float(
            _scalar_metric(psnr(sample_target, sample_prediction), "PSNR")
            .detach()
            .item()
        )
        metric_sums["rmse"] += float(
            _scalar_metric(rmse(sample_target, sample_prediction), "RMSE")
            .detach()
            .item()
        )
        metric_sums["sam"] += float(
            _scalar_metric(sam(sample_target, sample_prediction), "SAM")
            .detach()
            .item()
        )
        metric_sums["ssim"] += float(
            _scalar_metric(ssim(sample_target, sample_prediction), "SSIM")
            .detach()
            .item()
        )

    return metric_sums


def strip_prefix_if_present(
    state_dict: Dict[str, torch.Tensor],
    prefix: str,
) -> Dict[str, torch.Tensor]:
    if state_dict and all(key.startswith(prefix) for key in state_dict):
        return {key[len(prefix) :]: value for key, value in state_dict.items()}
    return state_dict


def extract_state_dict(
    checkpoint: object,
    candidate_keys: Sequence[str],
) -> Dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict):
        for key in candidate_keys:
            value = checkpoint.get(key)
            if (
                isinstance(value, dict)
                and value
                and all(torch.is_tensor(tensor) for tensor in value.values())
            ):
                return value
        if checkpoint and all(torch.is_tensor(value) for value in checkpoint.values()):
            return checkpoint
    raise KeyError(f"Could not find a state_dict using keys: {tuple(candidate_keys)}")


def normalize_mst_state_dict(
    state_dict: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    state_dict = strip_prefix_if_present(state_dict, "module.")
    state_dict = strip_prefix_if_present(state_dict, "model.")
    state_dict = strip_prefix_if_present(state_dict, "mst_model.")
    return state_dict


def save_training_checkpoint(
    path: Path,
    model: MST_Plus_Plus,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler,
    epoch: int,
    best_validation_mrae: float,
    training_metrics: dict,
    validation_metrics: dict,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "best_validation_mrae": best_validation_mrae,
            "training_metrics": training_metrics,
            "validation_metrics": validation_metrics,
            "model_config": model_config_dict(),
            "data_config": {
                "hsi_channels": HSI_CHANNELS,
                "hsi_key": HSI_KEY,
                "normalization": HSI_NORMALIZATION,
                "train_crop_size": TRAIN_CROP_SIZE,
                "validation_crop_size": VALIDATION_CROP_SIZE,
            },
        },
        path,
    )


def validate_checkpoint_architecture(checkpoint: dict) -> None:
    config = checkpoint.get("model_config", {})
    checkpoint_stage = config.get("stage")
    if checkpoint_stage is not None and int(checkpoint_stage) != MST_NUM_STAGES:
        raise ValueError(
            "Checkpoint/model stage mismatch: "
            f"checkpoint stage={checkpoint_stage}, configured "
            f"MST_NUM_STAGES={MST_NUM_STAGES}."
        )


def load_model_for_visualization(
    checkpoint_path: str | Path,
    device: torch.device,
) -> MST_Plus_Plus:
    checkpoint = load_torch_checkpoint(checkpoint_path, device="cpu")
    if isinstance(checkpoint, dict):
        validate_checkpoint_architecture(checkpoint)

    model = build_model(device)
    state_dict = extract_state_dict(
        checkpoint,
        candidate_keys=("model_state_dict", "state_dict", "model", "params"),
    )
    state_dict = normalize_mst_state_dict(state_dict)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    print(f"Loaded MST++ checkpoint: {checkpoint_path}")
    return model


# =============================================================================
# Pair preparation
# =============================================================================


def prepare_training_and_validation_pairs() -> Tuple[
    List[Tuple[Path, Path]],
    List[Tuple[Path, Path]],
]:
    train_pairs = pair_hsi_rgb_files(TRAIN_HSI_DIR, TRAIN_RGB_DIR)
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
# Training and validation
# =============================================================================


def train_one_epoch(
    model: MST_Plus_Plus,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    use_amp: bool,
) -> dict:
    model.train()

    loss_sum = 0.0
    sample_count = 0

    for batch_index, (hsi, rgb) in enumerate(loader, start=1):
        hsi = hsi.to(device, non_blocking=True)
        rgb = rgb.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        with autocast_context(device, use_amp):
            prediction = unwrap_prediction(model(rgb))

        loss = calculate_mrae_loss(prediction=prediction, target=hsi)
        if not torch.isfinite(loss):
            raise FloatingPointError(
                f"Non-finite training MRAE at batch {batch_index}: "
                f"{float(loss.detach())}"
            )

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)

        gradient_norm = nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=GRADIENT_CLIP_NORM,
            error_if_nonfinite=True,
        )

        scaler.step(optimizer)
        scaler.update()

        with torch.no_grad():
            batch_mrae_values = mrae_per_sample(
                prediction=prediction.detach(),
                target=hsi,
            )

        batch_size = hsi.shape[0]
        loss_sum += float(batch_mrae_values.sum().item())
        sample_count += batch_size

        if batch_index % PRINT_EVERY == 0 or batch_index == len(loader):
            print(
                f"  Batch {batch_index:04d}/{len(loader):04d} | "
                f"MRAE={loss_sum / sample_count:.6f} | "
                f"grad={float(gradient_norm):.4f}"
            )

    if sample_count == 0:
        raise RuntimeError("The training DataLoader produced no samples.")

    return {
        "loss": loss_sum / sample_count,
        "mrae": loss_sum / sample_count,
        "evaluated_images": sample_count,
    }


@torch.no_grad()
def validate_one_epoch(
    model: MST_Plus_Plus,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool,
) -> dict:
    model.eval()

    metric_sums = {
        "mrae": 0.0,
        "psnr": 0.0,
        "rmse": 0.0,
        "sam": 0.0,
        "ssim": 0.0,
    }
    sample_count = 0

    for hsi, rgb in loader:
        hsi = hsi.to(device, non_blocking=True)
        rgb = rgb.to(device, non_blocking=True)

        with autocast_context(device, use_amp):
            prediction = unwrap_prediction(model(rgb))

        batch_metrics = calculate_validation_metrics(
            prediction=prediction,
            target=hsi,
        )
        for key in metric_sums:
            metric_sums[key] += batch_metrics[key]
        sample_count += hsi.shape[0]

    if sample_count == 0:
        raise RuntimeError("The validation DataLoader produced no samples.")

    averaged = {
        key: value / sample_count for key, value in metric_sums.items()
    }
    averaged["loss"] = averaged["mrae"]
    averaged["evaluated_images"] = sample_count
    return averaged


def run_training(
    train_pairs: Sequence[Tuple[Path, Path]],
    validation_pairs: Sequence[Tuple[Path, Path]],
    device: torch.device,
    use_amp: bool,
) -> None:
    if TRAIN_CROP_SIZE % MST_SPATIAL_FACTOR != 0:
        raise ValueError(
            f"TRAIN_CROP_SIZE={TRAIN_CROP_SIZE} must be divisible by "
            f"MST_SPATIAL_FACTOR={MST_SPATIAL_FACTOR}."
        )
    if (
        VALIDATION_CROP_SIZE is not None
        and VALIDATION_CROP_SIZE % MST_SPATIAL_FACTOR != 0
    ):
        raise ValueError(
            f"VALIDATION_CROP_SIZE={VALIDATION_CROP_SIZE} must be divisible by "
            f"MST_SPATIAL_FACTOR={MST_SPATIAL_FACTOR}."
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
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=(DROP_LAST_TRAIN_BATCH and len(train_dataset) >= BATCH_SIZE),
        device=device,
    )
    validation_loader = make_loader(
        validation_dataset,
        batch_size=VALIDATION_BATCH_SIZE,
        shuffle=False,
        drop_last=False,
        device=device,
    )

    model = build_model(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=NUM_EPOCHS,
        eta_min=MIN_LEARNING_RATE,
    )

    amp_dtype = get_amp_dtype(device)
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=(use_amp and amp_dtype == torch.float16),
        init_scale=FP16_INITIAL_SCALE,
        growth_interval=FP16_GROWTH_INTERVAL,
    )

    start_epoch = 1
    best_validation_mrae = float("inf")

    if RESUME_CHECKPOINT is not None:
        checkpoint = load_torch_checkpoint(RESUME_CHECKPOINT, device="cpu")
        if not isinstance(checkpoint, dict):
            raise TypeError("A resumable checkpoint must be a dictionary.")
        validate_checkpoint_architecture(checkpoint)

        state_dict = extract_state_dict(
            checkpoint,
            candidate_keys=("model_state_dict", "state_dict", "model", "params"),
        )
        state_dict = normalize_mst_state_dict(state_dict)
        model.load_state_dict(state_dict, strict=True)

        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if "scaler_state_dict" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler_state_dict"])

        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        best_validation_mrae = float(
            checkpoint.get(
                "best_validation_mrae",
                checkpoint.get("best_mrae", float("inf")),
            )
        )
        print(
            f"Resumed training from {RESUME_CHECKPOINT} "
            f"at epoch {start_epoch}."
        )

    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(
        f"\nDevice: {device}\n"
        f"AMP: {use_amp} ({amp_dtype if use_amp else 'float32'})\n"
        f"Training pairs: {len(train_pairs)}\n"
        f"Validation pairs: {len(validation_pairs)}\n"
        f"Training samples/epoch: {len(train_dataset)}\n"
        f"Validation samples/epoch: {len(validation_dataset)}\n"
        f"MST++ stages: {MST_NUM_STAGES}\n"
        f"MST++ features: {MST_NUM_FEATURES}\n"
        f"Trainable parameters: {parameter_count:,}\n"
        f"Optimization loss: MRAE only"
    )

    for epoch in range(start_epoch, NUM_EPOCHS + 1):
        print(f"\n{'=' * 80}\nEpoch {epoch}/{NUM_EPOCHS}\n{'=' * 80}")

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
            use_amp=use_amp,
        )
        scheduler.step()

        print(
            f"Epoch {epoch:03d} | "
            f"LR={optimizer.param_groups[0]['lr']:.2e} | "
            f"train MRAE={training_metrics['mrae']:.6f} | "
            f"validation MRAE={validation_metrics['mrae']:.6f}"
        )
        print(
            "Validation reconstruction metrics "
            f"({validation_metrics['evaluated_images']} images) | "
            f"MRAE={validation_metrics['mrae']:.6f} | "
            f"PSNR={validation_metrics['psnr']:.4f} | "
            f"RMSE={validation_metrics['rmse']:.6f} | "
            f"SAM={validation_metrics['sam']:.6f} | "
            f"SSIM={validation_metrics['ssim']:.4f}"
        )

        improved = validation_metrics["mrae"] < best_validation_mrae
        if improved:
            best_validation_mrae = validation_metrics["mrae"]

        save_training_checkpoint(
            path=LAST_CHECKPOINT,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            epoch=epoch,
            best_validation_mrae=best_validation_mrae,
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
                best_validation_mrae=best_validation_mrae,
                training_metrics=training_metrics,
                validation_metrics=validation_metrics,
            )
            print(
                f"Saved new best checkpoint: {BEST_CHECKPOINT} | "
                f"validation MRAE={best_validation_mrae:.6f}"
            )


# =============================================================================
# Full-resolution visualization
# =============================================================================


def rgb_tensor_to_display(rgb: torch.Tensor) -> np.ndarray:
    array = rgb.detach().float().cpu().numpy().transpose(1, 2, 0)
    return np.clip(array, 0.0, 1.0)


def hsi_pair_to_display(
    target: torch.Tensor,
    prediction: torch.Tensor,
    bands: Tuple[int, int, int],
) -> Tuple[np.ndarray, np.ndarray]:
    target_np = target.detach().float().cpu().numpy()
    prediction_np = prediction.detach().float().cpu().numpy()

    for band in bands:
        if not 0 <= band < target_np.shape[0]:
            raise ValueError(
                f"Visualization band {band} is outside "
                f"[0, {target_np.shape[0] - 1}]."
            )

    target_rgb = np.stack([target_np[band] for band in bands], axis=-1)
    prediction_rgb = np.stack(
        [prediction_np[band] for band in bands],
        axis=-1,
    )

    # Use target-derived channel scaling for both panels.
    minimum = target_rgb.min(axis=(0, 1), keepdims=True)
    maximum = target_rgb.max(axis=(0, 1), keepdims=True)
    scale = maximum - minimum + 1e-8

    return (
        np.clip((target_rgb - minimum) / scale, 0.0, 1.0),
        np.clip((prediction_rgb - minimum) / scale, 0.0, 1.0),
    )


def residual_to_display(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> np.ndarray:
    residual = (prediction.float() - target.float()).abs().mean(dim=0)
    return residual.detach().cpu().numpy()


@torch.no_grad()
def run_visualization(
    model: MST_Plus_Plus,
    validation_pairs: Sequence[Tuple[Path, Path]],
    device: torch.device,
    use_amp: bool,
) -> Path:
    model.eval()
    if not validation_pairs:
        raise RuntimeError("The validation pair list is empty.")

    number_to_select = min(NUM_VISUALIZATION_IMAGES, len(validation_pairs))
    selected_indices = random.Random(SEED).sample(
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
        "Ground-truth HSI\n(pseudo-RGB)",
        "MST++ prediction\n(pseudo-RGB)",
        "Mean absolute\nspectral residual",
    )
    figure, axes = plt.subplots(
        number_to_select,
        4,
        figsize=(16, 4.0 * number_to_select),
        squeeze=False,
    )

    for row, dataset_index in enumerate(selected_indices):
        hsi, rgb, hsi_path_string, _ = dataset[dataset_index]
        padded_hsi, padded_rgb, original_height, original_width = pad_pair_to_multiple(
            hsi=hsi,
            rgb=rgb,
            multiple=MST_SPATIAL_FACTOR,
        )

        rgb_batch = padded_rgb.unsqueeze(0).to(device)
        with autocast_context(device, use_amp):
            prediction = unwrap_prediction(model(rgb_batch))

        prediction = prediction[
            0, :, :original_height, :original_width
        ].float().cpu()
        target = padded_hsi[
            :, :original_height, :original_width
        ].float().cpu()

        if CLAMP_VISUALIZATION_OUTPUT is not None:
            prediction = prediction.clamp(*CLAMP_VISUALIZATION_OUTPUT)

        metrics = calculate_validation_metrics(
            prediction=prediction.unsqueeze(0),
            target=target.unsqueeze(0),
        )

        rgb_display = rgb_tensor_to_display(
            rgb[:, :original_height, :original_width]
        )
        target_display, prediction_display = hsi_pair_to_display(
            target=target,
            prediction=prediction,
            bands=VISUALIZATION_BANDS,
        )
        residual_display = residual_to_display(prediction, target)

        panels = (
            rgb_display,
            target_display,
            prediction_display,
            residual_display,
        )
        for column, panel in enumerate(panels):
            axis = axes[row, column]
            if column == 3:
                image = axis.imshow(panel, cmap="inferno")
                figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
            else:
                axis.imshow(panel)
            axis.axis("off")
            if row == 0:
                axis.set_title(
                    column_titles[column],
                    fontsize=12,
                    fontweight="bold",
                )

        stem = Path(hsi_path_string).stem
        axes[row, 0].set_ylabel(
            f"{stem}\n"
            f"MRAE {metrics['mrae']:.4f}\n"
            f"PSNR {metrics['psnr']:.2f} dB\n"
            f"SAM {metrics['sam']:.4f}",
            fontsize=9,
            rotation=0,
            labelpad=80,
            va="center",
        )

    figure.suptitle(
        f"Random full-resolution MST++ validation examples "
        f"({MST_NUM_STAGES} stages)",
        fontsize=16,
        fontweight="bold",
        y=0.995,
    )
    figure.tight_layout(rect=(0.07, 0.01, 1.0, 0.985))

    VISUALIZATION_DIR.mkdir(parents=True, exist_ok=True)
    figure.savefig(VISUALIZATION_FILE, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved labelled visualization to: {VISUALIZATION_FILE}")
    return VISUALIZATION_FILE


# =============================================================================
# Mode parser and main
# =============================================================================


def parse_mode() -> str:
    parser = argparse.ArgumentParser(
        description="Train or visualize an MST++ RGB-to-HSI model."
    )
    parser.add_argument(
        "--mode",
        required=True,
        choices=("train", "visualize", "train_visualize"),
        help=(
            "train: train only; visualize: load VISUALIZATION_CHECKPOINT and "
            "render random full-resolution validation images; train_visualize: "
            "train and then visualize the best checkpoint."
        ),
    )
    return parser.parse_args().mode


def main() -> None:
    mode = parse_mode()
    set_seed(SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = USE_AMP and device.type == "cuda"

    if mode in {"train", "train_visualize"}:
        train_pairs, validation_pairs = prepare_training_and_validation_pairs()
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
        model = load_model_for_visualization(checkpoint_path, device)
        run_visualization(
            model=model,
            validation_pairs=validation_pairs,
            device=device,
            use_amp=use_amp,
        )


if __name__ == "__main__":
    main()
