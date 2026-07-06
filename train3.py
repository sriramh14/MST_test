"""Train the MST++-conditioned residual Brownian-bridge diffusion model.

A pretrained MST++ model produces a frozen coarse HSI estimate. The trainable
Brownian-bridge denoiser predicts the correction residual

    ground_truth_hsi - coarse_hsi

and reconstructs

    coarse_hsi + predicted_residual.

The dataset, validation, metric, checkpoint, AMP, and visualization structure is
kept aligned with the supplied training script. Visualization remains enabled
only through the --visualize command-line flag.
"""

from __future__ import annotations

import argparse
import hashlib
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import h5py
import numpy as np
import scipy.io as sio
import torch
import torch.nn.functional as F
from PIL import Image
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader, Dataset


# ============================================================================
# Project imports
# ============================================================================
# Adjust only these import paths if your project layout is different.
# Adjust these paths/classes to match your project layout.
from model.MST_Plus_Plus import MST_Plus_Plus
from model.RBBDM_rgb2hsi import (
    MSTResidualDenoiser,
    ResidualBBDM,
    MSTPlusPlusResidualBBDM,
)

from loss.mrae import mrae
from loss.psnr import psnr
from loss.rmse import rmse
from loss.sam import sam
from loss.ssim import ssim


# ============================================================================
# Configuration
# ============================================================================

# "train", "infer", or "train_and_infer"
RUN_MODE = "train"

# Training data.
TRAIN_HSI_DIR = (
    "/kaggle/input/datasets/sriramhari14/ntire-2022/"
    "Train_spectral/Train_spectral"
)
TRAIN_RGB_DIR = (
    "/kaggle/input/datasets/sriramhari14/ntire-2022/"
    "Train_RGB/Train_RGB"
)

# Validation data is intentionally in a separate pair of folders.
VALIDATION_HSI_DIR = (
    "/kaggle/input/datasets/sriramhari14/ntire-2022/Valid_spectral/Valid_spectral"
)
VALIDATION_RGB_DIR = (
    "/kaggle/input/datasets/sriramhari14/ntire-2022/Valid_RGB/Valid_RGB"
)

# Pretrained frozen MST++ checkpoint and residual-diffusion outputs.
MST_CHECKPOINT = "./mst_checkpoints/mst_plus_plus.pth"
OUTPUT_DIR = "./mst_bbdm_checkpoints"

# Add constructor arguments required by your MST++ implementation here.
MST_MODEL_KWARGS: Dict[str, Any] = {}
STRICT_MST_CHECKPOINT = True

# Dataset-validation caches. They are reused while the HSI file paths,
# sizes, and modification times remain unchanged.
TRAIN_PAIR_VALIDATION_CACHE = (
    Path(OUTPUT_DIR) / "training_pair_validation_cache.pth"
)
VALIDATION_PAIR_VALIDATION_CACHE = (
    Path(OUTPUT_DIR) / "validation_pair_validation_cache.pth"
)

# Used by RUN_MODE="infer". The best checkpoint is normally selected here.
INFERENCE_CHECKPOINT = (
    "./mst_bbdm_checkpoints/best_bbdm.pth"
)
RESUME_CHECKPOINT: Optional[str] = None

# When recovering from a run that overflowed, retain model/optimizer moments but
# reset the AMP scale and force the safer learning rate configured below.
LOAD_SCALER_STATE_ON_RESUME = False
OVERRIDE_RESUMED_LEARNING_RATE = True

# Number of randomly selected full-resolution validation images in inference mode.
NUM_RANDOM_INFERENCE_IMAGES = 5
INFERENCE_OUTPUT_DIR = "./mst_bbdm_inference"

# HSI/RGB file layout.
HSI_KEY = "cube"
SUPPORTED_HSI_EXTENSIONS = {".npy", ".npz", ".mat", ".pt", ".pth"}
SUPPORTED_RGB_EXTENSIONS = {".png", ".jpg", ".jpeg", ".npy", ".pt", ".pth"}

# This must match the normalization used to train MST++.
# "none", "minmax", or "band_minmax".
HSI_NORMALIZATION = "none"

# Range used only when CLAMP_PREDICTION_FOR_METRICS=True.
# The imported PSNR/SSIM functions retain their own definitions.
METRIC_DATA_RANGE = 1.0
CLAMP_PREDICTION_FOR_METRICS = False

# Reverse-sampling options used for validation and inference.
INFERENCE_CLIP_DENOISED = True
INFERENCE_STOCHASTIC = False

# Residual Brownian-bridge architecture.
HSI_CHANNELS = 31
RGB_CHANNELS = 3
RESIDUAL_N_FEAT = 31
RESIDUAL_BODY_DEPTH = 3
MST_DENOISER_STAGE = 2
MST_DENOISER_NUM_BLOCKS = (1, 1, 1)
NUM_DIFFUSION_STEPS = 50
MIDPOINT_VARIANCE = 0.05

# The denoiser pads internally to 2 ** MST_DENOISER_STAGE.
MODEL_DOWNSAMPLE_FACTOR = 2 ** MST_DENOISER_STAGE

# Training crops.
TRAIN_CROP_SIZE = 256
PATCHES_PER_IMAGE = 2

# Training.
BATCH_SIZE = 2
VALIDATION_BATCH_SIZE = 2
NUM_EPOCHS = 45
LEARNING_RATE = 5e-5
WEIGHT_DECAY = 1e-4
MIN_LEARNING_RATE = 1e-7
GRADIENT_CLIP_NORM = 1.0

NUM_WORKERS = 4
USE_AMP = True

# Prefer BF16 when the GPU supports it. BF16 has a much wider exponent range
# than FP16 and is substantially less likely to overflow during backprop.
PREFER_BFLOAT16 = True

# Used only when FP16 GradScaler is active.
FP16_INITIAL_SCALE = 1024.0
FP16_GROWTH_INTERVAL = 2000

# A single overflow should not terminate a long run. The affected optimizer
# step is skipped and the FP16 scale is reduced. Persistent overflows still
# raise an error so real divergence is not hidden.
MAX_CONSECUTIVE_NONFINITE_GRADIENTS = 10

USE_AUGMENTATION = True
SEED = 42
PRINT_EVERY = 30

# Main objective from the residual-prediction model.
# The residual network predicts the clean correction residual directly.
RESIDUAL_L1_WEIGHT = 1.0

# Optional direct L1 loss on the refined HSI reconstruction.
RECONSTRUCTION_L1_WEIGHT = 0.0

# Actual RGB -> HSI reconstruction metrics require iterative diffusion sampling.
# None evaluates every validation image. A positive integer limits cost.
VALIDATION_METRIC_MAX_IMAGES: Optional[int] = 20

# Training metrics use the single-step clean-residual estimate at the sampled t.
COMPUTE_TRAIN_ONE_STEP_METRICS = True
TRAIN_METRIC_EVERY = 1

# Pseudo-RGB HSI bands used only for saved inference previews.
# Change these indices for your wavelength ordering.
VISUALIZATION_BANDS = (20, 10, 2)

# ============================================================================
# Reproducibility
# ============================================================================

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


# ============================================================================
# AMP helpers
# ============================================================================

def get_amp_dtype(
    device: torch.device,
) -> torch.dtype:
    if (
        device.type == "cuda"
        and PREFER_BFLOAT16
        and torch.cuda.is_bf16_supported()
    ):
        return torch.bfloat16

    return torch.float16


def autocast_context(
    device: torch.device,
    enabled: bool,
):
    return torch.autocast(
        device_type=device.type,
        dtype=get_amp_dtype(device),
        enabled=enabled,
    )


# ============================================================================
# HSI/RGB loading
# ============================================================================

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
            array = np.asarray(h5_file[preferred_key])
            candidates.append((preferred_key, array))

        if not candidates:
            def visitor(name, obj):
                if not isinstance(obj, h5py.Dataset):
                    return
                if obj.ndim != 3:
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
    # convert_to_chw() below performs the final spectral-axis identification.
    cube = np.transpose(
        cube,
        axes=tuple(range(cube.ndim - 1, -1, -1)),
    )
    return cube


def load_hsi_file(file_path: Path) -> np.ndarray:
    extension = file_path.suffix.lower()

    if extension == ".npy":
        cube = np.load(file_path)

    elif extension == ".npz":
        loaded = np.load(file_path)
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
            loaded = torch.load(
                file_path,
                map_location="cpu",
            )

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
        raise ValueError(
            f"Unsupported HSI extension: {extension}"
        )

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
        return np.ascontiguousarray(
            np.transpose(cube, (2, 0, 1))
        )

    if cube.shape[1] == hsi_channels:
        return np.ascontiguousarray(
            np.transpose(cube, (1, 0, 2))
        )

    raise ValueError(
        f"Could not identify the spectral axis in {file_path}. "
        f"Found shape {cube.shape}; expected {hsi_channels} bands."
    )


def load_rgb_file(file_path: Path) -> np.ndarray:
    extension = file_path.suffix.lower()

    if extension in {".png", ".jpg", ".jpeg"}:
        image = Image.open(file_path).convert("RGB")
        array = np.asarray(image, dtype=np.float32) / 255.0
        return np.ascontiguousarray(
            np.transpose(array, (2, 0, 1))
        )

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
            loaded = torch.load(
                file_path,
                map_location="cpu",
            )

        if isinstance(loaded, torch.Tensor):
            array = loaded.detach().cpu().float().numpy()
        elif isinstance(loaded, np.ndarray):
            array = loaded.astype(np.float32)
        else:
            raise TypeError(
                f"Unsupported RGB object in {file_path}: {type(loaded)}"
            )

    else:
        raise ValueError(
            f"Unsupported RGB extension: {extension}"
        )

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

    # Normalize common uint-like NPY/PT representations.
    if np.nanmax(array) > 1.5:
        array = array / 255.0

    return np.ascontiguousarray(array)


def normalize_hsi_cube(
    cube: np.ndarray,
    mode: str,
) -> np.ndarray:
    if mode == "none":
        return cube

    if mode == "minmax":
        minimum = float(cube.min())
        maximum = float(cube.max())
        return (
            (cube - minimum)
            / (maximum - minimum + 1e-8)
        )

    if mode == "band_minmax":
        minimum = cube.min(
            axis=(1, 2),
            keepdims=True,
        )
        maximum = cube.max(
            axis=(1, 2),
            keepdims=True,
        )
        return (
            (cube - minimum)
            / (maximum - minimum + 1e-8)
        )

    raise ValueError(
        f"Unknown HSI normalization mode: {mode}"
    )


# ============================================================================
# File discovery, pairing, and validation
# ============================================================================

def find_files(
    directory: str,
    extensions: Sequence[str],
    kind: str,
) -> List[Path]:
    root = Path(directory)

    if not root.exists():
        raise FileNotFoundError(
            f"{kind} directory does not exist: {root}"
        )

    files = sorted(
        path
        for path in root.rglob("*")
        if path.is_file()
        and path.suffix.lower() in extensions
    )

    if not files:
        raise RuntimeError(
            f"No supported {kind} files were found in {root}."
        )

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

    hsi_by_stem = _index_unique_stems(
        hsi_files,
        "HSI",
    )
    rgb_by_stem = _index_unique_stems(
        rgb_files,
        "RGB",
    )

    shared_stems = sorted(
        set(hsi_by_stem) & set(rgb_by_stem)
    )

    missing_rgb = sorted(
        set(hsi_by_stem) - set(rgb_by_stem)
    )
    missing_hsi = sorted(
        set(rgb_by_stem) - set(hsi_by_stem)
    )

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
            "The paired files must have identical filename stems."
        )

    pairs = [
        (
            hsi_by_stem[stem],
            rgb_by_stem[stem],
        )
        for stem in shared_stems
    ]

    print(
        f"Found {len(pairs)} paired files in:\n"
        f"  HSI: {hsi_directory}\n"
        f"  RGB: {rgb_directory}"
    )
    return pairs


def make_files_fingerprint(
    files: Sequence[Path],
) -> str:
    """
    Create a cache fingerprint from file path, size, and modification time.

    The cache is invalidated automatically if an HSI file is added, removed,
    replaced, or modified.
    """
    records = []

    for file_path in files:
        stat = file_path.stat()
        records.append(
            f"{file_path.resolve()}|"
            f"{stat.st_size}|"
            f"{stat.st_mtime_ns}"
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
    """
    Inspect a MATLAB v7.3/HDF5 file without loading its full HSI cube.
    """
    candidates: List[
        Tuple[str, Tuple[int, ...]]
    ] = []

    with h5py.File(
        str(file_path),
        "r",
    ) as h5_file:
        if (
            hsi_key in h5_file
            and isinstance(
                h5_file[hsi_key],
                h5py.Dataset,
            )
        ):
            dataset = h5_file[hsi_key]
            candidates.append(
                (
                    hsi_key,
                    tuple(
                        int(value)
                        for value in dataset.shape
                    ),
                )
            )
        else:
            def visitor(name, obj):
                if (
                    not isinstance(obj, h5py.Dataset)
                    or obj.ndim != 3
                ):
                    return

                try:
                    if np.issubdtype(
                        obj.dtype,
                        np.number,
                    ):
                        candidates.append(
                            (
                                name,
                                tuple(
                                    int(value)
                                    for value in obj.shape
                                ),
                            )
                        )
                except TypeError:
                    return

            h5_file.visititems(visitor)

    if not candidates:
        raise ValueError(
            f"No numerical three-dimensional dataset "
            f"was found in {file_path}."
        )

    if not any(
        is_possible_hsi_shape(
            shape,
            hsi_channels,
        )
        for _, shape in candidates
    ):
        raise ValueError(
            f"No {hsi_channels}-band cube was found in "
            f"{file_path}. HDF5 datasets: {candidates}"
        )


def inspect_standard_mat_file(
    file_path: Path,
    hsi_channels: int,
    hsi_key: str,
) -> None:
    """
    Inspect a standard MATLAB file using scipy.io.whosmat(), which reads
    array metadata rather than loading every array.
    """
    try:
        metadata = sio.whosmat(file_path)
    except (
        NotImplementedError,
        ValueError,
        OSError,
    ):
        # MATLAB v7.3 files require HDF5 inspection.
        inspect_hdf5_mat_file(
            file_path=file_path,
            hsi_channels=hsi_channels,
            hsi_key=hsi_key,
        )
        return

    candidates = [
        (
            name,
            tuple(
                int(value)
                for value in shape
            ),
        )
        for name, shape, _ in metadata
        if len(shape) == 3
    ]

    if not candidates:
        raise ValueError(
            f"No three-dimensional array was found in "
            f"{file_path}."
        )

    preferred = [
        candidate
        for candidate in candidates
        if candidate[0] == hsi_key
    ]
    arrays_to_check = (
        preferred
        if preferred
        else candidates
    )

    if not any(
        is_possible_hsi_shape(
            shape,
            hsi_channels,
        )
        for _, shape in arrays_to_check
    ):
        raise ValueError(
            f"No {hsi_channels}-band cube was found in "
            f"{file_path}. MATLAB arrays: {candidates}"
        )


def inspect_hsi_file_metadata(
    file_path: Path,
    hsi_channels: int,
    hsi_key: str,
) -> None:
    """
    Validate an HSI file in the same way as the earlier training script.

    .mat:
        Inspect metadata only with whosmat() or an HDF5 header.

    .npy/.npz/.pt/.pth:
        Reuse the normal loader because these formats are comparatively
        inexpensive to inspect.
    """
    if file_path.suffix.lower() == ".mat":
        inspect_standard_mat_file(
            file_path=file_path,
            hsi_channels=hsi_channels,
            hsi_key=hsi_key,
        )
        return

    cube = load_hsi_file(file_path)

    if not is_possible_hsi_shape(
        cube.shape,
        hsi_channels,
    ):
        raise ValueError(
            f"Invalid HSI shape {cube.shape} in "
            f"{file_path}."
        )


def filter_valid_pairs(
    pairs: Sequence[Tuple[Path, Path]],
    hsi_channels: int,
    log_path: Path,
    cache_path: Path,
) -> List[Tuple[Path, Path]]:
    """
    Metadata-first HSI validation with a persistent cache.

    This mirrors the checking approach in the earlier script:
      1. Pair files by filename stem.
      2. Check MATLAB files through metadata rather than loading full cubes.
      3. Cache valid/invalid results using a file fingerprint.
      4. Skip invalid HSI files and write their errors to a log.

    RGB files have already been checked for a supported extension during
    pairing. Their full pixel data is loaded only by the Dataset.
    """
    cache_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    log_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    pairs = list(pairs)
    hsi_files = [
        hsi_path
        for hsi_path, _ in pairs
    ]

    fingerprint = make_files_fingerprint(
        hsi_files
    )

    pair_lookup = {
        str(hsi_path.resolve()): (
            hsi_path,
            rgb_path,
        )
        for hsi_path, rgb_path in pairs
    }

    # ------------------------------------------------------------------
    # Reuse an unchanged validation cache.
    # ------------------------------------------------------------------
    if cache_path.exists():
        try:
            try:
                cached = torch.load(
                    cache_path,
                    map_location="cpu",
                    weights_only=False,
                )
            except TypeError:
                cached = torch.load(
                    cache_path,
                    map_location="cpu",
                )

            if (
                isinstance(cached, dict)
                and cached.get("fingerprint")
                == fingerprint
            ):
                valid_paths = cached.get(
                    "valid_hsi_paths",
                    [],
                )
                invalid_records = cached.get(
                    "invalid_records",
                    [],
                )

                valid_pairs = [
                    pair_lookup[path]
                    for path in valid_paths
                    if path in pair_lookup
                ]

                print(
                    f"\nUsing cached pair validation: "
                    f"{cache_path}"
                )
                print(
                    f"Valid pairs:   {len(valid_pairs)}"
                )
                print(
                    f"Invalid files: "
                    f"{len(invalid_records)}"
                )

                for record in invalid_records:
                    print(
                        "\nCached invalid file:\n"
                        f"  File:  {record['path']}\n"
                        f"  Error: {record['error']}"
                    )

                if valid_pairs:
                    return valid_pairs

        except Exception as error:
            print(
                "\nCould not use the validation cache. "
                "The dataset will be checked again.\n"
                f"Reason: {error}"
            )

    # ------------------------------------------------------------------
    # Perform a fresh metadata-first scan.
    # ------------------------------------------------------------------
    print(
        "\nChecking HSI file metadata before use..."
    )

    valid_pairs: List[
        Tuple[Path, Path]
    ] = []
    invalid_records: List[dict] = []

    for index, (
        hsi_path,
        rgb_path,
    ) in enumerate(
        pairs,
        start=1,
    ):
        try:
            inspect_hsi_file_metadata(
                file_path=hsi_path,
                hsi_channels=hsi_channels,
                hsi_key=HSI_KEY,
            )
            valid_pairs.append(
                (hsi_path, rgb_path)
            )

        except Exception as error:
            invalid_records.append(
                {
                    "path": str(
                        hsi_path.resolve()
                    ),
                    "error": (
                        f"{type(error).__name__}: "
                        f"{error}"
                    ),
                }
            )

            print(
                "\nSkipping invalid HSI file:\n"
                f"  File:  {hsi_path}\n"
                f"  Error: {error}"
            )

        if (
            index % 100 == 0
            or index == len(pairs)
        ):
            print(
                f"Checked {index}/{len(pairs)} | "
                f"Valid: {len(valid_pairs)} | "
                f"Invalid: {len(invalid_records)}"
            )

    if not valid_pairs:
        raise RuntimeError(
            "No valid HSI/RGB pairs remain after "
            "metadata validation."
        )

    if invalid_records:
        with log_path.open(
            "w",
            encoding="utf-8",
        ) as log_file:
            for record in invalid_records:
                log_file.write(
                    f"{record['path']} | "
                    f"{record['error']}\n"
                )

        print(
            f"\nInvalid-file log saved to: "
            f"{log_path}"
        )

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

    print(
        f"Validation cache saved to: "
        f"{cache_path}"
    )

    return valid_pairs


# ============================================================================
# Paired spatial transforms
# ============================================================================

def _pad_tensor_to_minimum_size(
    tensor: torch.Tensor,
    minimum_height: int,
    minimum_width: int,
) -> torch.Tensor:
    _, height, width = tensor.shape

    pad_height = max(
        0,
        minimum_height - height,
    )
    pad_width = max(
        0,
        minimum_width - width,
    )

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

    top = random.randint(
        0,
        height - crop_size,
    )
    left = random.randint(
        0,
        width - crop_size,
    )

    return (
        hsi[
            :,
            top:top + crop_size,
            left:left + crop_size,
        ],
        rgb[
            :,
            top:top + crop_size,
            left:left + crop_size,
        ],
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
        hsi[
            :,
            top:top + crop_size,
            left:left + crop_size,
        ],
        rgb[
            :,
            top:top + crop_size,
            left:left + crop_size,
        ],
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
        hsi = torch.rot90(
            hsi,
            k=rotations,
            dims=(1, 2),
        )
        rgb = torch.rot90(
            rgb,
            k=rotations,
            dims=(1, 2),
        )

    return hsi.contiguous(), rgb.contiguous()


def pad_pair_to_multiple(
    hsi: torch.Tensor,
    rgb: torch.Tensor,
    multiple: int,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    int,
    int,
]:
    _, original_height, original_width = hsi.shape

    pad_height = (
        multiple - original_height % multiple
    ) % multiple
    pad_width = (
        multiple - original_width % multiple
    ) % multiple

    if pad_height == 0 and pad_width == 0:
        return (
            hsi,
            rgb,
            original_height,
            original_width,
        )

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

    return (
        hsi,
        rgb,
        original_height,
        original_width,
    )


# ============================================================================
# Dataset
# ============================================================================

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
    ):
        self.pairs = list(pairs)
        self.hsi_channels = hsi_channels
        self.crop_size = crop_size
        self.patches_per_image = patches_per_image
        self.training = training
        self.normalization = normalization
        self.augment = augment
        self.return_paths = return_paths

        if training and crop_size is None:
            raise ValueError(
                "Training requires a finite crop_size."
            )

    def __len__(self) -> int:
        multiplier = (
            self.patches_per_image
            if self.training
            else 1
        )
        return len(self.pairs) * multiplier

    def _load_pair(
        self,
        pair_index: int,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        Path,
        Path,
    ]:
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
            raise ValueError(
                f"HSI contains NaN/Inf: {hsi_path}"
            )
        if not np.isfinite(rgb_array).all():
            raise ValueError(
                f"RGB contains NaN/Inf: {rgb_path}"
            )

        hsi = torch.from_numpy(
            hsi_array.copy()
        ).float()
        rgb = torch.from_numpy(
            rgb_array.copy()
        ).float()

        return hsi, rgb, hsi_path, rgb_path

    def __getitem__(self, index: int):
        if self.training:
            pair_index = (
                index // self.patches_per_image
            )
        else:
            pair_index = index

        hsi, rgb, hsi_path, rgb_path = (
            self._load_pair(pair_index)
        )

        if self.crop_size is not None:
            if self.training:
                hsi, rgb = random_crop_pair(
                    hsi,
                    rgb,
                    crop_size=self.crop_size,
                )
            else:
                hsi, rgb = center_crop_pair(
                    hsi,
                    rgb,
                    crop_size=self.crop_size,
                )

        if self.training and self.augment:
            hsi, rgb = augment_pair(
                hsi,
                rgb,
            )

        if self.return_paths:
            return (
                hsi,
                rgb,
                str(hsi_path),
                str(rgb_path),
            )

        return hsi, rgb



# ============================================================================
# Residual-diffusion construction and checkpoint handling
# ============================================================================

def _load_torch_checkpoint(
    path: str | Path,
    device: torch.device | str,
):
    try:
        return torch.load(
            path,
            map_location=device,
            weights_only=False,
        )
    except TypeError:
        return torch.load(
            path,
            map_location=device,
        )


def _strip_module_prefix(
    state_dict: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    if not state_dict:
        return state_dict

    if all(
        key.startswith("module.")
        for key in state_dict
    ):
        return {
            key[len("module."):]: value
            for key, value in state_dict.items()
        }

    return state_dict


def _extract_state_dict(
    checkpoint,
    candidate_keys: Sequence[str],
) -> Dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict):
        for key in candidate_keys:
            state = checkpoint.get(key)
            if isinstance(state, dict):
                return _strip_module_prefix(state)

        if checkpoint and all(
            torch.is_tensor(value)
            for value in checkpoint.values()
        ):
            return _strip_module_prefix(checkpoint)

    raise KeyError(
        "Could not locate a residual-diffusion state_dict in the checkpoint."
    )


def default_model_config() -> dict:
    return {
        "hsi_channels": HSI_CHANNELS,
        "rgb_channels": RGB_CHANNELS,
        "n_feat": RESIDUAL_N_FEAT,
        "body_depth": RESIDUAL_BODY_DEPTH,
        "mst_stage": MST_DENOISER_STAGE,
        "num_blocks": tuple(MST_DENOISER_NUM_BLOCKS),
        "num_timesteps": NUM_DIFFUSION_STEPS,
        "midpoint_variance": MIDPOINT_VARIANCE,
    }


def _load_mst_weights(
    coarse_model: torch.nn.Module,
) -> None:
    checkpoint = _load_torch_checkpoint(
        MST_CHECKPOINT,
        device="cpu",
    )

    state_dict = _extract_state_dict(
        checkpoint,
        candidate_keys=(
            "model_state_dict",
            "mst_state_dict",
            "state_dict",
            "model",
        ),
    )

    coarse_model.load_state_dict(
        state_dict,
        strict=STRICT_MST_CHECKPOINT,
    )


def build_residual_diffusion(
    device: torch.device,
    model_config: Optional[dict] = None,
) -> MSTPlusPlusResidualBBDM:
    config = default_model_config()

    if model_config is not None:
        for key in config:
            if key in model_config:
                config[key] = model_config[key]

    coarse_model = MST_Plus_Plus(
        **MST_MODEL_KWARGS
    )
    _load_mst_weights(coarse_model)
    coarse_model.float()

    denoiser = MSTResidualDenoiser(
        hsi_channels=config["hsi_channels"],
        rgb_channels=config["rgb_channels"],
        n_feat=config["n_feat"],
        body_depth=config["body_depth"],
        mst_stage=config["mst_stage"],
        num_blocks=tuple(config["num_blocks"]),
    )

    bridge = ResidualBBDM(
        denoiser=denoiser,
        num_timesteps=config["num_timesteps"],
        midpoint_variance=config["midpoint_variance"],
    )

    model = MSTPlusPlusResidualBBDM(
        coarse_model=coarse_model,
        bridge=bridge,
        freeze_coarse_model=True,
    )

    return model.to(device)


def save_checkpoint(
    output_path: Path,
    model: MSTPlusPlusResidualBBDM,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: GradScaler,
    epoch: int,
    best_validation_residual_loss: float,
    validation_metrics: dict,
) -> None:
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "best_validation_residual_loss": (
                best_validation_residual_loss
            ),
            "validation_metrics": validation_metrics,
            "model_config": default_model_config(),
            "mst_checkpoint": MST_CHECKPOINT,
        },
        output_path,
    )


def load_residual_diffusion_weights(
    model: MSTPlusPlusResidualBBDM,
    checkpoint,
) -> None:
    state_dict = _extract_state_dict(
        checkpoint,
        candidate_keys=(
            "model_state_dict",
            "residual_bbdm_state_dict",
            "state_dict",
        ),
    )

    model.load_state_dict(
        state_dict,
        strict=True,
    )


def _extract_tensor_output(output) -> torch.Tensor:
    """Extract a tensor from common MST++ return formats."""
    if torch.is_tensor(output):
        return output

    if isinstance(output, (tuple, list)):
        for value in output:
            if torch.is_tensor(value):
                return value

    if isinstance(output, dict):
        for key in (
            "prediction",
            "output",
            "out",
            "hsi",
            "reconstruction",
        ):
            value = output.get(key)
            if torch.is_tensor(value):
                return value

        for value in output.values():
            if torch.is_tensor(value):
                return value

    raise TypeError(
        "MST++ must return a tensor, a tensor-containing sequence, "
        "or a tensor-containing dictionary."
    )


@torch.no_grad()
def get_coarse_prediction_fp32(
    model: MSTPlusPlusResidualBBDM,
    rgb: torch.Tensor,
) -> torch.Tensor:
    """Run frozen MST++ in FP32 outside the surrounding AMP context."""
    model.coarse_model.eval()

    with torch.autocast(
        device_type=rgb.device.type,
        enabled=False,
    ):
        output = model.coarse_model(
            rgb.detach().float().contiguous()
        )
        coarse_hsi = _extract_tensor_output(output)

    return coarse_hsi.detach().float().contiguous()


# ============================================================================
# Residual diffusion
# ============================================================================

def sample_training_timesteps(
    batch_size: int,
    num_steps: int,
    device: torch.device,
) -> torch.Tensor:
    return torch.randint(
        1,
        num_steps + 1,
        (batch_size,),
        device=device,
        dtype=torch.long,
    )


def calculate_training_losses(
    outputs: Dict[str, torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    residual_l1_loss = F.l1_loss(
        outputs["predicted_residual"].float(),
        outputs["target_residual"].float(),
    )

    reconstruction_l1_loss = F.l1_loss(
        outputs["reconstruction"].float(),
        (
            outputs["coarse_hsi"]
            + outputs["target_residual"]
        ).float(),
    )

    total_loss = (
        RESIDUAL_L1_WEIGHT
        * residual_l1_loss
        + RECONSTRUCTION_L1_WEIGHT
        * reconstruction_l1_loss
    )

    return (
        total_loss,
        residual_l1_loss,
        reconstruction_l1_loss,
    )

# ============================================================================
# HSI metric aggregation using the project's existing metric functions
# ============================================================================

def _metric_output_to_scalar(
    value,
    metric_name: str,
) -> float:
    if not torch.is_tensor(value):
        value = torch.as_tensor(value)

    value = value.detach().float()

    # The imported functions are expected to return a scalar. Taking the mean
    # also supports implementations that return a one-element/per-image tensor.
    if value.numel() != 1:
        value = value.mean()

    scalar = float(value.item())

    if not math_is_finite_or_positive_infinity(scalar):
        raise FloatingPointError(
            f"{metric_name} returned a non-finite value: {scalar}"
        )

    return scalar


def math_is_finite_or_positive_infinity(value: float) -> bool:
    # Positive infinity is valid for PSNR when prediction exactly equals target.
    return bool(
        np.isfinite(value)
        or value == float("inf")
    )


@torch.no_grad()
def calculate_project_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> dict:
    """
    Calculate metrics with the functions imported from the project's loss folder.

    The argument order matches the user's existing code:
        metric(target, reconstruction)
    """
    prediction = prediction.detach().float()
    target = target.detach().float()

    if prediction.shape != target.shape:
        raise ValueError(
            f"Metric shape mismatch: prediction={prediction.shape}, "
            f"target={target.shape}"
        )

    if not torch.isfinite(prediction).all():
        raise FloatingPointError(
            "Prediction contains NaN or Inf during metric calculation."
        )
    if not torch.isfinite(target).all():
        raise FloatingPointError(
            "Target contains NaN or Inf during metric calculation."
        )

    return {
        "mrae": _metric_output_to_scalar(
            mrae(target, prediction),
            "MRAE",
        ),
        "rmse": _metric_output_to_scalar(
            rmse(target, prediction),
            "RMSE",
        ),
        "sam": _metric_output_to_scalar(
            sam(target, prediction),
            "SAM",
        ),
        "psnr": _metric_output_to_scalar(
            psnr(target, prediction),
            "PSNR",
        ),
        "ssim": _metric_output_to_scalar(
            ssim(target, prediction),
            "SSIM",
        ),
    }


@dataclass
class HSIMetricAccumulator:
    """
    Average the project's existing metrics equally over validation images.

    Metrics are called separately for each image. This avoids depending on the
    internal batch reduction used by each imported function and correctly
    handles a smaller final batch.
    """

    data_range: float = 1.0
    clamp_prediction: bool = False

    mrae_sum: float = 0.0
    rmse_sum: float = 0.0
    sam_sum: float = 0.0
    psnr_sum: float = 0.0
    ssim_sum: float = 0.0
    image_count: int = 0

    @torch.no_grad()
    def update(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
    ) -> None:
        prediction = prediction.detach().float()
        target = target.detach().float()

        if prediction.shape != target.shape:
            raise ValueError(
                f"Metric shape mismatch: prediction={prediction.shape}, "
                f"target={target.shape}"
            )

        if self.clamp_prediction:
            prediction = prediction.clamp(
                0.0,
                self.data_range,
            )

        for sample_index in range(prediction.shape[0]):
            sample_metrics = calculate_project_metrics(
                prediction=prediction[
                    sample_index:sample_index + 1
                ],
                target=target[
                    sample_index:sample_index + 1
                ],
            )

            self.mrae_sum += sample_metrics["mrae"]
            self.rmse_sum += sample_metrics["rmse"]
            self.sam_sum += sample_metrics["sam"]
            self.psnr_sum += sample_metrics["psnr"]
            self.ssim_sum += sample_metrics["ssim"]
            self.image_count += 1

    def compute(self) -> dict:
        if self.image_count == 0:
            return {
                "mrae": float("nan"),
                "rmse": float("nan"),
                "sam": float("nan"),
                "psnr": float("nan"),
                "ssim": float("nan"),
            }

        denominator = float(self.image_count)

        return {
            "mrae": self.mrae_sum / denominator,
            "rmse": self.rmse_sum / denominator,
            "sam": self.sam_sum / denominator,
            "psnr": self.psnr_sum / denominator,
            "ssim": self.ssim_sum / denominator,
        }



# ============================================================================
# Discrete diffusion-timestep loss tracking
# ============================================================================

def create_timestep_tracker(
    num_steps: int,
) -> dict:
    return {
        step: {
            "loss_sum": 0.0,
            "count": 0,
        }
        for step in range(1, num_steps + 1)
    }


@torch.no_grad()
def update_timestep_tracker(
    tracker: dict,
    timestep: torch.Tensor,
    predicted_residual: torch.Tensor,
    target_residual: torch.Tensor,
) -> None:
    per_sample_loss = F.l1_loss(
        predicted_residual.detach().float(),
        target_residual.detach().float(),
        reduction="none",
    ).mean(dim=(1, 2, 3))

    for step in tracker:
        mask = timestep == step

        if not mask.any():
            continue

        tracker[step]["loss_sum"] += (
            per_sample_loss[mask].sum().item()
        )
        tracker[step]["count"] += int(
            mask.sum().item()
        )


def finalize_timestep_tracker(
    tracker: dict,
) -> dict:
    result = {}

    for step, values in tracker.items():
        count = values["count"]
        result[str(step)] = {
            "residual_l1": (
                values["loss_sum"] / count
                if count > 0
                else float("nan")
            ),
            "count": count,
        }

    return result


def print_timestep_tracker(
    title: str,
    result: dict,
) -> None:
    print(f"\n{title}")

    for step, values in result.items():
        count = values["count"]
        loss = values["residual_l1"]

        if count == 0:
            print(
                f"  t={step} | no samples"
            )
        else:
            print(
                f"  t={step} | "
                f"residual L1={loss:.6f} | "
                f"samples={count}"
            )


# ============================================================================
# Training and validation
# ============================================================================

def train_one_epoch(
    model: MSTPlusPlusResidualBBDM,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
    use_amp: bool,
) -> dict:
    model.train()

    trainable_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]

    total_loss_sum = 0.0
    residual_l1_sum = 0.0
    reconstruction_l1_sum = 0.0
    total_samples = 0

    metric_accumulator = HSIMetricAccumulator(
        data_range=METRIC_DATA_RANGE,
        clamp_prediction=(
            CLAMP_PREDICTION_FOR_METRICS
        ),
    )

    metric_batches = 0
    timestep_tracker = create_timestep_tracker(
        model.bridge.num_timesteps
    )

    skipped_nonfinite_batches = 0
    consecutive_nonfinite_batches = 0

    for batch_index, (hsi, rgb) in enumerate(
        loader,
        start=1,
    ):
        hsi = hsi.to(
            device,
            non_blocking=True,
        )
        rgb = rgb.to(
            device,
            non_blocking=True,
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        timestep = sample_training_timesteps(
            batch_size=hsi.shape[0],
            num_steps=model.bridge.num_timesteps,
            device=device,
        )

        # Keep the frozen MST++ branch in FP32. Only the trainable bridge
        # denoiser is evaluated under AMP.
        coarse_hsi = get_coarse_prediction_fp32(
            model=model,
            rgb=rgb,
        )

        with autocast_context(
            device=device,
            enabled=use_amp,
        ):
            outputs = model.bridge.training_predictions(
                rgb=rgb,
                coarse_hsi=coarse_hsi,
                ground_truth=hsi,
                t=timestep,
            )
            outputs["coarse_hsi"] = coarse_hsi

            (
                loss,
                residual_l1_loss,
                reconstruction_l1_loss,
            ) = calculate_training_losses(
                outputs
            )

        if not torch.isfinite(loss):
            raise FloatingPointError(
                f"Non-finite training loss at batch "
                f"{batch_index}: {loss.item()}"
            )

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)

        gradient_norm = torch.nn.utils.clip_grad_norm_(
            trainable_parameters,
            max_norm=GRADIENT_CLIP_NORM,
            error_if_nonfinite=False,
        )

        if not torch.isfinite(gradient_norm):
            skipped_nonfinite_batches += 1
            consecutive_nonfinite_batches += 1

            optimizer.zero_grad(
                set_to_none=True
            )

            old_scale = (
                float(scaler.get_scale())
                if scaler.is_enabled()
                else 1.0
            )

            if scaler.is_enabled():
                new_scale = max(
                    old_scale * 0.5,
                    1.0,
                )
                scaler.update(
                    new_scale=new_scale
                )
            else:
                new_scale = old_scale

            print(
                "\n  Warning: skipped batch "
                f"{batch_index} because the gradient norm was "
                f"{float(gradient_norm)}. "
                f"loss={float(loss.detach()):.6g}, "
                f"t=[{int(timestep.min())}, "
                f"{int(timestep.max())}], "
                f"|target residual|max="
                f"{float(outputs['target_residual'].detach().abs().max()):.4g}, "
                f"|x_t|max="
                f"{float(outputs['x_t'].detach().abs().max()):.4g}, "
                f"|predicted residual|max="
                f"{float(outputs['predicted_residual'].detach().abs().max()):.4g}, "
                f"AMP scale {old_scale:.1f} -> {new_scale:.1f}."
            )

            if (
                consecutive_nonfinite_batches
                >= MAX_CONSECUTIVE_NONFINITE_GRADIENTS
            ):
                raise FloatingPointError(
                    "Persistent non-finite gradients: "
                    f"{consecutive_nonfinite_batches} consecutive "
                    "batches failed. Reduce LEARNING_RATE, disable AMP, "
                    "and inspect the printed input/residual magnitudes."
                )

            continue

        consecutive_nonfinite_batches = 0

        scaler.step(optimizer)
        scaler.update()

        update_timestep_tracker(
            tracker=timestep_tracker,
            timestep=outputs["t"].detach(),
            predicted_residual=outputs["predicted_residual"],
            target_residual=outputs["target_residual"],
        )

        batch_size = hsi.shape[0]
        total_loss_sum += (
            loss.detach().item()
            * batch_size
        )
        residual_l1_sum += (
            residual_l1_loss.detach().item()
            * batch_size
        )
        reconstruction_l1_sum += (
            reconstruction_l1_loss.detach().item()
            * batch_size
        )
        total_samples += batch_size

        if (
            COMPUTE_TRAIN_ONE_STEP_METRICS
            and batch_index % TRAIN_METRIC_EVERY == 0
        ):
            with torch.no_grad():
                metric_accumulator.update(
                    prediction=outputs[
                        "reconstruction"
                    ],
                    target=hsi,
                )
                metric_batches += 1

        if (
            batch_index % PRINT_EVERY == 0
            or batch_index == len(loader)
        ):
            message = (
                f"  Batch {batch_index:04d}/"
                f"{len(loader):04d} | "
                f"total={total_loss_sum / total_samples:.6f} | "
                f"residual L1={residual_l1_sum / total_samples:.6f} | "
                f"reconstruction L1="
                f"{reconstruction_l1_sum / total_samples:.6f} | "
                f"grad={float(gradient_norm):.4f}"
            )

            if metric_batches > 0:
                current_metrics = (
                    metric_accumulator.compute()
                )
                message += (
                    f" | one-step MRAE="
                    f"{current_metrics['mrae']:.6f}"
                    f" | one-step RMSE="
                    f"{current_metrics['rmse']:.6f}"
                    f" | one-step SAM="
                    f"{current_metrics['sam']:.6f}"
                    f" | one-step PSNR="
                    f"{current_metrics['psnr']:.4f}"
                    f" | one-step SSIM="
                    f"{current_metrics['ssim']:.4f}"
                )

            print(message)

    if total_samples == 0:
        raise RuntimeError(
            "Every training batch was skipped because of non-finite gradients."
        )

    result = {
        "total_loss": (
            total_loss_sum / total_samples
        ),
        "residual_l1": (
            residual_l1_sum / total_samples
        ),
        "reconstruction_l1": (
            reconstruction_l1_sum / total_samples
        ),
        "skipped_nonfinite_batches": skipped_nonfinite_batches,
        "timestep_losses": (
            finalize_timestep_tracker(
                timestep_tracker
            )
        ),
    }

    if metric_batches > 0:
        one_step_metrics = (
            metric_accumulator.compute()
        )
        result.update(
            {
                f"one_step_{key}": value
                for key, value
                in one_step_metrics.items()
            }
        )

    return result


@torch.no_grad()
def validate_diffusion_loss(
    model: MSTPlusPlusResidualBBDM,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool,
) -> dict:
    model.eval()

    total_loss_sum = 0.0
    residual_l1_sum = 0.0
    reconstruction_l1_sum = 0.0
    total_samples = 0
    sample_offset = 0
    timestep_tracker = create_timestep_tracker(
        model.bridge.num_timesteps
    )

    generator = torch.Generator(
        device=device
    )
    generator.manual_seed(
        SEED + 10_000
    )

    for hsi, rgb in loader:
        hsi = hsi.to(
            device,
            non_blocking=True,
        )
        rgb = rgb.to(
            device,
            non_blocking=True,
        )

        batch_size = hsi.shape[0]

        timestep = (
            torch.arange(
                sample_offset,
                sample_offset + batch_size,
                device=device,
            )
            % model.bridge.num_timesteps
        ) + 1
        timestep = timestep.long()
        sample_offset += batch_size

        noise = torch.randn(
            hsi.shape,
            generator=generator,
            device=device,
            dtype=hsi.dtype,
        )

        coarse_hsi = get_coarse_prediction_fp32(
            model=model,
            rgb=rgb,
        )

        with autocast_context(
            device=device,
            enabled=use_amp,
        ):
            x_t, used_noise = model.bridge.q_sample(
                ground_truth=hsi,
                coarse_hsi=coarse_hsi,
                t=timestep,
                noise=noise,
            )
            predicted_residual = model.bridge.denoiser(
                x_t=x_t,
                coarse_hsi=coarse_hsi,
                rgb=rgb,
                t=timestep,
                total_steps=model.bridge.num_timesteps,
            )
            target_residual = hsi - coarse_hsi
            reconstruction = coarse_hsi + predicted_residual

            outputs = {
                "t": timestep,
                "x_t": x_t,
                "noise": used_noise,
                "target_residual": target_residual,
                "predicted_residual": predicted_residual,
                "reconstruction": reconstruction,
                "coarse_hsi": coarse_hsi,
            }

            (
                _,
                residual_l1_loss,
                reconstruction_l1_loss,
            ) = calculate_training_losses(
                outputs
            )

        per_sample_residual_l1 = F.l1_loss(
            outputs["predicted_residual"].float(),
            outputs["target_residual"].float(),
            reduction="none",
        ).mean(dim=(1, 2, 3))

        per_sample_reconstruction_l1 = F.l1_loss(
            outputs["reconstruction"].float(),
            hsi.float(),
            reduction="none",
        ).mean(dim=(1, 2, 3))

        per_sample_total = (
            RESIDUAL_L1_WEIGHT
            * per_sample_residual_l1
            + RECONSTRUCTION_L1_WEIGHT
            * per_sample_reconstruction_l1
        )

        total_loss_sum += per_sample_total.sum().item()
        residual_l1_sum += per_sample_residual_l1.sum().item()
        reconstruction_l1_sum += (
            per_sample_reconstruction_l1.sum().item()
        )
        total_samples += batch_size

        update_timestep_tracker(
            tracker=timestep_tracker,
            timestep=timestep,
            predicted_residual=outputs["predicted_residual"],
            target_residual=outputs["target_residual"],
        )

    return {
        "total_loss": (
            total_loss_sum / total_samples
        ),
        "residual_l1": (
            residual_l1_sum / total_samples
        ),
        "reconstruction_l1": (
            reconstruction_l1_sum / total_samples
        ),
        "timestep_losses": (
            finalize_timestep_tracker(
                timestep_tracker
            )
        ),
    }


@torch.no_grad()
def validate_reconstruction_metrics(
    model: MSTPlusPlusResidualBBDM,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool,
    max_images: Optional[int],
) -> dict:
    """Compute actual RGB -> HSI metrics using the complete reverse sampler."""
    model.eval()

    refined_accumulator = HSIMetricAccumulator(
        data_range=METRIC_DATA_RANGE,
        clamp_prediction=(
            CLAMP_PREDICTION_FOR_METRICS
        ),
    )
    mst_accumulator = HSIMetricAccumulator(
        data_range=METRIC_DATA_RANGE,
        clamp_prediction=(
            CLAMP_PREDICTION_FOR_METRICS
        ),
    )

    evaluated_images = 0

    rng_devices = []
    if device.type == "cuda":
        rng_devices = [
            device.index
            if device.index is not None
            else torch.cuda.current_device()
        ]

    with torch.random.fork_rng(
        devices=rng_devices,
        enabled=True,
    ):
        torch.manual_seed(SEED + 20_000)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(
                SEED + 20_000
            )

        for hsi, rgb in loader:
            if (
                max_images is not None
                and evaluated_images >= max_images
            ):
                break

            if max_images is not None:
                remaining = (
                    max_images - evaluated_images
                )
                hsi = hsi[:remaining]
                rgb = rgb[:remaining]

            hsi = hsi.to(
                device,
                non_blocking=True,
            )
            rgb = rgb.to(
                device,
                non_blocking=True,
            )

            coarse_hsi = get_coarse_prediction_fp32(
                model=model,
                rgb=rgb,
            )

            with autocast_context(
                device=device,
                enabled=use_amp,
            ):
                refined_hsi = model.bridge.sample(
                    rgb=rgb,
                    coarse_hsi=coarse_hsi,
                    clip_denoised=INFERENCE_CLIP_DENOISED,
                    stochastic=INFERENCE_STOCHASTIC,
                )

            refined_accumulator.update(
                prediction=refined_hsi,
                target=hsi,
            )
            mst_accumulator.update(
                prediction=coarse_hsi,
                target=hsi,
            )

            evaluated_images += hsi.shape[0]

    refined_metrics = refined_accumulator.compute()
    mst_metrics = mst_accumulator.compute()

    result = {
        **refined_metrics,
        "evaluated_images": evaluated_images,
        "sampling_steps": model.bridge.num_timesteps,
    }
    result.update(
        {
            f"mst_{key}": value
            for key, value in mst_metrics.items()
        }
    )

    return result


# ============================================================================
# Inference preview saving
# ============================================================================

def _normalize_rgb_for_display(
    rgb: np.ndarray,
) -> np.ndarray:
    rgb = np.transpose(
        rgb,
        (1, 2, 0),
    )
    return np.clip(
        rgb,
        0.0,
        1.0,
    )


def _hsi_to_pseudo_rgb_triplet(
    target: np.ndarray,
    mst_prediction: np.ndarray,
    refined_prediction: np.ndarray,
    bands: Tuple[int, int, int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    channel_count = target.shape[0]

    for band in bands:
        if not 0 <= band < channel_count:
            raise ValueError(
                f"Visualization band {band} is outside "
                f"the valid range [0, {channel_count - 1}]."
            )

    target_rgb = np.stack(
        [target[band] for band in bands],
        axis=-1,
    )
    mst_rgb = np.stack(
        [mst_prediction[band] for band in bands],
        axis=-1,
    )
    refined_rgb = np.stack(
        [refined_prediction[band] for band in bands],
        axis=-1,
    )

    # Use target-derived scaling for all HSI previews.
    minimum = target_rgb.min(
        axis=(0, 1),
        keepdims=True,
    )
    maximum = target_rgb.max(
        axis=(0, 1),
        keepdims=True,
    )
    scale = maximum - minimum + 1e-8

    target_rgb = (
        target_rgb - minimum
    ) / scale
    mst_rgb = (
        mst_rgb - minimum
    ) / scale
    refined_rgb = (
        refined_rgb - minimum
    ) / scale

    return (
        np.clip(target_rgb, 0.0, 1.0),
        np.clip(mst_rgb, 0.0, 1.0),
        np.clip(refined_rgb, 0.0, 1.0),
    )


def save_inference_preview(
    output_path: Path,
    rgb: np.ndarray,
    target_hsi: np.ndarray,
    mst_prediction_hsi: np.ndarray,
    refined_prediction_hsi: np.ndarray,
) -> None:
    rgb_display = _normalize_rgb_for_display(
        rgb
    )
    (
        target_display,
        mst_display,
        refined_display,
    ) = _hsi_to_pseudo_rgb_triplet(
        target=target_hsi,
        mst_prediction=mst_prediction_hsi,
        refined_prediction=refined_prediction_hsi,
        bands=VISUALIZATION_BANDS,
    )

    panels = [
        rgb_display,
        target_display,
        mst_display,
        refined_display,
    ]

    panel_images = [
        Image.fromarray(
            (panel * 255.0)
            .round()
            .astype(np.uint8)
        )
        for panel in panels
    ]

    width = sum(
        image.width
        for image in panel_images
    )
    height = max(
        image.height
        for image in panel_images
    )

    canvas = Image.new(
        "RGB",
        (width, height),
    )

    x_offset = 0
    for image in panel_images:
        canvas.paste(
            image,
            (x_offset, 0),
        )
        x_offset += image.width

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    canvas.save(output_path)


@torch.no_grad()
def run_random_validation_inference(
    model: MSTPlusPlusResidualBBDM,
    validation_pairs: Sequence[
        Tuple[Path, Path]
    ],
    device: torch.device,
    use_amp: bool,
    output_directory: Path,
    number_of_images: int,
    save_visualizations: bool,
) -> dict:
    model.eval()

    if not validation_pairs:
        raise RuntimeError(
            "The validation pair list is empty."
        )

    number_to_select = min(
        number_of_images,
        len(validation_pairs),
    )

    selected_indices = random.Random(
        SEED
    ).sample(
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

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    refined_accumulator = HSIMetricAccumulator(
        data_range=METRIC_DATA_RANGE,
        clamp_prediction=(
            CLAMP_PREDICTION_FOR_METRICS
        ),
    )
    mst_accumulator = HSIMetricAccumulator(
        data_range=METRIC_DATA_RANGE,
        clamp_prediction=(
            CLAMP_PREDICTION_FOR_METRICS
        ),
    )

    print(
        f"\nRunning full-resolution inference on "
        f"{number_to_select} random validation images..."
    )
    print(
        f"Visualizations: "
        f"{'enabled' if save_visualizations else 'disabled'}"
    )

    for output_index, dataset_index in enumerate(
        selected_indices,
        start=1,
    ):
        (
            hsi,
            rgb,
            hsi_path_string,
            rgb_path_string,
        ) = dataset[dataset_index]

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
            .to(device)
        )

        coarse_batch = get_coarse_prediction_fp32(
            model=model,
            rgb=rgb_batch,
        )

        with autocast_context(
            device=device,
            enabled=use_amp,
        ):
            refined_batch = model.bridge.sample(
                rgb=rgb_batch,
                coarse_hsi=coarse_batch,
                clip_denoised=INFERENCE_CLIP_DENOISED,
                stochastic=INFERENCE_STOCHASTIC,
            )

        prediction = refined_batch[
            :,
            :,
            :original_height,
            :original_width,
        ].float().cpu()

        mst_prediction = coarse_batch[
            :,
            :,
            :original_height,
            :original_width,
        ].float().cpu()

        target_batch = (
            hsi.unsqueeze(0).float()
        )

        refined_sample_accumulator = HSIMetricAccumulator(
            data_range=METRIC_DATA_RANGE,
            clamp_prediction=(
                CLAMP_PREDICTION_FOR_METRICS
            ),
        )
        refined_sample_accumulator.update(
            prediction=prediction,
            target=target_batch,
        )
        refined_metrics = (
            refined_sample_accumulator.compute()
        )

        mst_sample_accumulator = HSIMetricAccumulator(
            data_range=METRIC_DATA_RANGE,
            clamp_prediction=(
                CLAMP_PREDICTION_FOR_METRICS
            ),
        )
        mst_sample_accumulator.update(
            prediction=mst_prediction,
            target=target_batch,
        )
        mst_metrics = mst_sample_accumulator.compute()

        refined_accumulator.update(
            prediction=prediction,
            target=target_batch,
        )
        mst_accumulator.update(
            prediction=mst_prediction,
            target=target_batch,
        )

        stem = Path(hsi_path_string).stem
        prefix = (
            output_directory
            / f"{output_index:02d}_{stem}"
        )

        prediction_numpy = prediction[0].numpy()
        mst_prediction_numpy = mst_prediction[0].numpy()
        target_numpy = hsi.numpy()
        rgb_numpy = rgb.numpy()

        np.savez_compressed(
            str(prefix) + ".npz",
            prediction=prediction_numpy,
            mst_prediction=mst_prediction_numpy,
            predicted_residual=(
                prediction_numpy
                - mst_prediction_numpy
            ),
            target=target_numpy,
            rgb=rgb_numpy,
            hsi_path=hsi_path_string,
            rgb_path=rgb_path_string,
            refined_metrics=np.asarray(
                [
                    refined_metrics["mrae"],
                    refined_metrics["rmse"],
                    refined_metrics["sam"],
                    refined_metrics["psnr"],
                    refined_metrics["ssim"],
                ],
                dtype=np.float64,
            ),
            mst_metrics=np.asarray(
                [
                    mst_metrics["mrae"],
                    mst_metrics["rmse"],
                    mst_metrics["sam"],
                    mst_metrics["psnr"],
                    mst_metrics["ssim"],
                ],
                dtype=np.float64,
            ),
            metric_names=np.asarray(
                [
                    "mrae",
                    "rmse",
                    "sam_radians",
                    "psnr",
                    "ssim",
                ]
            ),
        )

        if save_visualizations:
            save_inference_preview(
                output_path=Path(
                    str(prefix) + "_preview.png"
                ),
                rgb=rgb_numpy,
                target_hsi=target_numpy,
                mst_prediction_hsi=(
                    mst_prediction_numpy
                ),
                refined_prediction_hsi=(
                    prediction_numpy
                ),
            )

        print(
            f"  [{output_index}/{number_to_select}] {stem} | "
            f"Refined MRAE={refined_metrics['mrae']:.6f} | "
            f"RMSE={refined_metrics['rmse']:.6f} | "
            f"SAM={refined_metrics['sam']:.6f} rad | "
            f"PSNR={refined_metrics['psnr']:.4f} | "
            f"SSIM={refined_metrics['ssim']:.4f}"
        )
        print(
            f"      MST++ MRAE={mst_metrics['mrae']:.6f} | "
            f"RMSE={mst_metrics['rmse']:.6f} | "
            f"SAM={mst_metrics['sam']:.6f} rad | "
            f"PSNR={mst_metrics['psnr']:.4f} | "
            f"SSIM={mst_metrics['ssim']:.4f}"
        )

    refined_overall = refined_accumulator.compute()
    mst_overall = mst_accumulator.compute()

    metrics_path = (
        output_directory
        / "random_inference_metrics.txt"
    )

    metrics_path.write_text(
        "\n".join(
            [
                f"images={number_to_select}",
                f"sampling_steps={model.bridge.num_timesteps}",
                f"visualizations={save_visualizations}",
                f"refined_mrae={refined_overall['mrae']}",
                f"refined_rmse={refined_overall['rmse']}",
                f"refined_sam_radians={refined_overall['sam']}",
                f"refined_psnr={refined_overall['psnr']}",
                f"refined_ssim={refined_overall['ssim']}",
                f"mst_mrae={mst_overall['mrae']}",
                f"mst_rmse={mst_overall['rmse']}",
                f"mst_sam_radians={mst_overall['sam']}",
                f"mst_psnr={mst_overall['psnr']}",
                f"mst_ssim={mst_overall['ssim']}",
            ]
        ),
        encoding="utf-8",
    )

    print(
        "\nRandom inference mean refined metrics | "
        f"MRAE={refined_overall['mrae']:.6f} | "
        f"RMSE={refined_overall['rmse']:.6f} | "
        f"SAM={refined_overall['sam']:.6f} rad | "
        f"PSNR={refined_overall['psnr']:.4f} | "
        f"SSIM={refined_overall['ssim']:.4f}"
    )
    print(
        "Random inference mean MST++ metrics | "
        f"MRAE={mst_overall['mrae']:.6f} | "
        f"RMSE={mst_overall['rmse']:.6f} | "
        f"SAM={mst_overall['sam']:.6f} rad | "
        f"PSNR={mst_overall['psnr']:.4f} | "
        f"SSIM={mst_overall['ssim']:.4f}"
    )

    return {
        "refined": refined_overall,
        "mst": mst_overall,
    }

# ============================================================================
# DataLoader construction
# ============================================================================

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
        persistent_workers=(
            NUM_WORKERS > 0
        ),
        worker_init_fn=seed_worker,
        generator=generator,
    )



# ============================================================================
# Main workflows
# ============================================================================

def prepare_pairs(
    output_directory: Path,
) -> Tuple[
    List[Tuple[Path, Path]],
    List[Tuple[Path, Path]],
]:
    train_pairs = pair_hsi_rgb_files(
        hsi_directory=TRAIN_HSI_DIR,
        rgb_directory=TRAIN_RGB_DIR,
    )
    validation_pairs = pair_hsi_rgb_files(
        hsi_directory=VALIDATION_HSI_DIR,
        rgb_directory=VALIDATION_RGB_DIR,
    )

    train_pairs = filter_valid_pairs(
        pairs=train_pairs,
        hsi_channels=HSI_CHANNELS,
        log_path=(
            output_directory
            / "invalid_training_pairs.txt"
        ),
        cache_path=TRAIN_PAIR_VALIDATION_CACHE,
    )
    validation_pairs = filter_valid_pairs(
        pairs=validation_pairs,
        hsi_channels=HSI_CHANNELS,
        log_path=(
            output_directory
            / "invalid_validation_pairs.txt"
        ),
        cache_path=VALIDATION_PAIR_VALIDATION_CACHE,
    )

    return train_pairs, validation_pairs


def train_workflow(
    device: torch.device,
    use_amp: bool,
    train_pairs: Sequence[Tuple[Path, Path]],
    validation_pairs: Sequence[
        Tuple[Path, Path]
    ],
    output_directory: Path,
) -> MSTPlusPlusResidualBBDM:
    if TRAIN_CROP_SIZE % MODEL_DOWNSAMPLE_FACTOR != 0:
        raise ValueError(
            f"TRAIN_CROP_SIZE={TRAIN_CROP_SIZE} must be divisible by "
            f"{MODEL_DOWNSAMPLE_FACTOR}."
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
        crop_size=TRAIN_CROP_SIZE,
        patches_per_image=1,
        training=False,
        normalization=HSI_NORMALIZATION,
        augment=False,
    )

    train_loader = make_loader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=(
            len(train_dataset) >= BATCH_SIZE
        ),
        device=device,
    )
    validation_loader = make_loader(
        dataset=validation_dataset,
        batch_size=VALIDATION_BATCH_SIZE,
        shuffle=False,
        drop_last=False,
        device=device,
    )

    model = build_residual_diffusion(
        device=device,
    )

    trainable_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]

    frozen_mst_parameters = sum(
        parameter.numel()
        for parameter in model.coarse_model.parameters()
    )

    print(
        f"\nDevice: {device}\n"
        f"Mixed precision: {use_amp}\n"
        f"AMP dtype: {get_amp_dtype(device) if use_amp else 'float32'}\n"
        f"Training pairs: {len(train_pairs)}\n"
        f"Validation pairs: {len(validation_pairs)}\n"
        f"Frozen MST++ parameters: {frozen_mst_parameters:,}\n"
        f"Trainable residual parameters: "
        f"{sum(p.numel() for p in trainable_parameters):,}"
    )

    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=NUM_EPOCHS,
            eta_min=MIN_LEARNING_RATE,
        )
    )

    amp_dtype = get_amp_dtype(device)

    scaler = GradScaler(
        enabled=(
            use_amp
            and amp_dtype == torch.float16
        ),
        init_scale=FP16_INITIAL_SCALE,
        growth_interval=FP16_GROWTH_INTERVAL,
    )

    print(
        f"GradScaler enabled: {scaler.is_enabled()} | "
        f"initial scale: {float(scaler.get_scale()):.1f}"
    )

    start_epoch = 1
    best_validation_residual_loss = float("inf")

    if RESUME_CHECKPOINT is not None:
        resume_checkpoint = (
            _load_torch_checkpoint(
                RESUME_CHECKPOINT,
                device="cpu",
            )
        )

        load_residual_diffusion_weights(
            model,
            resume_checkpoint,
        )

        optimizer.load_state_dict(
            resume_checkpoint[
                "optimizer_state_dict"
            ]
        )
        scheduler.load_state_dict(
            resume_checkpoint[
                "scheduler_state_dict"
            ]
        )

        if OVERRIDE_RESUMED_LEARNING_RATE:
            for parameter_group in optimizer.param_groups:
                parameter_group["lr"] = LEARNING_RATE
                parameter_group["initial_lr"] = LEARNING_RATE

            scheduler.base_lrs = [
                LEARNING_RATE
                for _ in optimizer.param_groups
            ]
            scheduler._last_lr = [
                LEARNING_RATE
                for _ in optimizer.param_groups
            ]

            print(
                "Overrode resumed learning rate with "
                f"LEARNING_RATE={LEARNING_RATE:.2e}."
            )

        if (
            LOAD_SCALER_STATE_ON_RESUME
            and "scaler_state_dict" in resume_checkpoint
        ):
            scaler.load_state_dict(
                resume_checkpoint[
                    "scaler_state_dict"
                ]
            )
        elif scaler.is_enabled():
            print(
                "Using a fresh, lower FP16 GradScaler state "
                "instead of the checkpoint scaler state."
            )

        start_epoch = int(
            resume_checkpoint.get(
                "epoch",
                0,
            )
        ) + 1

        best_validation_residual_loss = float(
            resume_checkpoint.get(
                "best_validation_residual_loss",
                float("inf"),
            )
        )

        print(
            f"\nResumed from {RESUME_CHECKPOINT} "
            f"at epoch {start_epoch}."
        )

    for epoch in range(
        start_epoch,
        NUM_EPOCHS + 1,
    ):
        print(
            f"\n{'=' * 80}\n"
            f"Epoch {epoch}/{NUM_EPOCHS}\n"
            f"{'=' * 80}"
        )

        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            use_amp=use_amp,
        )

        validation_diffusion = validate_diffusion_loss(
            model=model,
            loader=validation_loader,
            device=device,
            use_amp=use_amp,
        )

        validation_reconstruction = (
            validate_reconstruction_metrics(
                model=model,
                loader=validation_loader,
                device=device,
                use_amp=use_amp,
                max_images=(
                    VALIDATION_METRIC_MAX_IMAGES
                ),
            )
        )

        scheduler.step()

        current_learning_rate = (
            optimizer.param_groups[0]["lr"]
        )

        print(
            f"\nEpoch {epoch:03d}/{NUM_EPOCHS:03d} | "
            f"LR={current_learning_rate:.2e} | "
            f"train total={train_metrics['total_loss']:.6f} | "
            f"train residual L1={train_metrics['residual_l1']:.6f} | "
            f"skipped overflow batches="
            f"{train_metrics['skipped_nonfinite_batches']} | "
            f"validation residual L1="
            f"{validation_diffusion['residual_l1']:.6f}"
        )

        if "one_step_psnr" in train_metrics:
            print(
                "Training one-step reconstruction metrics | "
                f"MRAE={train_metrics['one_step_mrae']:.6f} | "
                f"RMSE={train_metrics['one_step_rmse']:.6f} | "
                f"SAM={train_metrics['one_step_sam']:.6f} rad | "
                f"PSNR={train_metrics['one_step_psnr']:.4f} | "
                f"SSIM={train_metrics['one_step_ssim']:.4f}"
            )

        print(
            "Validation sampled refined metrics | "
            f"images={validation_reconstruction['evaluated_images']} | "
            f"steps={validation_reconstruction['sampling_steps']} | "
            f"MRAE={validation_reconstruction['mrae']:.6f} | "
            f"RMSE={validation_reconstruction['rmse']:.6f} | "
            f"SAM={validation_reconstruction['sam']:.6f} rad | "
            f"PSNR={validation_reconstruction['psnr']:.4f} | "
            f"SSIM={validation_reconstruction['ssim']:.4f}"
        )
        print(
            "Validation frozen MST++ metrics | "
            f"MRAE={validation_reconstruction['mst_mrae']:.6f} | "
            f"RMSE={validation_reconstruction['mst_rmse']:.6f} | "
            f"SAM={validation_reconstruction['mst_sam']:.6f} rad | "
            f"PSNR={validation_reconstruction['mst_psnr']:.4f} | "
            f"SSIM={validation_reconstruction['mst_ssim']:.4f}"
        )

        print_timestep_tracker(
            title="Training residual L1 by timestep",
            result=train_metrics[
                "timestep_losses"
            ],
        )
        print_timestep_tracker(
            title="Validation residual L1 by timestep",
            result=validation_diffusion[
                "timestep_losses"
            ],
        )

        combined_validation_metrics = {
            "diffusion": validation_diffusion,
            "reconstruction": (
                validation_reconstruction
            ),
        }

        save_checkpoint(
            output_path=(
                output_directory
                / "last_bbdm.pth"
            ),
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            epoch=epoch,
            best_validation_residual_loss=(
                best_validation_residual_loss
            ),
            validation_metrics=(
                combined_validation_metrics
            ),
        )

        if (
            validation_diffusion["residual_l1"]
            < best_validation_residual_loss
        ):
            best_validation_residual_loss = (
                validation_diffusion["residual_l1"]
            )

            save_checkpoint(
                output_path=(
                    output_directory
                    / "best_bbdm.pth"
                ),
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                epoch=epoch,
                best_validation_residual_loss=(
                    best_validation_residual_loss
                ),
                validation_metrics=(
                    combined_validation_metrics
                ),
            )

            print(
                "New best checkpoint | "
                f"validation residual L1="
                f"{best_validation_residual_loss:.6f}"
            )

    return model


def load_model_for_inference(
    checkpoint_path: str,
    device: torch.device,
) -> MSTPlusPlusResidualBBDM:
    checkpoint = _load_torch_checkpoint(
        checkpoint_path,
        device="cpu",
    )

    model_config = (
        checkpoint.get("model_config", {})
        if isinstance(checkpoint, dict)
        else {}
    )

    model = build_residual_diffusion(
        device=device,
        model_config=model_config,
    )

    load_residual_diffusion_weights(
        model,
        checkpoint,
    )

    model.eval()
    return model


def parse_command_line_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train or evaluate the frozen-MST++ residual Brownian-bridge model."
        )
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help=(
            "Save RGB/ground-truth/MST++/refined preview images. "
            "In train mode this also runs random validation inference "
            "after training using the best checkpoint."
        ),
    )
    parser.add_argument(
        "--visualization-images",
        type=int,
        default=NUM_RANDOM_INFERENCE_IMAGES,
        help=(
            "Number of random validation images used for final inference "
            "and optional visualization."
        ),
    )

    arguments = parser.parse_args()

    if arguments.visualization_images < 1:
        parser.error(
            "--visualization-images must be at least 1."
        )

    return arguments


def main() -> None:
    arguments = parse_command_line_arguments()
    set_seed(SEED)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )
    use_amp = (
        USE_AMP
        and device.type == "cuda"
    )

    output_directory = Path(OUTPUT_DIR)
    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    if RUN_MODE not in {
        "train",
        "infer",
        "train_and_infer",
    }:
        raise ValueError(
            "RUN_MODE must be 'train', 'infer', "
            "or 'train_and_infer'."
        )

    if RUN_MODE in {
        "train",
        "train_and_infer",
    }:
        train_pairs, validation_pairs = (
            prepare_pairs(output_directory)
        )
        model = train_workflow(
            device=device,
            use_amp=use_amp,
            train_pairs=train_pairs,
            validation_pairs=validation_pairs,
            output_directory=output_directory,
        )
    else:
        validation_pairs = pair_hsi_rgb_files(
            hsi_directory=VALIDATION_HSI_DIR,
            rgb_directory=VALIDATION_RGB_DIR,
        )
        validation_pairs = filter_valid_pairs(
            pairs=validation_pairs,
            hsi_channels=HSI_CHANNELS,
            log_path=(
                output_directory
                / "invalid_validation_pairs.txt"
            ),
            cache_path=VALIDATION_PAIR_VALIDATION_CACHE,
        )

        model = load_model_for_inference(
            checkpoint_path=INFERENCE_CHECKPOINT,
            device=device,
        )

    should_run_final_inference = (
        RUN_MODE in {
            "infer",
            "train_and_infer",
        }
        or arguments.visualize
    )

    if should_run_final_inference:
        # After training, use the saved best checkpoint rather than the final
        # epoch's in-memory weights.
        if RUN_MODE in {
            "train",
            "train_and_infer",
        }:
            model = load_model_for_inference(
                checkpoint_path=str(
                    output_directory
                    / "best_bbdm.pth"
                ),
                device=device,
            )

        run_random_validation_inference(
            model=model,
            validation_pairs=validation_pairs,
            device=device,
            use_amp=use_amp,
            output_directory=Path(
                INFERENCE_OUTPUT_DIR
            ),
            number_of_images=(
                arguments.visualization_images
            ),
            save_visualizations=(
                arguments.visualize
            ),
        )


if __name__ == "__main__":
    main()
