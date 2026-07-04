"""Train a ResShiftSSR (RGB -> HSI) diffusion model from paired RGB/HSI images.

Key behaviour
-------------
1. Every HSI cube and its paired RGB image are loaded at their original
   (matching) spatial resolution.
2. Training samples are random spatial crops taken from the native-resolution
   RGB/HSI pair, using the SAME crop location for both.
3. The same spatial augmentation (flip / 90-degree rotation) is applied
   identically to the RGB image and every spectral band of the HSI cube.
4. Validation uses the complete native-resolution image pair, not a center crop.
5. Validation batch size is one so differently sized images are supported.
6. Images are padded only immediately before the model forward pass so their
   height and width are compatible with the denoiser's downsampling factor.
   Every prediction is cropped back to the original size before losses and
   metrics are calculated.
7. Unlike a VAE, this model is a conditional residual-shifting diffusion model
   (ResShift). There is no KL term / latent code to regularise. The training
   objective is therefore:
       loss = pixel loss(pred_x0, hsi_gt) + sam_weight(epoch) * spectral_angle_loss
   where `pred_x0` is the denoiser's direct estimate of the clean HSI cube at
   a randomly sampled diffusion timestep t (the standard ResShift training
   target). The spectral-angle term is included (with its own warm-up
   schedule, replacing the old KL-beta warm-up) because it is a well known,
   differentiable, HSI-specific regulariser that discourages spectral
   distortion introduced by pixel-wise losses alone.
8. Regardless of the training loss, mrae / rmse / sam / psnr / ssim are always
   computed every epoch for the coarse network's prediction (y0) and for the
   denoiser's cheap single-step x0 estimate. The fully diffusion-refined
   output (T-step reverse sampling) is only run ONCE, after the final
   training epoch, since it is far more expensive; its metrics are saved
   in the final checkpoint and printed at the end of the run.
"""

from __future__ import annotations

import hashlib
import random
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import h5py
import numpy as np
import scipy.io as sio
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset

# TODO: adjust this import to wherever you saved the ResShiftSSR model file
# (the file that defines ResShiftDiffusion, ResShiftDenoiser and ResShiftSSR).
from model.resshift import ResShiftSSR

from loss.mrae import mrae
from loss.psnr import psnr
from loss.rmse import rmse
from loss.sam import sam
from loss.ssim import ssim


# =============================================================================
# Configuration
# =============================================================================

HSI_DATA_DIR = (
    "/kaggle/input/datasets/sriramhari14/ntire-2022/"
    "Train_spectral/Train_spectral"
)
RGB_DATA_DIR = (
    "/kaggle/input/datasets/sriramhari14/ntire-2022/"
    "Train_RGB/Train_RGB"
)
OUTPUT_DIR = "./resshift_checkpoints"

HSI_KEY = "cube"
HSI_CHANNELS = 31
RGB_CHANNELS = 3
SUPPORTED_HSI_EXTENSIONS = {".mat", ".npy", ".npz", ".pt", ".pth"}
SUPPORTED_RGB_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

# Coarse RGB -> HSI network checkpoint (loaded inside ResShiftSSR).
MST_CKPT_PATH = None
FREEZE_COARSE = True

# Diffusion / denoiser architecture. These values must match models.ResShiftSSR.
DIFFUSION_T = 15
DIFFUSION_P = 0.3
DIFFUSION_KAPPA = 2.0
BASE_DIM = 64
DIM_MULTS = (1, 2, 4)
NUM_RES_BLOCKS = 2

# The denoiser downsamples (len(DIM_MULTS) - 1) times by a factor of 2 each
# time (its last level is an nn.Identity instead of a Downsample). Padding
# must make H and W divisible by this factor.
MODEL_DOWNSAMPLE_FACTOR = 2 ** (len(DIM_MULTS) - 1)

# Training uses native-resolution pairs as the source, then samples random
# crops. TRAIN_CROP_SIZE must already be divisible by MODEL_DOWNSAMPLE_FACTOR.
TRAIN_CROP_SIZE = 256
CROPS_PER_IMAGE = 4

# Spatial augmentation. These transforms are applied identically to the RGB
# image and every HSI band, preserving pixel/spectral values.
USE_AUGMENTATION = True
HORIZONTAL_FLIP_PROBABILITY = 0.5
VERTICAL_FLIP_PROBABILITY = 0.5
USE_RANDOM_90_DEGREE_ROTATION = True

# "none", "minmax", or "band_minmax". Applied to the HSI cube only; RGB is
# always scaled to [0, 1] since that is what the coarse RGB->HSI network
# expects.
NORMALIZATION = "none"

VALIDATION_FRACTION = 0.10
SEED = 42

TRAIN_BATCH_SIZE = 4
# Keep this at one because validation images can have different H x W shapes.
VALIDATION_BATCH_SIZE = 1
NUM_WORKERS = 4

NUM_EPOCHS = 75
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
GRADIENT_CLIP_NORM = 1.0
USE_AMP = True
PRINT_EVERY = 30

# Pixel loss used to regress the denoiser's predicted x0 towards the ground
# truth HSI cube. "mse", "l1", or "smooth_l1".
DIFFUSION_LOSS_TYPE = "mse"

# Spectral-angle regularisation on the predicted x0, ramped up over training
# (mirrors the old KL-beta warm-up, but there is no latent code here).
SAM_LOSS_WEIGHT_START = 0.0
SAM_LOSS_WEIGHT_END = 0.05
SAM_WARMUP_EPOCHS = 30

# Full reverse-diffusion sampling (T forward passes through the denoiser,
# per validation image) is expensive, so it is only run once, after the
# final training epoch, to report the model's true inference-time
# mrae/rmse/sam/psnr/ssim. Every other epoch only computes the cheap
# single-step x0 estimate (quick_metrics) for monitoring convergence.

# The best checkpoint is selected using the cheap single-step validation
# MRAE (quick_metrics), since that is available every epoch. A separate,
# final checkpoint with true full-sampling metrics is saved after the last
# epoch.
VALIDATION_CACHE = Path(OUTPUT_DIR) / "resshift_validation_cache.pth"
HSI_CHECKER_VERSION = "resshift-rgb-hsi-pair-cache-v1"


# =============================================================================
# Reproducibility
# =============================================================================


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id: int) -> None:
    """Give every DataLoader worker a reproducible independent RNG state."""
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# =============================================================================
# HSI loading (unchanged from the base HSI pipeline)
# =============================================================================


def _select_largest_3d_array(
    arrays: Sequence[Tuple[str, np.ndarray]],
    file_path: Path,
) -> np.ndarray:
    candidates = [
        (name, value)
        for name, value in arrays
        if isinstance(value, np.ndarray)
        and value.ndim == 3
        and np.issubdtype(value.dtype, np.number)
    ]
    if not candidates:
        raise ValueError(f"No numerical three-dimensional HSI array found in {file_path}")
    return max(candidates, key=lambda item: item[1].size)[1]


def load_mat_v73(file_path: Path, hsi_key: str) -> np.ndarray:
    """Load a MATLAB v7.3 HDF5 file."""
    candidates: List[Tuple[str, np.ndarray]] = []

    with h5py.File(str(file_path), "r") as h5_file:
        if hsi_key in h5_file and isinstance(h5_file[hsi_key], h5py.Dataset):
            dataset = h5_file[hsi_key]
            if dataset.ndim == 3:
                candidates.append((hsi_key, np.asarray(dataset)))

        if not candidates:
            def visitor(name: str, obj: Any) -> None:
                if not isinstance(obj, h5py.Dataset) or obj.ndim != 3:
                    return
                try:
                    array = np.asarray(obj)
                    if np.issubdtype(array.dtype, np.number):
                        candidates.append((name, array))
                except Exception:
                    return

            h5_file.visititems(visitor)

    cube = _select_largest_3d_array(candidates, file_path)

    # MATLAB v7.3 arrays commonly appear with reversed axis order in h5py.
    # convert_to_chw() below still verifies the location of the 31-band axis.
    return np.transpose(cube, axes=tuple(range(cube.ndim - 1, -1, -1)))


def extract_array_from_dictionary(
    data: Dict[str, Any],
    file_path: Path,
    hsi_key: str,
) -> np.ndarray:
    if hsi_key in data:
        preferred = data[hsi_key]
        if isinstance(preferred, torch.Tensor):
            preferred = preferred.detach().cpu().numpy()
        if isinstance(preferred, np.ndarray) and preferred.ndim == 3:
            return preferred

    arrays: List[Tuple[str, np.ndarray]] = []
    for key, value in data.items():
        if key.startswith("__"):
            continue
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
        if isinstance(value, np.ndarray):
            arrays.append((key, value))

    return _select_largest_3d_array(arrays, file_path)


def load_hsi_file(file_path: Path, hsi_key: str = HSI_KEY) -> np.ndarray:
    extension = file_path.suffix.lower()

    if extension == ".npy":
        cube = np.load(file_path)

    elif extension == ".npz":
        with np.load(file_path) as loaded:
            if hsi_key in loaded.files and loaded[hsi_key].ndim == 3:
                cube = loaded[hsi_key]
            else:
                candidates = [
                    (key, loaded[key])
                    for key in loaded.files
                    if loaded[key].ndim == 3
                ]
                cube = _select_largest_3d_array(candidates, file_path)

    elif extension == ".mat":
        try:
            loaded = sio.loadmat(file_path)
            cube = extract_array_from_dictionary(loaded, file_path, hsi_key)
        except (NotImplementedError, ValueError):
            cube = load_mat_v73(file_path, hsi_key)

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
            cube = extract_array_from_dictionary(loaded, file_path, hsi_key)
        else:
            raise TypeError(
                f"Unsupported object type {type(loaded).__name__} in {file_path}"
            )

    else:
        raise ValueError(f"Unsupported HSI extension: {extension}")

    cube = np.asarray(cube, dtype=np.float32)
    cube = np.squeeze(cube)

    if cube.ndim != 3:
        raise ValueError(
            f"Expected a three-dimensional HSI cube in {file_path}, "
            f"but found shape {cube.shape}"
        )

    return cube


def convert_to_chw(
    cube: np.ndarray,
    hsi_channels: int,
    file_path: Path,
) -> np.ndarray:
    """Convert [H,W,C] or [C,H,W] to [C,H,W]."""
    if cube.shape[0] == hsi_channels:
        return cube
    if cube.shape[-1] == hsi_channels:
        return np.transpose(cube, (2, 0, 1))

    raise ValueError(
        f"Cannot locate the {hsi_channels}-band spectral axis in {file_path}. "
        f"Found shape {cube.shape}."
    )


def align_hsi_orientation(
    cube_chw: np.ndarray,
    target_hw: Tuple[int, int],
    file_path: Path,
) -> np.ndarray:
    """Transpose a [C,H,W] cube's spatial axes if needed to match target_hw.

    Some HSI datasets (e.g. NTIRE / ARAD_1K) store the spectral cube with H
    and W swapped relative to the paired RGB image. filter_valid_pairs()
    already accepts such pairs; this function performs the actual correction
    at load time so the crop/augmentation code always sees matching shapes.
    """
    current_hw = (cube_chw.shape[1], cube_chw.shape[2])

    if current_hw == target_hw:
        return cube_chw
    if current_hw == (target_hw[1], target_hw[0]):
        return np.transpose(cube_chw, (0, 2, 1))

    raise ValueError(
        f"Cannot align HSI spatial size {current_hw} in {file_path} with "
        f"target size {target_hw}, even after transposing."
    )


def normalize_cube(cube: np.ndarray, mode: str) -> np.ndarray:
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

    raise ValueError(f"Unknown normalization mode: {mode}")


# =============================================================================
# RGB loading (new: paired input for the coarse RGB -> HSI network)
# =============================================================================


def load_rgb_file(file_path: Path) -> np.ndarray:
    """Load an RGB image as a [3,H,W] float32 array scaled to [0, 1]."""
    extension = file_path.suffix.lower()

    if extension == ".npy":
        image = np.load(file_path)
        image = np.asarray(image, dtype=np.float32)
        if image.max() > 1.0 + 1e-3:
            image = image / 255.0
    else:
        with Image.open(file_path) as handle:
            image = np.asarray(handle.convert("RGB"), dtype=np.float32) / 255.0

    image = np.squeeze(image)
    if image.ndim != 3:
        raise ValueError(
            f"Expected a three-dimensional RGB image in {file_path}, "
            f"but found shape {image.shape}"
        )

    if image.shape[0] == RGB_CHANNELS:
        return image
    if image.shape[-1] == RGB_CHANNELS:
        return np.transpose(image, (2, 0, 1))

    raise ValueError(
        f"Cannot locate the {RGB_CHANNELS}-channel axis in {file_path}. "
        f"Found shape {image.shape}."
    )


def get_rgb_spatial_size(file_path: Path) -> Tuple[int, int]:
    """Return (H, W) for an RGB file without loading pixel data where possible."""
    if file_path.suffix.lower() == ".npy":
        array = np.load(file_path, mmap_mode="r")
        array = np.squeeze(array)
        if array.shape[0] == RGB_CHANNELS:
            return int(array.shape[1]), int(array.shape[2])
        return int(array.shape[0]), int(array.shape[1])

    with Image.open(file_path) as handle:
        width, height = handle.size
    return height, width


# =============================================================================
# File discovery, pairing, and validation
# =============================================================================


def find_files_with_extensions(data_dir: str, extensions: set[str]) -> List[Path]:
    root = Path(data_dir)
    if not root.exists():
        raise FileNotFoundError(f"Directory does not exist: {root}")

    files = sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in extensions
    )
    if not files:
        raise RuntimeError(f"No supported files found in {root}")

    return files


def find_paired_files(
    hsi_dir: str,
    rgb_dir: str,
) -> List[Tuple[Path, Path]]:
    """Pair HSI cubes with RGB images that share the same file stem."""
    hsi_files = find_files_with_extensions(hsi_dir, SUPPORTED_HSI_EXTENSIONS)
    rgb_files = find_files_with_extensions(rgb_dir, SUPPORTED_RGB_EXTENSIONS)

    rgb_by_stem = {path.stem: path for path in rgb_files}

    pairs: List[Tuple[Path, Path]] = []
    missing_rgb: List[str] = []
    for hsi_path in hsi_files:
        rgb_path = rgb_by_stem.get(hsi_path.stem)
        if rgb_path is None:
            missing_rgb.append(hsi_path.stem)
            continue
        pairs.append((hsi_path, rgb_path))

    if missing_rgb:
        print(
            f"\nWarning: {len(missing_rgb)} HSI file(s) had no matching RGB "
            f"image and will be skipped, e.g. {missing_rgb[:5]}"
        )

    if not pairs:
        raise RuntimeError("No matching HSI/RGB pairs were found.")

    return pairs


def make_pairs_fingerprint(pairs: Sequence[Tuple[Path, Path]]) -> str:
    """Create a cache fingerprint from file paths, sizes, and modification times."""
    records = []
    for hsi_path, rgb_path in pairs:
        hsi_stat = hsi_path.stat()
        rgb_stat = rgb_path.stat()
        records.append(
            f"{hsi_path.resolve()}|{hsi_stat.st_size}|{hsi_stat.st_mtime_ns}|"
            f"{rgb_path.resolve()}|{rgb_stat.st_size}|{rgb_stat.st_mtime_ns}"
        )
    return hashlib.sha256("\n".join(records).encode("utf-8")).hexdigest()


def is_possible_hsi_shape(shape: Sequence[int], hsi_channels: int) -> bool:
    """Return True when shape can represent a non-empty 3D HSI cube."""
    return (
        len(shape) == 3
        and hsi_channels in shape
        and all(int(dimension) > 0 for dimension in shape)
    )


def get_hsi_spatial_dims(shape: Tuple[int, ...], hsi_channels: int) -> Tuple[int, int]:
    dims = [dimension for axis, dimension in enumerate(shape)]
    if shape[0] == hsi_channels:
        return int(shape[1]), int(shape[2])
    return int(shape[0]), int(shape[1])


def inspect_hdf5_mat_file(
    file_path: Path,
    hsi_channels: int,
    hsi_key: str,
) -> Tuple[int, ...]:
    """Inspect a MATLAB v7.3/HDF5 file without loading its full HSI cube."""
    candidates: List[Tuple[str, Tuple[int, ...]]] = []

    try:
        with h5py.File(str(file_path), "r") as h5_file:
            if hsi_key in h5_file and isinstance(h5_file[hsi_key], h5py.Dataset):
                dataset = h5_file[hsi_key]
                candidates.append(
                    (hsi_key, tuple(int(value) for value in dataset.shape))
                )
            else:
                def visitor(name: str, obj: Any) -> None:
                    if not isinstance(obj, h5py.Dataset) or obj.ndim != 3:
                        return
                    try:
                        if np.issubdtype(obj.dtype, np.number):
                            candidates.append(
                                (name, tuple(int(value) for value in obj.shape))
                            )
                    except TypeError:
                        pass

                h5_file.visititems(visitor)
    except OSError as error:
        raise OSError(
            f"Could not inspect MATLAB v7.3 file:\n{file_path}\n"
            f"Reason: {error}"
        ) from error

    if not candidates:
        raise ValueError(f"No numerical 3D dataset found in {file_path}")

    valid_shapes = [shape for _, shape in candidates if is_possible_hsi_shape(shape, hsi_channels)]
    if not valid_shapes:
        raise ValueError(
            f"No {hsi_channels}-band cube found in {file_path}. "
            f"HDF5 datasets: {candidates}"
        )
    return valid_shapes[0]


def inspect_standard_mat_file(
    file_path: Path,
    hsi_channels: int,
    hsi_key: str,
) -> Tuple[int, ...]:
    """Inspect a pre-v7.3 MATLAB file using metadata only."""
    try:
        metadata = sio.whosmat(file_path)
    except (NotImplementedError, ValueError, OSError):
        # MATLAB v7.3 files are HDF5 containers.
        return inspect_hdf5_mat_file(file_path, hsi_channels, hsi_key)

    candidates = [
        (name, tuple(int(value) for value in shape))
        for name, shape, _ in metadata
        if len(shape) == 3
    ]

    if not candidates:
        raise ValueError(f"No 3D array found in {file_path}")

    # Prefer HSI_KEY when it exists. Otherwise accept any valid 3D cube.
    preferred = [candidate for candidate in candidates if candidate[0] == hsi_key]
    to_check = preferred if preferred else candidates

    valid_shapes = [shape for _, shape in to_check if is_possible_hsi_shape(shape, hsi_channels)]
    if not valid_shapes:
        raise ValueError(
            f"No {hsi_channels}-band cube found in {file_path}. "
            f"MAT arrays: {candidates}"
        )
    return valid_shapes[0]


def inspect_hsi_shape(
    file_path: Path,
    hsi_channels: int,
    hsi_key: str,
) -> Tuple[int, ...]:
    """Return the HSI cube's shape, using metadata only for .mat files."""
    if file_path.suffix.lower() == ".mat":
        return inspect_standard_mat_file(file_path, hsi_channels, hsi_key)

    cube = load_hsi_file(file_path, hsi_key)
    if not is_possible_hsi_shape(cube.shape, hsi_channels):
        raise ValueError(
            f"Invalid HSI shape {cube.shape} in {file_path}. "
            f"Expected a non-empty 3D cube containing {hsi_channels} bands."
        )
    return tuple(int(value) for value in cube.shape)


def inspect_pair(
    hsi_path: Path,
    rgb_path: Path,
    hsi_channels: int,
    hsi_key: str,
) -> None:
    """Validate an HSI/RGB pair before training.

    MATLAB files are checked from their headers/metadata so the complete cube
    is not loaded during the initial validation pass. RGB images are checked
    from their headers using PIL.

    Some HSI datasets (e.g. NTIRE / ARAD_1K) store the spectral cube with its
    height and width transposed relative to the RGB image's orientation. A
    pair is therefore accepted if the two spatial sizes match either directly
    or with H/W swapped; the actual transpose (if needed) is corrected for at
    load time in RGBHSIDataset.__getitem__ via align_hsi_orientation().
    """
    hsi_shape = inspect_hsi_shape(hsi_path, hsi_channels, hsi_key)
    hsi_h, hsi_w = get_hsi_spatial_dims(hsi_shape, hsi_channels)
    rgb_h, rgb_w = get_rgb_spatial_size(rgb_path)

    direct_match = (hsi_h, hsi_w) == (rgb_h, rgb_w)
    transposed_match = (hsi_h, hsi_w) == (rgb_w, rgb_h)

    if not (direct_match or transposed_match):
        raise ValueError(
            f"Spatial size mismatch between HSI {hsi_path} ({hsi_h}x{hsi_w}) "
            f"and RGB {rgb_path} ({rgb_h}x{rgb_w}) -- sizes don't match even "
            f"when transposed."
        )


def filter_valid_pairs(
    pairs: List[Tuple[Path, Path]],
    hsi_channels: int,
    hsi_key: str,
    log_path: Path,
) -> List[Tuple[Path, Path]]:
    """
    Validate HSI/RGB pairs, caching the result by path/size/mtime fingerprint,
    exactly like the HSI-only pipeline this is adapted from.
    """
    VALIDATION_CACHE.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    fingerprint = make_pairs_fingerprint(pairs)
    pair_lookup = {
        str(hsi_path.resolve()): (hsi_path, rgb_path) for hsi_path, rgb_path in pairs
    }

    if VALIDATION_CACHE.exists():
        try:
            try:
                cached = torch.load(VALIDATION_CACHE, map_location="cpu", weights_only=False)
            except TypeError:
                cached = torch.load(VALIDATION_CACHE, map_location="cpu")

            if (
                isinstance(cached, dict)
                and cached.get("fingerprint") == fingerprint
                and cached.get("checker_version") == HSI_CHECKER_VERSION
            ):
                valid_keys = cached.get("valid_hsi_paths", [])
                invalid_records = cached.get("invalid_records", [])
                valid_pairs = [pair_lookup[key] for key in valid_keys if key in pair_lookup]

                print("\nUsing cached HSI/RGB pair validation.")
                print(f"Valid pairs:   {len(valid_pairs)}")
                print(f"Invalid pairs: {len(invalid_records)}")

                for record in invalid_records:
                    print(
                        "\nCached invalid pair:\n"
                        f"  HSI:   {record['hsi_path']}\n"
                        f"  RGB:   {record['rgb_path']}\n"
                        f"  Error: {record['error']}"
                    )

                if valid_pairs:
                    return valid_pairs

        except Exception as error:
            print(f"Could not use validation cache. Re-scanning.\nReason: {error}")

    print("\nChecking HSI/RGB pair metadata before training...")

    valid_pairs: List[Tuple[Path, Path]] = []
    invalid_records: List[Dict[str, str]] = []

    for index, (hsi_path, rgb_path) in enumerate(pairs, start=1):
        try:
            inspect_pair(hsi_path, rgb_path, hsi_channels, hsi_key)
            valid_pairs.append((hsi_path, rgb_path))

        except Exception as error:
            invalid_records.append(
                {
                    "hsi_path": str(hsi_path.resolve()),
                    "rgb_path": str(rgb_path.resolve()),
                    "error": f"{type(error).__name__}: {error}",
                }
            )
            print(
                "\nSkipping invalid pair:\n"
                f"  HSI:   {hsi_path}\n"
                f"  RGB:   {rgb_path}\n"
                f"  Error: {error}"
            )

        if index % 100 == 0 or index == len(pairs):
            print(
                f"Checked {index}/{len(pairs)} | "
                f"Valid: {len(valid_pairs)} | "
                f"Invalid: {len(invalid_records)}"
            )

    if not valid_pairs:
        raise RuntimeError("No valid HSI/RGB pairs remain after validation.")

    if invalid_records:
        with log_path.open("w", encoding="utf-8") as handle:
            for record in invalid_records:
                handle.write(
                    f"{record['hsi_path']} | {record['rgb_path']} | {record['error']}\n"
                )
        print(f"\nInvalid-pair log saved to: {log_path}")
    elif log_path.exists():
        # Prevent a stale invalid-pair log from an older dataset scan.
        log_path.unlink()

    torch.save(
        {
            "checker_version": HSI_CHECKER_VERSION,
            "fingerprint": fingerprint,
            "valid_hsi_paths": [
                str(hsi_path.resolve()) for hsi_path, _ in valid_pairs
            ],
            "invalid_records": invalid_records,
        },
        VALIDATION_CACHE,
    )
    print(f"Validation cache saved to: {VALIDATION_CACHE}")

    return valid_pairs


# =============================================================================
# Native-resolution crops and augmentation (synced across RGB and HSI)
# =============================================================================


def pad_to_minimum_size(tensor: torch.Tensor, minimum_size: int) -> torch.Tensor:
    """Replicate-pad [C,H,W] only when an image is smaller than a train crop."""
    _, height, width = tensor.shape
    pad_bottom = max(0, minimum_size - height)
    pad_right = max(0, minimum_size - width)

    if pad_bottom == 0 and pad_right == 0:
        return tensor

    return F.pad(tensor, (0, pad_right, 0, pad_bottom), mode="replicate")


def synced_random_crop(
    rgb: torch.Tensor,
    hsi: torch.Tensor,
    crop_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Take the same random crop location from a paired RGB/HSI cube."""
    if rgb.shape[-2:] != hsi.shape[-2:]:
        raise ValueError(
            f"RGB spatial size {tuple(rgb.shape[-2:])} does not match "
            f"HSI spatial size {tuple(hsi.shape[-2:])}"
        )

    rgb = pad_to_minimum_size(rgb, crop_size)
    hsi = pad_to_minimum_size(hsi, crop_size)
    _, height, width = hsi.shape

    top = random.randint(0, height - crop_size)
    left = random.randint(0, width - crop_size)

    rgb_crop = rgb[:, top : top + crop_size, left : left + crop_size]
    hsi_crop = hsi[:, top : top + crop_size, left : left + crop_size]
    return rgb_crop, hsi_crop


def augment_paired(
    rgb: torch.Tensor,
    hsi: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply identical spatial transforms to the RGB image and every HSI band."""
    if random.random() < HORIZONTAL_FLIP_PROBABILITY:
        rgb = torch.flip(rgb, dims=(2,))
        hsi = torch.flip(hsi, dims=(2,))

    if random.random() < VERTICAL_FLIP_PROBABILITY:
        rgb = torch.flip(rgb, dims=(1,))
        hsi = torch.flip(hsi, dims=(1,))

    if USE_RANDOM_90_DEGREE_ROTATION:
        rotations = random.randint(0, 3)
        if rotations:
            rgb = torch.rot90(rgb, k=rotations, dims=(1, 2))
            hsi = torch.rot90(hsi, k=rotations, dims=(1, 2))

    return rgb.contiguous(), hsi.contiguous()


# =============================================================================
# Dataset and split
# =============================================================================


class RGBHSIDataset(Dataset):
    """Paired RGB/HSI dataset.

    Training:
        Load the complete native-resolution pair -> synced random crop ->
        synced augmentation.

    Validation:
        Load and return the complete native-resolution pair without any crop.
    """

    def __init__(
        self,
        pairs: List[Tuple[Path, Path]],
        hsi_channels: int,
        training: bool,
        normalization: str,
        crop_size: int | None = None,
        crops_per_image: int = 1,
        augment: bool = False,
    ) -> None:
        self.pairs = pairs
        self.hsi_channels = hsi_channels
        self.training = training
        self.normalization = normalization
        self.crop_size = crop_size
        self.crops_per_image = crops_per_image
        self.augment = augment

        if self.training and (self.crop_size is None or self.crop_size <= 0):
            raise ValueError("A positive crop_size is required for training.")
        if self.crops_per_image <= 0:
            raise ValueError("crops_per_image must be positive.")

    def __len__(self) -> int:
        if self.training:
            return len(self.pairs) * self.crops_per_image
        return len(self.pairs)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        pair_index = index // self.crops_per_image if self.training else index
        hsi_path, rgb_path = self.pairs[pair_index]

        cube = load_hsi_file(hsi_path, HSI_KEY)
        cube = convert_to_chw(cube, self.hsi_channels, hsi_path)

        if not np.isfinite(cube).all():
            raise ValueError(f"NaN or Inf values found in {hsi_path}")

        rgb_array = load_rgb_file(rgb_path)
        if not np.isfinite(rgb_array).all():
            raise ValueError(f"NaN or Inf values found in {rgb_path}")

        # Correct for HSI/RGB orientation mismatches (e.g. NTIRE/ARAD_1K
        # stores spectral cubes with H and W transposed relative to the RGB
        # image) before normalization, cropping, or augmentation.
        target_hw = (rgb_array.shape[1], rgb_array.shape[2])
        cube = align_hsi_orientation(cube, target_hw, hsi_path)

        cube = normalize_cube(cube, self.normalization)
        hsi_tensor = torch.from_numpy(np.ascontiguousarray(cube)).float()
        rgb_tensor = torch.from_numpy(np.ascontiguousarray(rgb_array)).float()

        if rgb_tensor.shape[-2:] != hsi_tensor.shape[-2:]:
            raise ValueError(
                f"Spatial size mismatch between {rgb_path} "
                f"{tuple(rgb_tensor.shape[-2:])} and {hsi_path} "
                f"{tuple(hsi_tensor.shape[-2:])}"
            )

        if self.training:
            rgb_tensor, hsi_tensor = synced_random_crop(
                rgb_tensor, hsi_tensor, int(self.crop_size)
            )
            if self.augment:
                rgb_tensor, hsi_tensor = augment_paired(rgb_tensor, hsi_tensor)

        return rgb_tensor, hsi_tensor


def split_pairs(
    pairs: List[Tuple[Path, Path]],
    validation_fraction: float,
    seed: int,
) -> Tuple[List[Tuple[Path, Path]], List[Tuple[Path, Path]]]:
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("VALIDATION_FRACTION must be between zero and one.")

    shuffled = pairs.copy()
    random.Random(seed).shuffle(shuffled)

    validation_size = max(1, int(round(len(shuffled) * validation_fraction)))
    validation_pairs = shuffled[:validation_size]
    training_pairs = shuffled[validation_size:]

    if not training_pairs:
        raise RuntimeError("No pairs remain for training after the split.")

    return training_pairs, validation_pairs


# =============================================================================
# Model forward helpers
# =============================================================================


def pad_to_multiple(
    tensor: torch.Tensor,
    multiple: int,
) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """Pad [B,C,H,W] on the bottom/right and return the original H,W."""
    if tensor.ndim != 4:
        raise ValueError(f"Expected [B,C,H,W], found shape {tuple(tensor.shape)}")
    if multiple <= 0:
        raise ValueError("multiple must be positive")

    original_height, original_width = tensor.shape[-2:]
    padded_height = ((original_height + multiple - 1) // multiple) * multiple
    padded_width = ((original_width + multiple - 1) // multiple) * multiple

    pad_bottom = padded_height - original_height
    pad_right = padded_width - original_width

    if pad_bottom == 0 and pad_right == 0:
        return tensor, (original_height, original_width)

    padded = F.pad(tensor, (0, pad_right, 0, pad_bottom), mode="replicate")
    return padded, (original_height, original_width)


def pixel_loss(reconstruction: torch.Tensor, target: torch.Tensor, loss_type: str) -> torch.Tensor:
    if loss_type == "mse":
        return F.mse_loss(reconstruction, target)
    if loss_type == "l1":
        return F.l1_loss(reconstruction, target)
    if loss_type == "smooth_l1":
        return F.smooth_l1_loss(reconstruction, target)
    raise ValueError(f"Unknown reconstruction loss: {loss_type}")


def get_sam_weight(epoch: int) -> float:
    if SAM_WARMUP_EPOCHS <= 0:
        return SAM_LOSS_WEIGHT_END

    progress = min(max((epoch - 1) / SAM_WARMUP_EPOCHS, 0.0), 1.0)
    return SAM_LOSS_WEIGHT_START + progress * (SAM_LOSS_WEIGHT_END - SAM_LOSS_WEIGHT_START)


def diffusion_training_step(
    model: ResShiftSSR,
    rgb: torch.Tensor,
    hsi_gt: torch.Tensor,
    loss_type: str,
    sam_weight: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run one ResShift training step and return (loss, pred_x0, y0), all
    cropped back to the input's original spatial size.

    This mirrors ResShiftSSR.forward(), but is written out explicitly so a
    configurable pixel-loss type and a spectral-angle regulariser can be
    applied to the denoiser's predicted x0, instead of the hard-coded MSE
    used inside the model file's own p_losses().
    """
    padded_rgb, (height, width) = pad_to_multiple(rgb, MODEL_DOWNSAMPLE_FACTOR)
    padded_hsi, _ = pad_to_multiple(hsi_gt, MODEL_DOWNSAMPLE_FACTOR)

    with torch.set_grad_enabled(not model.freeze_coarse):
        y0_padded = model.coarse_net(padded_rgb)

    batch_size = hsi_gt.shape[0]
    t = torch.randint(
        0, model.diffusion.T, (batch_size,), device=hsi_gt.device, dtype=torch.long
    )
    noise = torch.randn_like(padded_hsi)
    x_t = model.diffusion.q_sample(padded_hsi, y0_padded, t, noise=noise)
    pred_x0_padded = model.denoiser(x_t, y0_padded, t)

    pred_x0 = pred_x0_padded[..., :height, :width]
    y0 = y0_padded[..., :height, :width]

    loss = pixel_loss(pred_x0, hsi_gt, loss_type)
    if sam_weight > 0:
        loss = loss + sam_weight * sam(hsi_gt, pred_x0)

    return loss, pred_x0.detach(), y0.detach()


@torch.no_grad()
def diffusion_full_sample(
    model: ResShiftSSR,
    rgb: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Run the full reverse-diffusion sampling loop and crop back to size."""
    padded_rgb, (height, width) = pad_to_multiple(rgb, MODEL_DOWNSAMPLE_FACTOR)
    fine_hsi_padded, y0_padded = model.sample(padded_rgb)
    fine_hsi = fine_hsi_padded[..., :height, :width]
    y0 = y0_padded[..., :height, :width]
    return fine_hsi, y0


# =============================================================================
# Metrics
# =============================================================================


@torch.no_grad()
def calculate_metrics(
    reconstruction: torch.Tensor,
    target: torch.Tensor,
) -> Dict[str, float]:
    reconstruction = reconstruction.float()
    target = target.float()

    return {
        "mrae": float(mrae(target, reconstruction).item()),
        "rmse": float(rmse(target, reconstruction).item()),
        "sam": float(sam(target, reconstruction).item()),
        "psnr": float(psnr(target, reconstruction).item()),
        "ssim": float(ssim(target, reconstruction).item()),
    }


def _empty_metric_totals() -> Dict[str, float]:
    return {"mrae": 0.0, "rmse": 0.0, "sam": 0.0, "psnr": 0.0, "ssim": 0.0}


# =============================================================================
# Training and validation
# =============================================================================


def train_one_epoch(
    model: ResShiftSSR,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
    use_amp: bool,
    loss_type: str,
    sam_weight: float,
) -> Dict[str, float]:
    model.train()

    totals = {"total_loss": 0.0, **_empty_metric_totals()}
    sample_count = 0

    for batch_index, (rgb, hsi) in enumerate(loader, start=1):
        rgb = rgb.to(device, non_blocking=True)
        hsi = hsi.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=use_amp):
            loss, pred_x0, _y0 = diffusion_training_step(
                model, rgb, hsi, loss_type, sam_weight
            )

        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite training loss: {loss.item()}")

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], GRADIENT_CLIP_NORM
        )
        scaler.step(optimizer)
        scaler.update()

        # pred_x0 is the denoiser's single-step x0 estimate at a random
        # timestep, not the fully sampled output. It is a fast, noisy but
        # useful proxy for tracking whether the denoiser is learning.
        metrics = calculate_metrics(pred_x0, hsi.detach())
        batch_size = hsi.size(0)
        sample_count += batch_size

        totals["total_loss"] += float(loss.detach().item()) * batch_size
        for name, value in metrics.items():
            totals[name] += value * batch_size

        if batch_index % PRINT_EVERY == 0 or batch_index == len(loader):
            denominator = max(sample_count, 1)
            print(
                f"  Batch {batch_index:04d}/{len(loader):04d} | "
                f"Loss: {totals['total_loss'] / denominator:.6f} | "
                f"MRAE: {totals['mrae'] / denominator:.6f} | "
                f"PSNR: {totals['psnr'] / denominator:.4f}"
            )

    return {name: value / max(sample_count, 1) for name, value in totals.items()}


@torch.no_grad()
def validate(
    model: ResShiftSSR,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool,
    loss_type: str,
    sam_weight: float,
    run_full_sampling: bool,
) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float]]:
    """Validate on full images.

    Returns
    -------
    quick_metrics:
        Diffusion training loss and single-step x0 metrics (cheap, always
        computed, comparable across every epoch).
    coarse_metrics:
        mrae/rmse/sam/psnr/ssim of the frozen coarse network's output y0
        alone, for reference.
    full_sample_metrics:
        mrae/rmse/sam/psnr/ssim of the fully diffusion-refined output,
        obtained by running the complete T-step reverse sampling loop. Only
        populated when run_full_sampling is True; otherwise returned empty.
    """
    model.eval()

    quick_totals = {"total_loss": 0.0, **_empty_metric_totals()}
    coarse_totals = _empty_metric_totals()
    full_totals = _empty_metric_totals()
    image_count = 0

    for rgb, hsi in loader:
        # VALIDATION_BATCH_SIZE is one, so each iteration can have a different
        # full-resolution H x W shape.
        rgb = rgb.to(device, non_blocking=True)
        hsi = hsi.to(device, non_blocking=True)

        with autocast(enabled=use_amp):
            loss, pred_x0, y0 = diffusion_training_step(
                model, rgb, hsi, loss_type, sam_weight
            )

        if not torch.isfinite(loss):
            raise FloatingPointError(
                f"Non-finite validation loss for shape {tuple(hsi.shape)}"
            )

        image_count += 1
        quick_totals["total_loss"] += float(loss.item())
        for name, value in calculate_metrics(pred_x0, hsi).items():
            quick_totals[name] += value
        for name, value in calculate_metrics(y0, hsi).items():
            coarse_totals[name] += value

        if run_full_sampling:
            fine_hsi, _y0_full = diffusion_full_sample(model, rgb)
            for name, value in calculate_metrics(fine_hsi, hsi).items():
                full_totals[name] += value

    quick_metrics = {
        name: value / max(image_count, 1) for name, value in quick_totals.items()
    }
    coarse_metrics = {
        name: value / max(image_count, 1) for name, value in coarse_totals.items()
    }
    full_sample_metrics = (
        {name: value / max(image_count, 1) for name, value in full_totals.items()}
        if run_full_sampling
        else {}
    )

    return quick_metrics, coarse_metrics, full_sample_metrics


# =============================================================================
# Checkpointing
# =============================================================================


def save_checkpoint(
    output_path: Path,
    model: ResShiftSSR,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: GradScaler,
    epoch: int,
    sam_weight: float,
    quick_metrics: Dict[str, float],
    coarse_metrics: Dict[str, float],
    full_sample_metrics: Dict[str, float],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "lr_scheduler_state_dict": lr_scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "sam_loss_weight": sam_weight,
            "quick_validation_metrics": quick_metrics,
            "coarse_network_validation_metrics": coarse_metrics,
            "full_sample_validation_metrics": full_sample_metrics,
            "model_config": {
                "hsi_channels": HSI_CHANNELS,
                "diffusion_T": DIFFUSION_T,
                "diffusion_p": DIFFUSION_P,
                "diffusion_kappa": DIFFUSION_KAPPA,
                "base_dim": BASE_DIM,
                "dim_mults": DIM_MULTS,
                "num_res_blocks": NUM_RES_BLOCKS,
                "freeze_coarse": FREEZE_COARSE,
            },
            "training_config": {
                "normalization": NORMALIZATION,
                "train_crop_size": TRAIN_CROP_SIZE,
                "model_downsample_factor": MODEL_DOWNSAMPLE_FACTOR,
                "diffusion_loss_type": DIFFUSION_LOSS_TYPE,
                "sam_loss_weight_start": SAM_LOSS_WEIGHT_START,
                "sam_loss_weight_end": SAM_LOSS_WEIGHT_END,
                "sam_warmup_epochs": SAM_WARMUP_EPOCHS,
                "use_augmentation": USE_AUGMENTATION,
            },
        },
        output_path,
    )


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    set_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = USE_AMP and device.type == "cuda"

    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    if TRAIN_CROP_SIZE % MODEL_DOWNSAMPLE_FACTOR != 0:
        raise ValueError(
            f"TRAIN_CROP_SIZE ({TRAIN_CROP_SIZE}) must be divisible by "
            f"MODEL_DOWNSAMPLE_FACTOR ({MODEL_DOWNSAMPLE_FACTOR})."
        )

    all_pairs = find_paired_files(HSI_DATA_DIR, RGB_DATA_DIR)
    print("\nHSI/RGB pair validation mode: metadata inspection + fingerprint cache")
    all_pairs = filter_valid_pairs(
        pairs=all_pairs,
        hsi_channels=HSI_CHANNELS,
        hsi_key=HSI_KEY,
        log_path=output_dir / "invalid_pairs.txt",
    )
    training_pairs, validation_pairs = split_pairs(all_pairs, VALIDATION_FRACTION, SEED)

    print(f"\nDevice: {device}")
    print(f"Mixed precision: {use_amp}")
    print(f"Total HSI/RGB pairs: {len(all_pairs)}")
    print(f"Validation cache: {VALIDATION_CACHE}")
    print(f"Invalid-pair log: {output_dir / 'invalid_pairs.txt'}")
    print(f"Training pairs: {len(training_pairs)}")
    print(f"Validation pairs: {len(validation_pairs)}")
    print(f"Training crop size: {TRAIN_CROP_SIZE} x {TRAIN_CROP_SIZE}")
    print("Validation mode: complete native-resolution image pairs")
    if USE_AUGMENTATION:
        print(
            "Training augmentation: augment_paired() is called for every "
            "training crop after the synced random crop. Each flip/rotation "
            "is stochastic, so an individual crop can still receive an "
            "identity transform."
        )
    else:
        print("Training augmentation: disabled")

    training_dataset = RGBHSIDataset(
        pairs=training_pairs,
        hsi_channels=HSI_CHANNELS,
        training=True,
        normalization=NORMALIZATION,
        crop_size=TRAIN_CROP_SIZE,
        crops_per_image=CROPS_PER_IMAGE,
        augment=USE_AUGMENTATION,
    )
    validation_dataset = RGBHSIDataset(
        pairs=validation_pairs,
        hsi_channels=HSI_CHANNELS,
        training=False,
        normalization=NORMALIZATION,
        crop_size=None,
        crops_per_image=1,
        augment=False,
    )

    generator = torch.Generator()
    generator.manual_seed(SEED)

    training_loader = DataLoader(
        training_dataset,
        batch_size=TRAIN_BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=device.type == "cuda",
        drop_last=False,
        persistent_workers=NUM_WORKERS > 0,
        worker_init_fn=seed_worker,
        generator=generator,
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=VALIDATION_BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=device.type == "cuda",
        drop_last=False,
        persistent_workers=NUM_WORKERS > 0,
        worker_init_fn=seed_worker,
    )

    model = ResShiftSSR(
        mst_ckpt_path=MST_CKPT_PATH,
        channels=HSI_CHANNELS,
        T=DIFFUSION_T,
        p=DIFFUSION_P,
        kappa=DIFFUSION_KAPPA,
        base_dim=BASE_DIM,
        dim_mults=DIM_MULTS,
        num_res_blocks=NUM_RES_BLOCKS,
        freeze_coarse=FREEZE_COARSE,
    ).to(device)

    trainable_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    print(
        f"Trainable parameters (denoiser"
        f"{', coarse net unfrozen' if not FREEZE_COARSE else ''}): "
        f"{sum(parameter.numel() for parameter in trainable_parameters):,}"
    )

    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=NUM_EPOCHS,
        eta_min=1e-7,
    )
    scaler = GradScaler(enabled=use_amp)

    best_validation_mrae = float("inf")

    for epoch in range(1, NUM_EPOCHS + 1):
        sam_weight = get_sam_weight(epoch)
        is_last_epoch = epoch == NUM_EPOCHS

        print(
            f"\nEpoch {epoch:03d}/{NUM_EPOCHS:03d} | "
            f"SAM loss weight: {sam_weight:.8f}"
        )

        training_metrics = train_one_epoch(
            model=model,
            loader=training_loader,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            use_amp=use_amp,
            loss_type=DIFFUSION_LOSS_TYPE,
            sam_weight=sam_weight,
        )

        # Full T-step reverse-diffusion sampling only runs after the final
        # epoch, since it is far more expensive than the single-step
        # quick_metrics computed every epoch below.
        quick_metrics, coarse_metrics, full_sample_metrics = validate(
            model=model,
            loader=validation_loader,
            device=device,
            use_amp=use_amp,
            loss_type=DIFFUSION_LOSS_TYPE,
            sam_weight=sam_weight,
            run_full_sampling=is_last_epoch,
        )

        lr_scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"\nEpoch {epoch:03d}/{NUM_EPOCHS:03d} | "
            f"LR: {current_lr:.2e} | SAM weight: {sam_weight:.8f}\n"
            f"  Train loss (single-step x0): {training_metrics['total_loss']:.6f} | "
            f"Train MRAE: {training_metrics['mrae']:.6f} | "
            f"Train PSNR: {training_metrics['psnr']:.4f}\n"
            f"  Val loss (single-step x0): {quick_metrics['total_loss']:.6f} | "
            f"Val MRAE: {quick_metrics['mrae']:.6f} | "
            f"Val RMSE: {quick_metrics['rmse']:.6f} | "
            f"Val SAM: {quick_metrics['sam']:.6f} | "
            f"Val PSNR: {quick_metrics['psnr']:.4f} | "
            f"Val SSIM: {quick_metrics['ssim']:.4f}\n"
            f"  Coarse net only | MRAE: {coarse_metrics['mrae']:.6f} | "
            f"RMSE: {coarse_metrics['rmse']:.6f} | "
            f"SAM: {coarse_metrics['sam']:.6f} | "
            f"PSNR: {coarse_metrics['psnr']:.4f} | "
            f"SSIM: {coarse_metrics['ssim']:.4f}"
        )

        if is_last_epoch:
            print(
                f"  Full diffusion sample (final epoch) | "
                f"MRAE: {full_sample_metrics['mrae']:.6f} | "
                f"RMSE: {full_sample_metrics['rmse']:.6f} | "
                f"SAM: {full_sample_metrics['sam']:.6f} | "
                f"PSNR: {full_sample_metrics['psnr']:.4f} | "
                f"SSIM: {full_sample_metrics['ssim']:.4f}"
            )

        save_checkpoint(
            output_path=output_dir / "last_resshift.pth",
            model=model,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            scaler=scaler,
            epoch=epoch,
            sam_weight=sam_weight,
            quick_metrics=quick_metrics,
            coarse_metrics=coarse_metrics,
            full_sample_metrics=full_sample_metrics,
        )

        # Best checkpoint is tracked using the cheap single-step MRAE, since
        # that is the only quality signal available every epoch now that
        # full sampling only happens at the very end.
        if quick_metrics["mrae"] < best_validation_mrae:
            best_validation_mrae = quick_metrics["mrae"]
            save_checkpoint(
                output_path=output_dir / "best_resshift.pth",
                model=model,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                scaler=scaler,
                epoch=epoch,
                sam_weight=sam_weight,
                quick_metrics=quick_metrics,
                coarse_metrics=coarse_metrics,
                full_sample_metrics=full_sample_metrics,
            )
            print(
                "  New best checkpoint: "
                f"single-step validation MRAE = {best_validation_mrae:.6f}"
            )

        if is_last_epoch:
            save_checkpoint(
                output_path=output_dir / "final_resshift_with_full_sampling.pth",
                model=model,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                scaler=scaler,
                epoch=epoch,
                sam_weight=sam_weight,
                quick_metrics=quick_metrics,
                coarse_metrics=coarse_metrics,
                full_sample_metrics=full_sample_metrics,
            )
            print(
                "\nTraining complete. Final full-sampling validation metrics:\n"
                f"  MRAE: {full_sample_metrics['mrae']:.6f} | "
                f"RMSE: {full_sample_metrics['rmse']:.6f} | "
                f"SAM: {full_sample_metrics['sam']:.6f} | "
                f"PSNR: {full_sample_metrics['psnr']:.4f} | "
                f"SSIM: {full_sample_metrics['ssim']:.4f}\n"
                f"Saved to: {output_dir / 'final_resshift_with_full_sampling.pth'}"
            )


if __name__ == "__main__":
    main()
