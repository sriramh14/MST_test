"""Train a Residual Diffusion (RGB -> HSI) model from paired RGB/HSI images.

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
   height and width are compatible with the residual U-Net's downsampling
   factor. Every prediction is cropped back to the original size before
   losses and metrics are calculated.
7. This model is a conditional residual diffusion model adapted from DiffUIR
   (CVPR 2024). A frozen MST++ network produces a coarse hyperspectral
   prediction from the RGB input; a DiffUIR-style diffusion model then
   learns only the residual between that coarse prediction and the ground
   truth HSI cube (never the full HSI directly). The training objective is
   therefore:
       loss = pixel_loss(predicted_residual, target_residual)
              + sam_weight(epoch) * spectral_angle_loss(hsi_gt, reconstruction)
   where `predicted_residual` is the model's estimate of
   (GroundTruthHSI - MSTPrediction) at a randomly sampled diffusion
   timestep t (DiffUIR's own training objective, Eq. 11 of the paper), and
   `reconstruction = MSTPrediction + predicted_residual`. The spectral-angle
   term is included (with its own warm-up schedule, replacing the old
   KL-beta warm-up from the original VAE-style script this is adapted from)
   because it is a well known, differentiable, HSI-specific regulariser that
   discourages spectral distortion introduced by pixel-wise losses alone. It
   is applied to the reconstruction rather than to the residual itself,
   since spectral angle is only meaningful on absolute spectral vectors.
8. Regardless of the training loss, mrae / rmse / sam / psnr / ssim are always
   computed every epoch for the coarse network's prediction (y0) and for the
   diffusion model's cheap single-step reconstruction estimate (pred_x0 =
   y0 + predicted_residual at a random timestep). The fully diffusion-refined
   output (multi-step DDIM/DDPM reverse sampling) is only run ONCE, after the
   final training epoch, since it is far more expensive; its metrics are
   saved in the final checkpoint and printed at the end of the run.

Compatibility note
-------------------
This script was checked against the corrected `ResidualDiffusionRGB2HSI`
model file (which fixes a skip-connection channel-bookkeeping bug in
`ResidualUNet` and replaces every `GroupNorm` with a `LayerNorm2d`). Neither
fix changes the model's public interface: `model(rgb, gt_hsi)` still returns
a dict with `mst_prediction` / `predicted_residual` / `target_residual`;
`model.sample(...)` and `model.get_coarse_prediction(...)` are unchanged;
`model.mst_plus_plus` and `model._freeze_mst_plus_plus()` are unchanged. No
changes were required in this file for compatibility with the corrected
model.
"""

from __future__ import annotations

import argparse
import hashlib
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import h5py
import numpy as np
import scipy.io as sio
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset

# TODO: adjust this import to wherever you saved the residual diffusion model
# file (the file that defines ResidualDiffusionRGB2HSI, ResidualDiffusionScheduler
# and ResidualUNet).
from model.DiffUIR_res_pred import ResidualDiffusionRGB2HSI

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
OUTPUT_DIR = "./residual_diffusion_checkpoints"

HSI_KEY = "cube"
HSI_CHANNELS = 31
RGB_CHANNELS = 3
SUPPORTED_HSI_EXTENSIONS = {".mat", ".npy", ".npz", ".pt", ".pth"}
SUPPORTED_RGB_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

# Pretrained MST++ checkpoint. The model file assumes MST++ weights are
# loaded externally (it only instantiates and freezes the network), so this
# training script loads the checkpoint into `model.mst_plus_plus` itself,
# right after the model is constructed. MST++ is frozen inside the model
# implementation itself (all parameters have requires_grad=False and it is
# always run under torch.no_grad()) -- no additional freezing logic is
# required here.
MST_CKPT_PATH = None

# Diffusion / residual U-Net architecture. These values must match
# model.residual_diffusion_rgb2hsi.ResidualDiffusionRGB2HSI.
DIFFUSION_T = 1000
DIFFUSION_ALPHA_BAR_MAX = 1.0
DIFFUSION_BETA_BAR_MIN = 1e-4
DIFFUSION_BETA_BAR_MAX = 1.0
DIFFUSION_DELTA_BAR_MAX = 0.9
DIFFUSION_SCHEDULE_TYPE = "linear"
UNET_BASE_CHANNELS = 64
UNET_CHANNEL_MULTIPLIERS = (1, 2, 4)
UNET_NUM_RES_BLOCKS = 2
UNET_ATTENTION_RESOLUTIONS = (16,)
UNET_DROPOUT = 0.0

# Fast DDIM sampling for the (expensive, end-of-training-only) full reverse
# diffusion pass. DiffUIR itself uses as few as 3 timesteps at inference.
NUM_SAMPLING_STEPS = 3
USE_DDIM_SAMPLING = True

# The residual U-Net downsamples (len(UNET_CHANNEL_MULTIPLIERS) - 1) times by
# a factor of 2 each time. Padding must make H and W divisible by this factor.
MODEL_DOWNSAMPLE_FACTOR = 2 ** (len(UNET_CHANNEL_MULTIPLIERS) - 1)

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
# always scaled to [0, 1] since that is what the frozen MST++ network
# expects.
NORMALIZATION = "none"

VALIDATION_FRACTION = 0.10
SEED = 42

TRAIN_BATCH_SIZE = 4
# Keep this at one because validation images can have different H x W shapes.
VALIDATION_BATCH_SIZE = 1
NUM_WORKERS = 4

NUM_EPOCHS = 50
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
GRADIENT_CLIP_NORM = 1.0
USE_AMP = True
PRINT_EVERY = 30

# Pixel loss used to regress the model's predicted residual towards the
# target residual (GroundTruthHSI - MSTPrediction). "mse", "l1", or
# "smooth_l1". DiffUIR itself uses an L1 objective (Eq. 11).
DIFFUSION_LOSS_TYPE = "l1"

# Spectral-angle regularisation on the reconstructed HSI (mst_prediction +
# predicted_residual), ramped up over training (mirrors the old KL-beta
# warm-up, but there is no latent code here).
SAM_LOSS_WEIGHT_START = 0.0
SAM_LOSS_WEIGHT_END = 0.05
SAM_WARMUP_EPOCHS = 30

# Full reverse-diffusion sampling (multi-step DDIM/DDPM through the residual
# U-Net, per validation image) is expensive, so it is only run once, after
# the final training epoch, to report the model's true inference-time
# mrae/rmse/sam/psnr/ssim. Every other epoch only computes the cheap
# single-step reconstruction estimate (quick_metrics) for monitoring
# convergence.

# The best checkpoint is selected using the cheap single-step validation
# MRAE (quick_metrics), since that is available every epoch. A separate,
# final checkpoint with true full-sampling metrics is saved after the last
# epoch.
VALIDATION_CACHE = Path(OUTPUT_DIR) / "resshift_validation_cache.pth"
HSI_CHECKER_VERSION = "resshift-rgb-hsi-pair-cache-v1"

# =============================================================================
# Inference configuration
# =============================================================================

# Defaults used by `python this_script.py infer ...` when the corresponding
# CLI flag is not supplied. All of these can be overridden on the command
# line; see `build_arg_parser()` / `run_inference_cli()` below.
INFER_CHECKPOINT_PATH = str(Path(OUTPUT_DIR) / "best_residual_diffusion.pth")
INFER_INPUT_DIR = RGB_DATA_DIR
INFER_OUTPUT_DIR = "./residual_diffusion_predictions"
INFER_SAVE_FORMAT = "mat"  # "mat" or "npy"
INFER_SAVE_COARSE = False  # also save the frozen MST++ prediction (y0)
INFER_GT_HSI_DIR: Optional[str] = None  # optional, for reporting metrics only


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
# RGB loading (paired input for the frozen MST++ network)
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
    model: ResidualDiffusionRGB2HSI,
    rgb: torch.Tensor,
    hsi_gt: torch.Tensor,
    loss_type: str,
    sam_weight: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run one residual-diffusion training step and return
    (loss, pred_x0, y0), all cropped back to the input's original spatial
    size.

    This calls model.forward() directly and uses the quantities it returns
    (mst_prediction, predicted_residual, target_residual) rather than
    recomputing them, then applies a configurable pixel-loss type to the
    residual and an optional spectral-angle regulariser to the
    reconstruction, mirroring how the base script applied a configurable
    pixel loss plus a spectral-angle regulariser to its denoiser's predicted
    x0.

    `y0` here is the frozen MST++ coarse prediction (mst_prediction).
    `pred_x0` here is the cheap single-timestep reconstruction estimate
    (mst_prediction + predicted_residual) -- the residual-diffusion
    equivalent of the base script's single-step x0 estimate.
    """
    padded_rgb, (height, width) = pad_to_multiple(rgb, MODEL_DOWNSAMPLE_FACTOR)
    padded_hsi, _ = pad_to_multiple(hsi_gt, MODEL_DOWNSAMPLE_FACTOR)

    # MST++ freezing and the no_grad forward pass are both handled inside
    # the model itself; no external freezing logic is required here.
    outputs = model(padded_rgb, padded_hsi)

    mst_prediction_padded = outputs["mst_prediction"]
    predicted_residual_padded = outputs["predicted_residual"]
    target_residual_padded = outputs["target_residual"]

    y0 = mst_prediction_padded[..., :height, :width]
    predicted_residual = predicted_residual_padded[..., :height, :width]
    target_residual = target_residual_padded[..., :height, :width]

    pred_x0 = y0 + predicted_residual

    # Primary objective: DiffUIR's residual regression loss (Eq. 11 of the
    # paper), applied to the predicted residual rather than to reconstructed
    # pixel values.
    loss = pixel_loss(predicted_residual, target_residual, loss_type)
    if sam_weight > 0:
        # The spectral-angle regulariser is computed on the reconstruction
        # (absolute spectral vectors), since spectral angle is not a
        # meaningful quantity on a residual difference alone.
        loss = loss + sam_weight * sam(hsi_gt, pred_x0)

    return loss, pred_x0.detach(), y0.detach()


@torch.no_grad()
def diffusion_full_sample(
    model: ResidualDiffusionRGB2HSI,
    rgb: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Run the full reverse-diffusion sampling loop and crop back to size."""
    padded_rgb, (height, width) = pad_to_multiple(rgb, MODEL_DOWNSAMPLE_FACTOR)

    fine_hsi_padded = model.sample(
        padded_rgb,
        num_sampling_steps=NUM_SAMPLING_STEPS,
        use_ddim=USE_DDIM_SAMPLING,
    )
    y0_padded = model.get_coarse_prediction(padded_rgb)

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
    model: ResidualDiffusionRGB2HSI,
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

        # pred_x0 is the residual-diffusion model's single-step reconstruction
        # estimate at a random timestep, not the fully sampled output. It is
        # a fast, noisy but useful proxy for tracking whether the residual
        # U-Net is learning.
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
    model: ResidualDiffusionRGB2HSI,
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
        Diffusion training loss and single-step reconstruction metrics
        (cheap, always computed, comparable across every epoch).
    coarse_metrics:
        mrae/rmse/sam/psnr/ssim of the frozen MST++ network's output y0
        alone, for reference.
    full_sample_metrics:
        mrae/rmse/sam/psnr/ssim of the fully diffusion-refined output,
        obtained by running the complete multi-step reverse sampling loop.
        Only populated when run_full_sampling is True; otherwise returned
        empty.
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
    model: ResidualDiffusionRGB2HSI,
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
                "rgb_channels": RGB_CHANNELS,
                "hsi_channels": HSI_CHANNELS,
                "num_timesteps": DIFFUSION_T,
                "unet_base_channels": UNET_BASE_CHANNELS,
                "unet_channel_multipliers": UNET_CHANNEL_MULTIPLIERS,
                "unet_num_res_blocks": UNET_NUM_RES_BLOCKS,
                "unet_attention_resolutions": UNET_ATTENTION_RESOLUTIONS,
                "unet_dropout": UNET_DROPOUT,
                "diffusion_alpha_bar_max": DIFFUSION_ALPHA_BAR_MAX,
                "diffusion_beta_bar_min": DIFFUSION_BETA_BAR_MIN,
                "diffusion_beta_bar_max": DIFFUSION_BETA_BAR_MAX,
                "diffusion_delta_bar_max": DIFFUSION_DELTA_BAR_MAX,
                "diffusion_schedule_type": DIFFUSION_SCHEDULE_TYPE,
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
                "num_sampling_steps": NUM_SAMPLING_STEPS,
                "use_ddim_sampling": USE_DDIM_SAMPLING,
            },
        },
        output_path,
    )


# =============================================================================
# Inference
# =============================================================================


def build_model_from_checkpoint(
    checkpoint: Dict[str, Any],
    device: torch.device,
) -> ResidualDiffusionRGB2HSI:
    """Reconstruct a ResidualDiffusionRGB2HSI from a saved checkpoint.

    Falls back to the module-level architecture constants above for any
    field that an older checkpoint (saved before this key existed) might be
    missing, so checkpoints from earlier versions of this script still load.
    """
    config = checkpoint.get("model_config", {})

    model = ResidualDiffusionRGB2HSI(
        rgb_channels=config.get("rgb_channels", RGB_CHANNELS),
        hsi_channels=config.get("hsi_channels", HSI_CHANNELS),
        num_timesteps=config.get("num_timesteps", DIFFUSION_T),
        unet_base_channels=config.get("unet_base_channels", UNET_BASE_CHANNELS),
        unet_channel_multipliers=tuple(
            config.get("unet_channel_multipliers", UNET_CHANNEL_MULTIPLIERS)
        ),
        unet_num_res_blocks=config.get("unet_num_res_blocks", UNET_NUM_RES_BLOCKS),
        unet_attention_resolutions=tuple(
            config.get("unet_attention_resolutions", UNET_ATTENTION_RESOLUTIONS)
        ),
        unet_dropout=config.get("unet_dropout", UNET_DROPOUT),
        alpha_bar_max=config.get("diffusion_alpha_bar_max", DIFFUSION_ALPHA_BAR_MAX),
        beta_bar_min=config.get("diffusion_beta_bar_min", DIFFUSION_BETA_BAR_MIN),
        beta_bar_max=config.get("diffusion_beta_bar_max", DIFFUSION_BETA_BAR_MAX),
        delta_bar_max=config.get("diffusion_delta_bar_max", DIFFUSION_DELTA_BAR_MAX),
        schedule_type=config.get("diffusion_schedule_type", DIFFUSION_SCHEDULE_TYPE),
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    # MST++ is already frozen by the model's own constructor; this is just a
    # safety re-assertion, mirroring main()'s behaviour after weight loading.
    model._freeze_mst_plus_plus()
    model.eval()
    return model


def load_model_for_inference(
    checkpoint_path: str,
    device: torch.device,
) -> Tuple[ResidualDiffusionRGB2HSI, Dict[str, Any]]:
    checkpoint_file = Path(checkpoint_path)
    if not checkpoint_file.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_file}")

    try:
        checkpoint = torch.load(checkpoint_file, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_file, map_location=device)

    model = build_model_from_checkpoint(checkpoint, device)

    print(f"Loaded checkpoint: {checkpoint_file}")
    print(f"  Epoch: {checkpoint.get('epoch', 'unknown')}")
    quick = checkpoint.get("quick_validation_metrics")
    if quick:
        print(f"  Checkpoint's single-step validation MRAE: {quick.get('mrae', float('nan')):.6f}")

    return model, checkpoint


def save_hsi_cube(cube_chw: np.ndarray, output_path: Path, save_format: str) -> None:
    """Save an [C,H,W] HSI cube to disk as either .mat (key 'cube') or .npy."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if save_format == "mat":
        sio.savemat(str(output_path), {HSI_KEY: cube_chw})
    elif save_format == "npy":
        np.save(str(output_path), cube_chw)
    else:
        raise ValueError(f"Unknown save format: {save_format}")


def discover_inference_inputs(input_path: str) -> List[Path]:
    """Accept either a single RGB image file or a directory of RGB images."""
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"Inference input not found: {path}")

    if path.is_file():
        if path.suffix.lower() not in SUPPORTED_RGB_EXTENSIONS:
            raise ValueError(
                f"{path} does not have a supported RGB extension "
                f"({sorted(SUPPORTED_RGB_EXTENSIONS)})"
            )
        return [path]

    return find_files_with_extensions(input_path, SUPPORTED_RGB_EXTENSIONS)


@torch.no_grad()
def run_inference(
    model: ResidualDiffusionRGB2HSI,
    input_path: str,
    output_dir: str,
    device: torch.device,
    use_amp: bool,
    save_format: str = INFER_SAVE_FORMAT,
    save_coarse: bool = INFER_SAVE_COARSE,
    gt_hsi_dir: Optional[str] = INFER_GT_HSI_DIR,
    num_sampling_steps: int = NUM_SAMPLING_STEPS,
    use_ddim_sampling: bool = USE_DDIM_SAMPLING,
) -> Dict[str, float]:
    """Run full reverse-diffusion inference on one RGB image or a directory.

    For every RGB input this predicts the fine HSI cube with
    `diffusion_full_sample()` (the same multi-step DDIM/DDPM reverse
    sampling used once at the end of training) and saves it next to the
    coarse MST++ prediction if `save_coarse` is set.

    If `gt_hsi_dir` is given, ground-truth cubes with a matching file stem
    are loaded (via the same `load_hsi_file` / `align_hsi_orientation`
    helpers used for training) purely to report mrae/rmse/sam/psnr/ssim;
    they are never required for making a prediction.

    Returns the averaged metrics across all inputs with an available ground
    truth (empty dict if none was found).
    """
    model.eval()

    rgb_paths = discover_inference_inputs(input_path)
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    metric_totals = _empty_metric_totals()
    metric_count = 0

    print(f"\nRunning inference on {len(rgb_paths)} RGB image(s)")
    print(f"Sampling steps: {num_sampling_steps} | DDIM: {use_ddim_sampling}")
    print(f"Saving predictions to: {output_root}")

    for index, rgb_path in enumerate(rgb_paths, start=1):
        rgb_array = load_rgb_file(rgb_path)
        if not np.isfinite(rgb_array).all():
            print(f"  Skipping {rgb_path.name}: NaN or Inf values in RGB input")
            continue

        rgb_tensor = torch.from_numpy(np.ascontiguousarray(rgb_array)).float()
        rgb_tensor = rgb_tensor.unsqueeze(0).to(device, non_blocking=True)

        with autocast(enabled=use_amp):
            padded_rgb, (height, width) = pad_to_multiple(rgb_tensor, MODEL_DOWNSAMPLE_FACTOR)
            fine_hsi_padded = model.sample(
                padded_rgb,
                num_sampling_steps=num_sampling_steps,
                use_ddim=use_ddim_sampling,
            )
            y0_padded = model.get_coarse_prediction(padded_rgb)

        fine_hsi = fine_hsi_padded[..., :height, :width].float()
        y0 = y0_padded[..., :height, :width].float()

        fine_hsi_np = fine_hsi.squeeze(0).cpu().numpy()
        save_hsi_cube(
            fine_hsi_np,
            output_root / f"{rgb_path.stem}_pred.{save_format}",
            save_format,
        )
        if save_coarse:
            save_hsi_cube(
                y0.squeeze(0).cpu().numpy(),
                output_root / f"{rgb_path.stem}_coarse.{save_format}",
                save_format,
            )

        status = f"  [{index:04d}/{len(rgb_paths):04d}] {rgb_path.name} -> saved"

        if gt_hsi_dir is not None:
            gt_candidates = [
                candidate
                for candidate in find_files_with_extensions(gt_hsi_dir, SUPPORTED_HSI_EXTENSIONS)
                if candidate.stem == rgb_path.stem
            ]
            if gt_candidates:
                gt_cube = load_hsi_file(gt_candidates[0], HSI_KEY)
                gt_cube = convert_to_chw(gt_cube, HSI_CHANNELS, gt_candidates[0])
                gt_cube = align_hsi_orientation(
                    gt_cube, (rgb_array.shape[1], rgb_array.shape[2]), gt_candidates[0]
                )
                gt_tensor = (
                    torch.from_numpy(np.ascontiguousarray(gt_cube))
                    .float()
                    .unsqueeze(0)
                    .to(device)
                )
                metrics = calculate_metrics(fine_hsi, gt_tensor)
                for name, value in metrics.items():
                    metric_totals[name] += value
                metric_count += 1
                status += (
                    f" | MRAE: {metrics['mrae']:.4f} | PSNR: {metrics['psnr']:.2f} "
                    f"| SAM: {metrics['sam']:.4f}"
                )

        print(status)

    averaged_metrics = (
        {name: value / metric_count for name, value in metric_totals.items()}
        if metric_count > 0
        else {}
    )

    if averaged_metrics:
        print(
            f"\nAverage metrics over {metric_count} image(s) with ground truth:\n"
            f"  MRAE: {averaged_metrics['mrae']:.6f} | "
            f"RMSE: {averaged_metrics['rmse']:.6f} | "
            f"SAM: {averaged_metrics['sam']:.6f} | "
            f"PSNR: {averaged_metrics['psnr']:.4f} | "
            f"SSIM: {averaged_metrics['ssim']:.4f}"
        )
    elif gt_hsi_dir is not None:
        print("\nNo matching ground-truth HSI cubes were found; no metrics computed.")

    return averaged_metrics


def run_inference_cli(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = USE_AMP and device.type == "cuda"

    model, _checkpoint = load_model_for_inference(args.checkpoint, device)

    run_inference(
        model=model,
        input_path=args.input,
        output_dir=args.output,
        device=device,
        use_amp=use_amp,
        save_format=args.save_format,
        save_coarse=args.save_coarse,
        gt_hsi_dir=args.gt_hsi_dir,
        num_sampling_steps=args.sampling_steps,
        use_ddim_sampling=not args.no_ddim,
    )


# =============================================================================
# Dataset visualisation (sanity-check a handful of random RGB/HSI pairs)
# =============================================================================

# Defaults used by `python this_script.py visualize ...` when the
# corresponding CLI flag is not supplied.
VISUALIZE_NUM_IMAGES = 5
VISUALIZE_OUTPUT_PATH = str(Path(OUTPUT_DIR) / "dataset_preview.png")

# Which HSI bands to average together to build a human-viewable "false
# colour" composite (NTIRE/ARAD_1K cubes run roughly 400-700nm across 31
# bands, so the low/middle/high thirds approximate blue/green/red).
_HSI_FALSE_COLOR_BAND_FRACTIONS = ((0.75, 0.95), (0.45, 0.65), (0.05, 0.25))


def hsi_cube_to_false_color(cube_chw: np.ndarray) -> np.ndarray:
    """Turn a [C,H,W] HSI cube into a viewable [H,W,3] false-colour image.

    Each output channel is the mean of a band range spread across the
    spectrum (see `_HSI_FALSE_COLOR_BAND_FRACTIONS`), then the whole image
    is min-max stretched to [0, 1] purely for display -- this does not
    affect anything used in training.
    """
    num_bands = cube_chw.shape[0]
    channels = []

    for low_fraction, high_fraction in _HSI_FALSE_COLOR_BAND_FRACTIONS:
        low_index = int(round(low_fraction * (num_bands - 1)))
        high_index = int(round(high_fraction * (num_bands - 1)))
        low_index, high_index = sorted((low_index, high_index))
        channels.append(cube_chw[low_index : high_index + 1].mean(axis=0))

    composite = np.stack(channels, axis=-1)
    minimum = composite.min()
    maximum = composite.max()
    composite = (composite - minimum) / (maximum - minimum + 1e-8)
    return composite


def select_random_pairs(
    pairs: Sequence[Tuple[Path, Path]],
    num_images: int,
    seed: Optional[int] = None,
) -> List[Tuple[Path, Path]]:
    """Pick `num_images` distinct random HSI/RGB pairs to preview."""
    if not pairs:
        raise RuntimeError("No HSI/RGB pairs available to visualise.")

    sample_size = min(num_images, len(pairs))
    rng = random.Random(seed) if seed is not None else random
    return rng.sample(list(pairs), sample_size)


def visualize_random_pairs(
    pairs: Sequence[Tuple[Path, Path]],
    num_images: int,
    output_path: str,
    seed: Optional[int] = None,
) -> Path:
    """Save a figure previewing `num_images` random RGB/HSI pairs.

    For every sampled pair this shows, side by side:
      - the RGB photo, as a human would normally see it, and
      - a false-colour composite of the paired HSI cube (built by
        averaging low/mid/high spectral bands into blue/green/red so the
        31-band cube becomes viewable), with the number of bands and the
        image resolution captioned underneath.

    This is a read-only sanity check -- it does not touch training,
    validation, checkpoints, or any of the loss/metric computations above.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    chosen_pairs = select_random_pairs(pairs, num_images, seed=seed)

    figure, axes = plt.subplots(
        len(chosen_pairs), 2, figsize=(8, 4 * len(chosen_pairs)), squeeze=False
    )

    for row, (hsi_path, rgb_path) in enumerate(chosen_pairs):
        rgb_array = load_rgb_file(rgb_path)  # [3,H,W] in [0,1]
        rgb_image = np.transpose(rgb_array, (1, 2, 0))

        cube = load_hsi_file(hsi_path, HSI_KEY)
        cube = convert_to_chw(cube, HSI_CHANNELS, hsi_path)
        cube = align_hsi_orientation(
            cube, (rgb_array.shape[1], rgb_array.shape[2]), hsi_path
        )
        false_color_image = hsi_cube_to_false_color(cube)

        left_axis, right_axis = axes[row]

        left_axis.imshow(np.clip(rgb_image, 0.0, 1.0))
        left_axis.set_title(f"RGB: {rgb_path.name}", fontsize=10)
        left_axis.axis("off")

        right_axis.imshow(np.clip(false_color_image, 0.0, 1.0))
        right_axis.set_title(
            f"HSI (false colour): {hsi_path.name}\n"
            f"{cube.shape[0]} bands, {cube.shape[1]}x{cube.shape[2]} px",
            fontsize=10,
        )
        right_axis.axis("off")

    figure.suptitle(
        f"Random preview of {len(chosen_pairs)} RGB/HSI pair(s) from the dataset",
        fontsize=12,
    )
    figure.tight_layout()

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close(figure)

    print(f"\nSaved a preview of {len(chosen_pairs)} random RGB/HSI pair(s) to: {output_file}")
    for hsi_path, rgb_path in chosen_pairs:
        print(f"  RGB: {rgb_path}\n  HSI: {hsi_path}")

    return output_file


def run_visualization_cli(args: argparse.Namespace) -> None:
    """Entry point for `python this_script.py visualize ...`.

    Discovers RGB/HSI pairs from HSI_DATA_DIR/RGB_DATA_DIR (the same
    directories and pairing logic used for training), picks a random subset,
    and saves a side-by-side preview image. This never loads the model and
    never touches training/validation/checkpointing code.
    """
    pairs = find_paired_files(HSI_DATA_DIR, RGB_DATA_DIR)
    print(f"Found {len(pairs)} total HSI/RGB pairs in the dataset directories.")

    visualize_random_pairs(
        pairs=pairs,
        num_images=args.num_images,
        output_path=args.output,
        seed=args.seed,
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

    model = ResidualDiffusionRGB2HSI(
        rgb_channels=RGB_CHANNELS,
        hsi_channels=HSI_CHANNELS,
        num_timesteps=DIFFUSION_T,
        unet_base_channels=UNET_BASE_CHANNELS,
        unet_channel_multipliers=UNET_CHANNEL_MULTIPLIERS,
        unet_num_res_blocks=UNET_NUM_RES_BLOCKS,
        unet_attention_resolutions=UNET_ATTENTION_RESOLUTIONS,
        unet_dropout=UNET_DROPOUT,
        alpha_bar_max=DIFFUSION_ALPHA_BAR_MAX,
        beta_bar_min=DIFFUSION_BETA_BAR_MIN,
        beta_bar_max=DIFFUSION_BETA_BAR_MAX,
        delta_bar_max=DIFFUSION_DELTA_BAR_MAX,
        schedule_type=DIFFUSION_SCHEDULE_TYPE,
    ).to(device)

    if MST_CKPT_PATH is not None:
        # The model file itself does not load MST++ weights (by design, per
        # its own spec) -- it only instantiates and freezes the network. The
        # training script is therefore responsible for loading the
        # pretrained MST++ checkpoint into model.mst_plus_plus.
        mst_state_dict = torch.load(MST_CKPT_PATH, map_location=device)
        if isinstance(mst_state_dict, dict) and "state_dict" in mst_state_dict:
            mst_state_dict = mst_state_dict["state_dict"]
        model.mst_plus_plus.load_state_dict(mst_state_dict, strict=True)
        print(f"Loaded pretrained MST++ weights from: {MST_CKPT_PATH}")

    # Safety check only: MST++ is already frozen (requires_grad=False on all
    # of its parameters, always run under torch.no_grad(), forced into eval
    # mode by the model's own train() override) as soon as the model is
    # constructed. Re-asserting this after loading external weights adds no
    # new freezing behaviour; it just guards against the weights being
    # loaded from a checkpoint that happened to store requires_grad state.
    model._freeze_mst_plus_plus()

    trainable_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    print(
        f"Trainable parameters (residual diffusion U-Net): "
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

        # Full multi-step reverse-diffusion sampling only runs after the
        # final epoch, since it is far more expensive than the single-step
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
            f"  Train loss (single-step): {training_metrics['total_loss']:.6f} | "
            f"Train MRAE: {training_metrics['mrae']:.6f} | "
            f"Train PSNR: {training_metrics['psnr']:.4f}\n"
            f"  Val loss (single-step): {quick_metrics['total_loss']:.6f} | "
            f"Val MRAE: {quick_metrics['mrae']:.6f} | "
            f"Val RMSE: {quick_metrics['rmse']:.6f} | "
            f"Val SAM: {quick_metrics['sam']:.6f} | "
            f"Val PSNR: {quick_metrics['psnr']:.4f} | "
            f"Val SSIM: {quick_metrics['ssim']:.4f}\n"
            f"  Coarse net (MST++) only | MRAE: {coarse_metrics['mrae']:.6f} | "
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
            output_path=output_dir / "last_residual_diffusion.pth",
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
                output_path=output_dir / "best_residual_diffusion.pth",
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
                output_path=output_dir / "final_residual_diffusion_with_full_sampling.pth",
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
                f"Saved to: {output_dir / 'final_residual_diffusion_with_full_sampling.pth'}"
            )


# =============================================================================
# CLI entry point
# =============================================================================


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train the residual diffusion RGB->HSI model, or run inference "
            "with a saved checkpoint."
        )
    )
    subparsers = parser.add_subparsers(dest="mode")

    subparsers.add_parser(
        "train",
        help="Train the model using the configuration constants at the top of this file.",
    )

    infer_parser = subparsers.add_parser(
        "infer",
        help="Run full reverse-diffusion inference on one RGB image or a directory of RGB images.",
    )
    infer_parser.add_argument(
        "--checkpoint",
        type=str,
        default=INFER_CHECKPOINT_PATH,
        help=f"Path to a .pth checkpoint saved by this script (default: {INFER_CHECKPOINT_PATH})",
    )
    infer_parser.add_argument(
        "--input",
        type=str,
        default=INFER_INPUT_DIR,
        help="Path to a single RGB image file or a directory of RGB images.",
    )
    infer_parser.add_argument(
        "--output",
        type=str,
        default=INFER_OUTPUT_DIR,
        help=f"Directory to save predicted HSI cubes to (default: {INFER_OUTPUT_DIR})",
    )
    infer_parser.add_argument(
        "--save-format",
        type=str,
        choices=["mat", "npy"],
        default=INFER_SAVE_FORMAT,
        help="File format for saved HSI cubes.",
    )
    infer_parser.add_argument(
        "--save-coarse",
        action="store_true",
        default=INFER_SAVE_COARSE,
        help="Also save the frozen MST++ coarse prediction (y0) for each input.",
    )
    infer_parser.add_argument(
        "--gt-hsi-dir",
        type=str,
        default=INFER_GT_HSI_DIR,
        help=(
            "Optional directory of ground-truth HSI cubes (matched to RGB "
            "inputs by file stem) used only to report mrae/rmse/sam/psnr/ssim."
        ),
    )
    infer_parser.add_argument(
        "--sampling-steps",
        type=int,
        default=NUM_SAMPLING_STEPS,
        help=f"Number of reverse-diffusion sampling steps (default: {NUM_SAMPLING_STEPS})",
    )
    infer_parser.add_argument(
        "--no-ddim",
        action="store_true",
        help="Use full DDPM ancestral sampling instead of DDIM.",
    )

    visualize_parser = subparsers.add_parser(
        "visualize",
        help=(
            "Preview a handful of random RGB/HSI pairs from the dataset "
            "directories as a side-by-side image (RGB photo vs. false-colour "
            "HSI). Does not load the model or touch training/checkpoints."
        ),
    )
    visualize_parser.add_argument(
        "--num-images",
        type=int,
        default=VISUALIZE_NUM_IMAGES,
        help=f"How many random pairs to preview (default: {VISUALIZE_NUM_IMAGES})",
    )
    visualize_parser.add_argument(
        "--output",
        type=str,
        default=VISUALIZE_OUTPUT_PATH,
        help=f"Where to save the preview image (default: {VISUALIZE_OUTPUT_PATH})",
    )
    visualize_parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed, for a reproducible choice of preview images.",
    )

    return parser


if __name__ == "__main__":
    cli_args = build_arg_parser().parse_args()

    if cli_args.mode == "infer":
        run_inference_cli(cli_args)
    elif cli_args.mode == "visualize":
        run_visualization_cli(cli_args)
    else:
        # Default to training when no subcommand (or "train") is given, so
        # `python this_script.py` keeps working exactly as before.
        main()
