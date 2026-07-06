"""Evaluate a trained I2SB RGB -> HSI model on the complete validation split
and visualize a few randomly selected validation reconstructions.

This parser-free script:
    1. Recreates exactly the same deterministic validation split used during
       training.
    2. Runs full RGB -> frozen MST++ -> I2SB reverse sampling on every image
       in the validation split.
    3. Reports the mean validation MRAE, RMSE, SAM, PSNR, and SSIM for:
           - the frozen MST++ coarse prediction
           - the fully sampled I2SB reconstruction
    4. Selects a small number of validation images for visualization.
    5. Creates a separate 2 x 2 visualization for every selected image:
           - Ground-truth HSI pseudo-RGB
           - Full I2SB reconstruction pseudo-RGB
           - Mean absolute error over spectral bands
           - Ground-truth and reconstructed spectra at three spatial locations

The selected visualization samples are retained during the full validation
pass, so they are not inferred a second time. Metrics use the same imported
loss functions used by training, with the target passed first.
There is no argparse parser; edit the Configuration section directly.
"""

from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import h5py
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import torch
import torch.nn.functional as F
from PIL import Image
from torch.cuda.amp import autocast

# Adjust these imports to match the repository layout used for training.
from model.I2I_SB import I2SBModel
from loss.mrae import mrae
from loss.psnr import psnr
from loss.rmse import rmse
from loss.sam import sam
from loss.ssim import ssim


# =============================================================================
# Configuration -- edit values here; no command-line parser is used
# =============================================================================

HSI_DATA_DIR = (
    "/kaggle/input/datasets/sriramhari14/ntire-2022/"
    "Train_spectral/Train_spectral"
)
RGB_DATA_DIR = (
    "/kaggle/input/datasets/sriramhari14/ntire-2022/"
    "Train_RGB/Train_RGB"
)

# Prefer the final checkpoint when it exists because it contains the model from
# the final epoch. Change this to best_i2sb.pth to inspect the best quick-MRAE
# checkpoint instead.
CHECKPOINT_PATH = "./i2sb_checkpoints/best_i2sb.pth"

# Only needed when CHECKPOINT_PATH does not contain the frozen MST++ weights.
# Checkpoints produced by the supplied training script save model.state_dict(),
# so they normally already contain MST++ and this should remain None.
MST_CKPT_PATH = None

OUTPUT_DIR = Path("./i2sb_visualisation")

HSI_KEY = "cube"
HSI_CHANNELS = 31
RGB_CHANNELS = 3
SUPPORTED_HSI_EXTENSIONS = {".mat", ".npy", ".npz", ".pt", ".pth"}
SUPPORTED_RGB_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".npy"}

# These values must exactly match training so the same deterministic
# validation split is reconstructed. Inference is restricted to this split.
VALIDATION_FRACTION = 0.10
SEED = 42

# -----------------------------------------------------------------------------
# Image selection
# -----------------------------------------------------------------------------
# Selection priority:
#   1. SELECTED_IMAGE_STEMS, when non-empty.
#   2. SELECTED_DATASET_INDICES, when non-empty.
#   3. RANDOM_SELECTION_COUNT random images from the chosen pool.
#
# A stem is the filename without its extension. For example,
# "ARAD_1K_0001" selects ARAD_1K_0001.mat and ARAD_1K_0001.jpg/png.
SELECTED_IMAGE_STEMS: list[str] = [
    # "ARAD_1K_0001",
    # "ARAD_1K_0017",
    # "ARAD_1K_0052",
]

# Indices refer only to the deterministic validation split returned by
# split_pairs(). Training-split and complete-dataset indices are not accepted.
SELECTED_DATASET_INDICES: list[int] = []

# Used only when both lists above are empty.
RANDOM_SELECTION_COUNT = 3

# Must match training-time preprocessing.
NORMALIZATION = "none"  # "none", "minmax", or "band_minmax"

# Print running validation progress at this interval.
EVALUATION_PRINT_EVERY = 1

# Fallback architecture values. When available, model_config stored inside the
# checkpoint overrides these values automatically.
UNET_BASE_CHANNELS = 64
UNET_CHANNEL_MULTIPLIERS = (1, 2, 4)
UNET_NUM_RES_BLOCKS = 2
UNET_TIME_DIM = 256
UNET_CONDITION_ON_X1 = True
I2SB_NUM_TRAIN_TIMESTEPS = 1000
I2SB_BETA_MIN = 1.0e-6
I2SB_BETA_MAX = 1.2e-4

# Full I2SB reverse sampling settings.
NUM_SAMPLING_STEPS = 50
USE_AMP = True

# Full-image mode exactly mirrors the validation logic in the training script.
# If the bottleneck attention causes CUDA OOM at native resolution, set this to
# True. Tiled reverse sampling computes MST++ on the full padded image, then
# refines overlapping X1 tiles and averages overlaps.
USE_TILED_REVERSE = False
REVERSE_TILE_SIZE = 256
REVERSE_TILE_OVERLAP = 32

# Save the selected full HSI cubes in addition to the PNG figure.
SAVE_VISUALISED_CUBES = True

# Approximate ARAD/NTIRE wavelengths. Change these when using another sensor.
WAVELENGTH_START_NM = 400.0
WAVELENGTH_END_NM = 700.0

# Pseudo-RGB bands for a 31-band 400--700 nm cube:
# R ~= 640 nm, G ~= 550 nm, B ~= 460 nm.
DISPLAY_RGB_BAND_INDICES = (24, 15, 6)
DISPLAY_PERCENTILES = (1.0, 99.0)

# Relative (y, x) positions used for the three spectral signature comparisons.
# These match the visualization style in the supplied VAE script.
SPECTRAL_LOCATIONS = (
    (0.25, 0.25),
    (0.50, 0.50),
    (0.75, 0.75),
)

SAVE_FIGURES = True
SHOW_FIGURES = True
FIGURE_DPI = 180


# =============================================================================
# Reproducibility
# =============================================================================


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# =============================================================================
# Paired file discovery and loading (adapted from the training script)
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


def find_paired_files(hsi_dir: str, rgb_dir: str) -> List[Tuple[Path, Path]]:
    hsi_files = find_files_with_extensions(hsi_dir, SUPPORTED_HSI_EXTENSIONS)
    rgb_files = find_files_with_extensions(rgb_dir, SUPPORTED_RGB_EXTENSIONS)

    rgb_by_stem: Dict[str, Path] = {}
    duplicate_stems: set[str] = set()
    for rgb_path in rgb_files:
        if rgb_path.stem in rgb_by_stem:
            duplicate_stems.add(rgb_path.stem)
        else:
            rgb_by_stem[rgb_path.stem] = rgb_path

    if duplicate_stems:
        examples = sorted(duplicate_stems)[:5]
        raise RuntimeError(
            "Multiple RGB files share the same stem, so pairing is ambiguous. "
            f"Example duplicate stems: {examples}"
        )

    pairs: List[Tuple[Path, Path]] = []
    missing: List[str] = []
    for hsi_path in hsi_files:
        rgb_path = rgb_by_stem.get(hsi_path.stem)
        if rgb_path is None:
            missing.append(hsi_path.stem)
        else:
            pairs.append((hsi_path, rgb_path))

    if missing:
        print(
            f"Warning: skipped {len(missing)} HSI files without paired RGB; "
            f"examples: {missing[:5]}"
        )
    if not pairs:
        raise RuntimeError("No paired HSI/RGB files were found.")
    return pairs


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
    return shuffled[validation_size:], shuffled[:validation_size]


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
        raise ValueError(f"No numerical 3D HSI array found in {file_path}")
    return max(candidates, key=lambda item: item[1].size)[1]


def load_mat_v73(file_path: Path, hsi_key: str) -> np.ndarray:
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
    # MATLAB v7.3 arrays commonly appear in reversed axis order through h5py.
    return np.transpose(cube, axes=tuple(range(cube.ndim - 1, -1, -1)))


def extract_array_from_dictionary(
    data: Mapping[str, Any],
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
        elif isinstance(loaded, Mapping):
            cube = extract_array_from_dictionary(loaded, file_path, hsi_key)
        else:
            raise TypeError(
                f"Unsupported object type {type(loaded).__name__} in {file_path}"
            )
    else:
        raise ValueError(f"Unsupported HSI extension: {extension}")

    cube = np.squeeze(np.asarray(cube, dtype=np.float32))
    if cube.ndim != 3:
        raise ValueError(
            f"Expected a 3D HSI cube in {file_path}, found shape {cube.shape}"
        )
    return cube


def convert_to_chw(cube: np.ndarray, hsi_channels: int, file_path: Path) -> np.ndarray:
    if cube.shape[0] == hsi_channels:
        return cube
    if cube.shape[-1] == hsi_channels:
        return np.transpose(cube, (2, 0, 1))
    raise ValueError(
        f"Cannot locate the {hsi_channels}-band axis in {file_path}; "
        f"found shape {cube.shape}."
    )


def align_hsi_orientation(
    cube_chw: np.ndarray,
    target_hw: Tuple[int, int],
    file_path: Path,
) -> np.ndarray:
    current_hw = (cube_chw.shape[1], cube_chw.shape[2])
    if current_hw == target_hw:
        return cube_chw
    if current_hw == (target_hw[1], target_hw[0]):
        return np.transpose(cube_chw, (0, 2, 1))
    raise ValueError(
        f"Cannot align HSI size {current_hw} in {file_path} with RGB size "
        f"{target_hw}, even after transposition."
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


def load_rgb_file(file_path: Path) -> np.ndarray:
    if file_path.suffix.lower() == ".npy":
        image = np.asarray(np.load(file_path), dtype=np.float32)
        if image.max() > 1.0 + 1e-3:
            image = image / 255.0
    else:
        with Image.open(file_path) as handle:
            image = np.asarray(handle.convert("RGB"), dtype=np.float32) / 255.0

    image = np.squeeze(image)
    if image.ndim != 3:
        raise ValueError(
            f"Expected a 3D RGB image in {file_path}, found shape {image.shape}"
        )
    if image.shape[0] == RGB_CHANNELS:
        return image
    if image.shape[-1] == RGB_CHANNELS:
        return np.transpose(image, (2, 0, 1))
    raise ValueError(
        f"Cannot locate the RGB axis in {file_path}; found shape {image.shape}."
    )




# =============================================================================
# Checkpoint and model construction
# =============================================================================


def torch_load_compat(path: str | Path, map_location: torch.device | str) -> Any:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def strip_repeated_prefix(key: str, prefixes: Iterable[str]) -> str:
    changed = True
    while changed:
        changed = False
        for prefix in prefixes:
            if key.startswith(prefix):
                key = key[len(prefix):]
                changed = True
    return key


def normalize_full_model_state_dict(state_dict: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    cleaned: Dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        # DataParallel/DistributedDataParallel wrappers are the common case.
        cleaned[strip_repeated_prefix(key, ("module.",))] = value
    return cleaned


def extract_checkpoint_state(checkpoint: Any) -> Mapping[str, torch.Tensor]:
    if not isinstance(checkpoint, Mapping):
        raise TypeError(
            f"Checkpoint must be a mapping, found {type(checkpoint).__name__}."
        )

    for key in ("model_state_dict", "state_dict", "model"):
        value = checkpoint.get(key)
        if isinstance(value, Mapping) and value:
            return value

    if checkpoint and all(isinstance(value, torch.Tensor) for value in checkpoint.values()):
        return checkpoint

    raise KeyError(
        "Could not find model weights. Expected model_state_dict, state_dict, "
        "model, or a raw tensor state dictionary."
    )


def load_mst_weights(model: I2SBModel, checkpoint_path: str, device: torch.device) -> None:
    checkpoint = torch_load_compat(checkpoint_path, device)
    state = extract_checkpoint_state(checkpoint)

    cleaned: Dict[str, torch.Tensor] = {}
    for key, value in state.items():
        key = strip_repeated_prefix(
            key,
            (
                "module.",
                "model.",
                "backbone.mstpp_model.",
                "mstpp_model.",
            ),
        )
        cleaned[key] = value

    model.backbone.mstpp_model.load_state_dict(cleaned, strict=True)
    print(f"Loaded standalone MST++ checkpoint: {checkpoint_path}")


def build_model_from_checkpoint(
    checkpoint_path: str,
    device: torch.device,
) -> Tuple[I2SBModel, Mapping[str, Any]]:
    checkpoint = torch_load_compat(checkpoint_path, "cpu")
    checkpoint_config = checkpoint.get("model_config", {}) if isinstance(checkpoint, Mapping) else {}

    hsi_channels = int(checkpoint_config.get("hsi_channels", HSI_CHANNELS))
    base_channels = int(checkpoint_config.get("base_channels", UNET_BASE_CHANNELS))
    channel_mults = tuple(
        int(value)
        for value in checkpoint_config.get(
            "channel_mults", UNET_CHANNEL_MULTIPLIERS
        )
    )
    num_res_blocks = int(
        checkpoint_config.get("num_res_blocks", UNET_NUM_RES_BLOCKS)
    )
    time_dim = int(checkpoint_config.get("time_dim", UNET_TIME_DIM))
    condition_on_x1 = bool(
        checkpoint_config.get("condition_on_x1", UNET_CONDITION_ON_X1)
    )
    num_train_timesteps = int(
        checkpoint_config.get(
            "num_train_timesteps", I2SB_NUM_TRAIN_TIMESTEPS
        )
    )
    beta_min = float(checkpoint_config.get("beta_min", I2SB_BETA_MIN))
    beta_max = float(checkpoint_config.get("beta_max", I2SB_BETA_MAX))

    model = I2SBModel(
        mstpp_model=None,
        hsi_channels=hsi_channels,
        base_channels=base_channels,
        channel_mults=channel_mults,
        num_res_blocks=num_res_blocks,
        time_dim=time_dim,
        condition_on_x1=condition_on_x1,
        num_train_timesteps=num_train_timesteps,
        beta_min=beta_min,
        beta_max=beta_max,
        freeze_mstpp=True,
    )

    if MST_CKPT_PATH is not None:
        load_mst_weights(model, MST_CKPT_PATH, device=torch.device("cpu"))

    state = normalize_full_model_state_dict(extract_checkpoint_state(checkpoint))
    model_keys = set(model.state_dict().keys())
    state_keys = set(state.keys())

    # Standard checkpoint produced by the supplied training script.
    if state_keys & model_keys:
        incompatible = model.load_state_dict(state, strict=False)
        missing = list(incompatible.missing_keys)
        unexpected = list(incompatible.unexpected_keys)

        missing_eps = [key for key in missing if key.startswith("eps_net.")]
        if missing_eps:
            raise RuntimeError(
                "The checkpoint is missing epsilon-network weights required for "
                f"inference. Examples: {missing_eps[:8]}"
            )

        # Missing backbone keys are allowed only when an external MST checkpoint
        # was explicitly supplied.
        missing_backbone = [
            key for key in missing if key.startswith("backbone.mstpp_model.")
        ]
        if missing_backbone and MST_CKPT_PATH is None:
            raise RuntimeError(
                "The I2SB checkpoint does not contain all MST++ weights. Set "
                "MST_CKPT_PATH to the pretrained MST++ checkpoint. Missing "
                f"examples: {missing_backbone[:8]}"
            )

        if unexpected:
            print(f"Warning: ignored {len(unexpected)} unexpected checkpoint keys.")
    else:
        # Also support a checkpoint that contains only UNetEpsilonNet weights
        # without an eps_net. prefix.
        eps_state = {
            strip_repeated_prefix(key, ("module.", "eps_net.")): value
            for key, value in state.items()
        }
        model.eps_net.load_state_dict(eps_state, strict=True)
        if MST_CKPT_PATH is None:
            raise RuntimeError(
                "Checkpoint appears to contain only epsilon-network weights. "
                "Set MST_CKPT_PATH so the frozen MST++ boundary model is loaded."
            )

    for parameter in model.backbone.mstpp_model.parameters():
        parameter.requires_grad_(False)
    model.backbone.mstpp_model.eval()
    model.eval()
    model.to(device)

    epoch = checkpoint.get("epoch") if isinstance(checkpoint, Mapping) else None
    print(f"Loaded I2SB checkpoint: {checkpoint_path}")
    if epoch is not None:
        print(f"Checkpoint epoch: {epoch}")
    print(
        "Model configuration: "
        f"channels={hsi_channels}, base={base_channels}, mults={channel_mults}, "
        f"res_blocks={num_res_blocks}, time_dim={time_dim}, "
        f"condition_on_x1={condition_on_x1}"
    )

    return model, checkpoint


# =============================================================================
# Padding and inference
# =============================================================================


def pad_to_multiple(
    tensor: torch.Tensor,
    multiple: int,
) -> Tuple[torch.Tensor, Tuple[int, int]]:
    if tensor.ndim != 4:
        raise ValueError(f"Expected [B,C,H,W], found {tuple(tensor.shape)}")
    if multiple <= 0:
        raise ValueError("multiple must be positive")

    original_height, original_width = tensor.shape[-2:]
    padded_height = math.ceil(original_height / multiple) * multiple
    padded_width = math.ceil(original_width / multiple) * multiple
    pad_bottom = padded_height - original_height
    pad_right = padded_width - original_width

    if pad_bottom == 0 and pad_right == 0:
        return tensor, (original_height, original_width)
    return (
        F.pad(tensor, (0, pad_right, 0, pad_bottom), mode="replicate"),
        (original_height, original_width),
    )


def sliding_starts(length: int, tile_size: int, overlap: int) -> List[int]:
    if tile_size <= 0:
        raise ValueError("REVERSE_TILE_SIZE must be positive.")
    if overlap < 0 or overlap >= tile_size:
        raise ValueError("REVERSE_TILE_OVERLAP must satisfy 0 <= overlap < tile_size.")
    if length <= tile_size:
        return [0]

    stride = tile_size - overlap
    starts = list(range(0, max(length - tile_size + 1, 1), stride))
    final_start = length - tile_size
    if starts[-1] != final_start:
        starts.append(final_start)
    return starts


@torch.inference_mode()
def reverse_sample_tiled(
    model: I2SBModel,
    x1: torch.Tensor,
    num_steps: int,
    tile_size: int,
    overlap: int,
) -> torch.Tensor:
    """Refine a full X1 boundary using overlapping tiles and mean blending."""
    if x1.shape[0] != 1:
        raise ValueError("Tiled reverse sampling currently requires batch size one.")

    _, channels, height, width = x1.shape
    y_starts = sliding_starts(height, tile_size, overlap)
    x_starts = sliding_starts(width, tile_size, overlap)

    accumulation = torch.zeros_like(x1, dtype=torch.float32)
    weights = torch.zeros(
        (1, 1, height, width), device=x1.device, dtype=torch.float32
    )

    for top in y_starts:
        for left in x_starts:
            bottom = min(top + tile_size, height)
            right = min(left + tile_size, width)
            tile = x1[..., top:bottom, left:right]
            tile_h, tile_w = tile.shape[-2:]

            pad_bottom = tile_size - tile_h
            pad_right = tile_size - tile_w
            if pad_bottom or pad_right:
                tile_padded = F.pad(
                    tile, (0, pad_right, 0, pad_bottom), mode="replicate"
                )
            else:
                tile_padded = tile

            cond = tile_padded if model.condition_on_x1 else None
            refined_padded = model.reverse_sample(
                tile_padded,
                cond=cond,
                num_steps=num_steps,
            )
            refined = refined_padded[..., :tile_h, :tile_w].float()

            accumulation[..., top:bottom, left:right] += refined
            weights[..., top:bottom, left:right] += 1.0

    return accumulation / weights.clamp_min(1.0)


@torch.inference_mode()
def run_full_inference(
    model: I2SBModel,
    rgb: torch.Tensor,
    model_downsample_factor: int,
    num_steps: int,
    use_amp: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return (refined I2SB HSI, coarse MST++ HSI), cropped to native size."""
    padded_rgb, (height, width) = pad_to_multiple(rgb, model_downsample_factor)

    with autocast(enabled=use_amp):
        x1_padded = model.get_degraded_prediction(padded_rgb)

        if USE_TILED_REVERSE:
            refined_padded = reverse_sample_tiled(
                model=model,
                x1=x1_padded,
                num_steps=num_steps,
                tile_size=REVERSE_TILE_SIZE,
                overlap=REVERSE_TILE_OVERLAP,
            )
        else:
            cond = x1_padded if model.condition_on_x1 else None
            refined_padded = model.reverse_sample(
                x1_padded,
                cond=cond,
                num_steps=num_steps,
            )

    refined = refined_padded[..., :height, :width].float()
    coarse = x1_padded[..., :height, :width].float()
    return refined, coarse


# =============================================================================
# Metrics and aggregation
# =============================================================================


def empty_metric_totals() -> Dict[str, float]:
    return {
        "mrae": 0.0,
        "rmse": 0.0,
        "sam": 0.0,
        "psnr": 0.0,
        "ssim": 0.0,
    }


@torch.inference_mode()
def calculate_metrics(
    reconstruction: torch.Tensor,
    target: torch.Tensor,
) -> Dict[str, float]:
    """Calculate the same five metrics used by the training script.

    The imported metric functions receive the ground-truth target first and
    the reconstruction second, matching the convention used during training.
    """
    reconstruction = reconstruction.float()
    target = target.float()

    if reconstruction.shape != target.shape:
        raise ValueError(
            f"Metric shape mismatch: reconstruction={tuple(reconstruction.shape)}, "
            f"target={tuple(target.shape)}"
        )

    values = {
        "mrae": float(mrae(target, reconstruction).item()),
        "rmse": float(rmse(target, reconstruction).item()),
        "sam": float(sam(target, reconstruction).item()),
        "psnr": float(psnr(target, reconstruction).item()),
        "ssim": float(ssim(target, reconstruction).item()),
    }

    non_finite = [
        name for name, value in values.items()
        if not math.isfinite(value)
    ]
    if non_finite:
        raise FloatingPointError(f"Non-finite metrics: {non_finite}")

    return values


def add_metrics(
    totals: Dict[str, float],
    values: Mapping[str, float],
) -> None:
    for name in totals:
        totals[name] += float(values[name])


def average_metrics(
    totals: Mapping[str, float],
    count: int,
) -> Dict[str, float]:
    if count <= 0:
        raise ValueError("Cannot average metrics over zero images.")
    return {
        name: float(value) / count
        for name, value in totals.items()
    }


def format_metrics(metrics: Mapping[str, float]) -> str:
    return (
        f"MRAE: {metrics['mrae']:.6f} | "
        f"RMSE: {metrics['rmse']:.6f} | "
        f"SAM: {metrics['sam']:.6f} | "
        f"PSNR: {metrics['psnr']:.4f} | "
        f"SSIM: {metrics['ssim']:.4f}"
    )




# =============================================================================
# Visualisation
# =============================================================================


def derive_display_limits(gt_hsi: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    selected = gt_hsi[list(DISPLAY_RGB_BAND_INDICES)]
    lower, upper = DISPLAY_PERCENTILES
    lows = np.percentile(selected, lower, axis=(1, 2)).astype(np.float32)
    highs = np.percentile(selected, upper, axis=(1, 2)).astype(np.float32)
    highs = np.maximum(highs, lows + 1e-8)
    return lows, highs


def hsi_to_pseudo_rgb(
    hsi_chw: np.ndarray,
    lows: np.ndarray,
    highs: np.ndarray,
) -> np.ndarray:
    selected = hsi_chw[list(DISPLAY_RGB_BAND_INDICES)]
    normalized = (selected - lows[:, None, None]) / (
        highs[:, None, None] - lows[:, None, None]
    )
    return np.clip(np.transpose(normalized, (1, 2, 0)), 0.0, 1.0)


def select_highest_error_pixel(
    refined_hsi: np.ndarray,
    target_hsi: np.ndarray,
) -> Tuple[int, int, np.ndarray]:
    error_map = np.mean(np.abs(refined_hsi - target_hsi), axis=0)
    flat_index = int(np.argmax(error_map))
    row, col = np.unravel_index(flat_index, error_map.shape)
    return int(row), int(col), error_map


def create_visualisation_figure(
    sample: Mapping[str, Any],
    output_dir: Path,
) -> Path | None:
    """Create one VAE-style 2 x 2 figure for a selected validation sample.

    The reconstruction uses the ground-truth percentile limits for pseudo-RGB
    conversion, so differences are not hidden by separate contrast stretching.
    """
    target = np.asarray(sample["target"], dtype=np.float32)
    refined = np.asarray(sample["refined"], dtype=np.float32)
    stem = str(sample["stem"])
    metrics = sample["refined_metrics"]

    if target.shape != refined.shape:
        raise ValueError(
            f"Cannot visualize {stem}: target shape {target.shape} and "
            f"reconstruction shape {refined.shape} differ."
        )

    lows, highs = derive_display_limits(target)
    target_rgb = hsi_to_pseudo_rgb(target, lows, highs)
    refined_rgb = hsi_to_pseudo_rgb(refined, lows, highs)
    error_map = np.mean(np.abs(refined - target), axis=0)

    figure, axes = plt.subplots(
        2,
        2,
        figsize=(13, 10),
    )

    axes[0, 0].imshow(target_rgb)
    axes[0, 0].set_title("Ground-truth HSI pseudo-RGB")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(refined_rgb)
    axes[0, 1].set_title("I2SB reconstruction pseudo-RGB")
    axes[0, 1].axis("off")

    error_vmax = max(float(np.percentile(error_map, 99.0)), 1e-8)
    error_image = axes[1, 0].imshow(
        error_map,
        cmap="magma",
        vmin=0.0,
        vmax=error_vmax,
    )
    axes[1, 0].set_title("Mean absolute error over spectral bands")
    axes[1, 0].axis("off")
    figure.colorbar(
        error_image,
        ax=axes[1, 0],
        fraction=0.046,
        pad=0.04,
    )

    height, width = target.shape[1:]
    wavelengths = np.linspace(
        WAVELENGTH_START_NM,
        WAVELENGTH_END_NM,
        target.shape[0],
        dtype=np.float32,
    )

    for point_index, (relative_y, relative_x) in enumerate(
        SPECTRAL_LOCATIONS,
        start=1,
    ):
        y = int(round(relative_y * (height - 1)))
        x = int(round(relative_x * (width - 1)))

        target_line = axes[1, 1].plot(
            wavelengths,
            target[:, y, x],
            linewidth=2.0,
            label=f"GT P{point_index} ({y},{x})",
        )[0]

        axes[1, 1].plot(
            wavelengths,
            refined[:, y, x],
            linestyle="--",
            linewidth=2.0,
            color=target_line.get_color(),
            label=f"I2SB P{point_index}",
        )

        for image_axis in (axes[0, 0], axes[0, 1]):
            image_axis.scatter(
                [x],
                [y],
                s=35,
                marker="o",
                facecolors="none",
                edgecolors=target_line.get_color(),
                linewidths=1.5,
            )
            image_axis.text(
                x + 5,
                y + 5,
                f"P{point_index}",
                color=target_line.get_color(),
                fontsize=8,
                weight="bold",
            )

    axes[1, 1].set_title("Spectral signatures")
    axes[1, 1].set_xlabel("Wavelength (nm)")
    axes[1, 1].set_ylabel("Intensity")
    axes[1, 1].grid(alpha=0.3)
    axes[1, 1].legend(fontsize=8, ncol=2)

    figure.suptitle(
        f"{stem}\n"
        f"MRAE: {metrics['mrae']:.6f} | "
        f"RMSE: {metrics['rmse']:.6f} | "
        f"PSNR: {metrics['psnr']:.3f} dB | "
        f"SAM: {metrics['sam']:.3f} | "
        f"SSIM: {metrics['ssim']:.4f}",
        fontsize=13,
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.93))

    saved_path: Path | None = None
    if SAVE_FIGURES:
        output_dir.mkdir(parents=True, exist_ok=True)
        saved_path = output_dir / f"{stem}_i2sb_reconstruction.png"
        figure.savefig(
            saved_path,
            dpi=FIGURE_DPI,
            bbox_inches="tight",
        )
        print(f"Saved visualization: {saved_path}")

    if SHOW_FIGURES:
        plt.show()

    plt.close(figure)
    return saved_path


# =============================================================================
# Selected-sample orchestration
# =============================================================================


def choose_visualisation_pairs(
    pairs: Sequence[Tuple[Path, Path]],
) -> List[Tuple[int, Path, Path]]:
    """Select pairs only from the reconstructed validation split.

    The caller must pass validation_pairs, never all_pairs or training_pairs.
    Requested stems that are not members of validation raise an error instead
    of silently falling back to another dataset partition.
    """
    if not pairs:
        raise RuntimeError("The visualisation pool is empty.")

    if SELECTED_IMAGE_STEMS:
        duplicate_requested = {
            stem for stem in SELECTED_IMAGE_STEMS
            if SELECTED_IMAGE_STEMS.count(stem) > 1
        }
        if duplicate_requested:
            raise ValueError(
                "SELECTED_IMAGE_STEMS contains duplicates: "
                f"{sorted(duplicate_requested)}"
            )

        by_stem = {hsi_path.stem: (index, hsi_path, rgb_path)
                   for index, (hsi_path, rgb_path) in enumerate(pairs)}
        missing = [stem for stem in SELECTED_IMAGE_STEMS if stem not in by_stem]
        if missing:
            available_examples = sorted(by_stem)[:10]
            raise KeyError(
                "The following selected stems are not members of the validation "
                f"split: {missing}. Validation examples: {available_examples}"
            )
        return [by_stem[stem] for stem in SELECTED_IMAGE_STEMS]

    if SELECTED_DATASET_INDICES:
        duplicate_indices = {
            index for index in SELECTED_DATASET_INDICES
            if SELECTED_DATASET_INDICES.count(index) > 1
        }
        if duplicate_indices:
            raise ValueError(
                "SELECTED_DATASET_INDICES contains duplicates: "
                f"{sorted(duplicate_indices)}"
            )

        invalid = [
            index for index in SELECTED_DATASET_INDICES
            if index < 0 or index >= len(pairs)
        ]
        if invalid:
            raise IndexError(
                f"Selected indices {invalid} are outside [0, {len(pairs) - 1}]."
            )
        return [
            (index, pairs[index][0], pairs[index][1])
            for index in SELECTED_DATASET_INDICES
        ]

    if RANDOM_SELECTION_COUNT <= 0:
        raise ValueError("RANDOM_SELECTION_COUNT must be positive.")
    if RANDOM_SELECTION_COUNT > len(pairs):
        raise ValueError(
            f"RANDOM_SELECTION_COUNT={RANDOM_SELECTION_COUNT}, but the pool "
            f"contains only {len(pairs)} images."
        )

    selected_indices = random.Random(SEED).sample(
        range(len(pairs)), k=RANDOM_SELECTION_COUNT
    )
    return [
        (index, pairs[index][0], pairs[index][1])
        for index in selected_indices
    ]


def save_visualised_sample_cubes(
    sample: Mapping[str, Any],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_dir / f"{sample['stem']}_visualised_sample.npz",
        rgb=sample["rgb"].astype(np.float32),
        target_hsi=sample["target"].astype(np.float32),
        mst_coarse_hsi=sample["coarse"].astype(np.float32),
        i2sb_refined_hsi=sample["refined"].astype(np.float32),
    )


@torch.inference_mode()
def evaluate_validation_split(
    model: I2SBModel,
    validation_pairs: Sequence[Tuple[Path, Path]],
    selected_pairs: Sequence[Tuple[int, Path, Path]],
    device: torch.device,
    model_downsample_factor: int,
    use_amp: bool,
) -> Tuple[Dict[str, float], Dict[str, float], List[Dict[str, Any]]]:
    """Evaluate every validation image and retain selected samples for plots.

    Dataset-level values are the arithmetic mean of the per-image metrics,
    matching the validation aggregation used in the supplied training script.
    Validation batch size is effectively one, so native-resolution images of
    different sizes are supported.
    """
    if not validation_pairs:
        raise RuntimeError("The validation split is empty.")

    selected_indices = [index for index, _, _ in selected_pairs]
    selected_index_set = set(selected_indices)
    selected_samples: Dict[int, Dict[str, Any]] = {}

    coarse_totals = empty_metric_totals()
    refined_totals = empty_metric_totals()
    evaluated_count = 0

    for pool_index, (hsi_path, rgb_path) in enumerate(validation_pairs):
        hsi = load_hsi_file(hsi_path, HSI_KEY)
        hsi = convert_to_chw(hsi, HSI_CHANNELS, hsi_path)
        rgb = load_rgb_file(rgb_path)

        if not np.isfinite(hsi).all():
            raise ValueError(f"NaN/Inf found in {hsi_path}")
        if not np.isfinite(rgb).all():
            raise ValueError(f"NaN/Inf found in {rgb_path}")

        hsi = align_hsi_orientation(
            hsi,
            target_hw=(rgb.shape[1], rgb.shape[2]),
            file_path=hsi_path,
        )
        hsi = normalize_cube(hsi, NORMALIZATION)

        rgb_tensor = (
            torch.from_numpy(np.ascontiguousarray(rgb))
            .float()
            .unsqueeze(0)
            .to(device)
        )
        target_tensor = (
            torch.from_numpy(np.ascontiguousarray(hsi))
            .float()
            .unsqueeze(0)
            .to(device)
        )

        if rgb_tensor.shape[-2:] != target_tensor.shape[-2:]:
            raise ValueError(
                f"Spatial mismatch for {hsi_path.stem}: RGB "
                f"{tuple(rgb_tensor.shape[-2:])}, HSI "
                f"{tuple(target_tensor.shape[-2:])}"
            )

        # Keep the stochastic reverse trajectory reproducible per image.
        sample_seed = SEED + pool_index
        torch.manual_seed(sample_seed)
        torch.cuda.manual_seed_all(sample_seed)

        refined, coarse = run_full_inference(
            model=model,
            rgb=rgb_tensor,
            model_downsample_factor=model_downsample_factor,
            num_steps=NUM_SAMPLING_STEPS,
            use_amp=use_amp,
        )

        if refined.shape != target_tensor.shape:
            raise RuntimeError(
                f"I2SB shape mismatch for {hsi_path.stem}: "
                f"target={tuple(target_tensor.shape)}, "
                f"prediction={tuple(refined.shape)}"
            )
        if coarse.shape != target_tensor.shape:
            raise RuntimeError(
                f"MST++ shape mismatch for {hsi_path.stem}: "
                f"target={tuple(target_tensor.shape)}, "
                f"prediction={tuple(coarse.shape)}"
            )
        if not torch.isfinite(refined).all():
            raise FloatingPointError(
                f"Non-finite I2SB output for {hsi_path.stem}"
            )
        if not torch.isfinite(coarse).all():
            raise FloatingPointError(
                f"Non-finite MST++ output for {hsi_path.stem}"
            )

        coarse_metrics = calculate_metrics(coarse, target_tensor)
        refined_metrics = calculate_metrics(refined, target_tensor)
        add_metrics(coarse_totals, coarse_metrics)
        add_metrics(refined_totals, refined_metrics)
        evaluated_count += 1

        if pool_index in selected_index_set:
            sample: Dict[str, Any] = {
                "index": pool_index,
                "stem": hsi_path.stem,
                "rgb": rgb_tensor[0].cpu().float().numpy(),
                "target": target_tensor[0].cpu().float().numpy(),
                "coarse": coarse[0].cpu().float().numpy(),
                "refined": refined[0].cpu().float().numpy(),
                "coarse_metrics": coarse_metrics,
                "refined_metrics": refined_metrics,
            }
            selected_samples[pool_index] = sample

            if SAVE_VISUALISED_CUBES:
                save_visualised_sample_cubes(
                    sample,
                    OUTPUT_DIR / "visualised_cubes",
                )

        if (
            EVALUATION_PRINT_EVERY > 0
            and (
                evaluated_count % EVALUATION_PRINT_EVERY == 0
                or evaluated_count == len(validation_pairs)
            )
        ):
            running_refined = average_metrics(
                refined_totals,
                evaluated_count,
            )
            print(
                f"[{evaluated_count:03d}/{len(validation_pairs):03d}] "
                f"{hsi_path.stem} | "
                f"I2SB running MRAE {running_refined['mrae']:.6f} | "
                f"PSNR {running_refined['psnr']:.4f}"
            )

    missing_selected = [
        index for index in selected_indices
        if index not in selected_samples
    ]
    if missing_selected:
        raise RuntimeError(
            "Selected visualization samples were not retained during "
            f"evaluation: {missing_selected}"
        )

    ordered_samples = [
        selected_samples[index]
        for index in selected_indices
    ]

    return (
        average_metrics(coarse_totals, evaluated_count),
        average_metrics(refined_totals, evaluated_count),
        ordered_samples,
    )



# =============================================================================
# Main
# =============================================================================


def main() -> None:
    set_seed(SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = USE_AMP and device.type == "cuda"

    checkpoint_path = Path(CHECKPOINT_PATH)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")

    all_pairs = find_paired_files(HSI_DATA_DIR, RGB_DATA_DIR)

    # Always reconstruct the exact deterministic validation split used during
    # training. No image from the training split can enter the inference pool.
    _training_pairs, validation_pairs = split_pairs(
        all_pairs,
        validation_fraction=VALIDATION_FRACTION,
        seed=SEED,
    )
    visualisation_pool = validation_pairs
    pool_name = "deterministic validation split"

    selected_pairs = choose_visualisation_pairs(visualisation_pool)

    model, checkpoint = build_model_from_checkpoint(CHECKPOINT_PATH, device)
    channel_mults = (
        tuple(
            checkpoint.get("model_config", {}).get(
                "channel_mults", UNET_CHANNEL_MULTIPLIERS
            )
        )
        if isinstance(checkpoint, Mapping)
        else UNET_CHANNEL_MULTIPLIERS
    )
    model_downsample_factor = 2 ** (len(channel_mults) - 1)

    print("\nVisualisation configuration")
    print(f"  Device: {device}")
    print(f"  Mixed precision: {use_amp}")
    print(f"  Evaluation pool: {pool_name} ({len(visualisation_pool)} images)")
    print("  Full validation evaluation: enabled")
    print("  Inference outside the validation split: disabled")
    print(f"  Selected images: {len(selected_pairs)}")
    print(f"  Sampling steps: {NUM_SAMPLING_STEPS}")
    print(f"  Tiled reverse sampling: {USE_TILED_REVERSE}")
    if USE_TILED_REVERSE:
        print(
            f"  Reverse tile: {REVERSE_TILE_SIZE}, "
            f"overlap: {REVERSE_TILE_OVERLAP}"
        )
    for pool_index, hsi_path, _rgb_path in selected_pairs:
        print(f"    [{pool_index}] {hsi_path.stem}")

    print("\nEvaluating the complete validation split...")
    coarse_validation_metrics, refined_validation_metrics, samples = (
        evaluate_validation_split(
            model=model,
            validation_pairs=validation_pairs,
            selected_pairs=selected_pairs,
            device=device,
            model_downsample_factor=model_downsample_factor,
            use_amp=use_amp,
        )
    )

    print("\n" + "=" * 80)
    print("FULL VALIDATION-SPLIT RESULTS")
    print("=" * 80)
    print(f"Images evaluated: {len(validation_pairs)}")
    print("MST++ coarse:")
    print(f"  {format_metrics(coarse_validation_metrics)}")
    print("I2SB full reverse sample:")
    print(f"  {format_metrics(refined_validation_metrics)}")
    print("=" * 80)

    figure_paths: List[Path] = []
    for sample in samples:
        saved_path = create_visualisation_figure(
            sample=sample,
            output_dir=OUTPUT_DIR,
        )
        if saved_path is not None:
            figure_paths.append(saved_path)

    print("\nCompleted evaluation and visualisations")
    print(f"  Validation images evaluated: {len(validation_pairs)}")
    print(f"  Images visualised: {len(samples)}")
    if figure_paths:
        print(f"  Figure directory: {OUTPUT_DIR.resolve()}")
    if SAVE_VISUALISED_CUBES:
        print(f"  Selected cubes: {(OUTPUT_DIR / 'visualised_cubes').resolve()}")


if __name__ == "__main__":
    main()
