"""Simple training and visualization script for MST++ + DGSolver.

The dataloader follows the paired RGB/HSI loader used in the supplied script.
MST++ is loaded and frozen inside ``MSTPlusPlusDGSolver``. The diffusion model
learns the residual

    coarse_hsi - ground_truth_hsi

and validation metrics are computed on the final second-order UPS sample.

Only ``--mode`` is exposed as a command-line argument. Edit every other setting
in the configuration section below.

Usage
-----
python train_mstpp_dgsolver.py --mode train
python train_mstpp_dgsolver.py --mode visualize
python train_mstpp_dgsolver.py --mode train_visualize
"""

from __future__ import annotations

import argparse
import math
import random
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import h5py
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

# -----------------------------------------------------------------------------
# Change this import path to wherever the previously created model file lives.
# -----------------------------------------------------------------------------
from model.dgsolver import MSTPlusPlusDGSolver

# Prefer the metric implementations already present in your project. A local
# fallback is provided so this file remains self-contained during early testing.
try:
    from loss.mrae import mrae as project_mrae
    from loss.psnr import psnr as project_psnr
    from loss.rmse import rmse as project_rmse
    from loss.sam import sam as project_sam
    from loss.ssim import ssim as project_ssim

    USING_PROJECT_METRICS = True
except ImportError:
    project_mrae = None
    project_psnr = None
    project_rmse = None
    project_sam = None
    project_ssim = None
    USING_PROJECT_METRICS = False


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

MST_CHECKPOINT = "./mst_checkpoints/mst_plus_plus.pth"
MST_MODEL_KWARGS: Dict[str, Any] = {}
STRICT_MST_LOADING = True

OUTPUT_DIR = Path("./mstpp_dgsolver_checkpoints")
BEST_CHECKPOINT = OUTPUT_DIR / "best_mstpp_dgsolver.pth"
LAST_CHECKPOINT = OUTPUT_DIR / "last_mstpp_dgsolver.pth"
RESUME_CHECKPOINT: Optional[str] = None

VISUALIZATION_CHECKPOINT = BEST_CHECKPOINT
VISUALIZATION_DIR = Path("./mstpp_dgsolver_visualizations")
VISUALIZATION_FILE = VISUALIZATION_DIR / "random_validation_examples.png"

HSI_KEY = "cube"
HSI_CHANNELS = 31
SUPPORTED_HSI_EXTENSIONS = {".npy", ".npz", ".mat", ".pt", ".pth"}
SUPPORTED_RGB_EXTENSIONS = {".png", ".jpg", ".jpeg", ".npy", ".pt", ".pth"}
HSI_NORMALIZATION = "none"  # "none", "minmax", or "band_minmax"

# DGSolver model settings.
IMAGE_SIZE = 256
BASE_DIM = 64
INIT_DIM: Optional[int] = None
DIM_MULTS = (1, 2, 4, 8)
NUM_DIFFUSION_TIMESTEPS = 50
SAMPLING_TIMESTEPS = 8
DELTA_END = 2.0e-3
SUM_SCALE = 0.01
LOSS_TYPE = "l1"
MODEL_PAD_MULTIPLE = 2 ** len(DIM_MULTS)

# Dataset and optimization settings.
TRAIN_CROP_SIZE = IMAGE_SIZE
VALIDATION_CROP_SIZE = IMAGE_SIZE
PATCHES_PER_IMAGE = 2
USE_AUGMENTATION = True
BATCH_SIZE = 2
VALIDATION_BATCH_SIZE = 1
NUM_EPOCHS = 10
LEARNING_RATE = 5.0e-5
WEIGHT_DECAY = 1.0e-4
MIN_LEARNING_RATE = 1.0e-7
GRADIENT_CLIP_NORM = 1.0
NUM_WORKERS = 4
USE_AMP = True
PREFER_BFLOAT16 = True
FP16_INITIAL_SCALE = 1024.0
FP16_GROWTH_INTERVAL = 2000
PRINT_EVERY = 30
SEED = 42

# Validation sampling can be expensive because UPS requires input gradients.
# Keep this as None to evaluate the complete validation set.
VALIDATION_MAX_IMAGES: Optional[int] = None

# Visualization settings.
NUM_VISUALIZATION_IMAGES = 5
VISUALIZATION_BANDS = (20, 10, 2)
FIGURE_DPI = 180


# =============================================================================
# Reproducibility and AMP helpers
# =============================================================================


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id: int) -> None:
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
        raise ValueError(f"No numeric 3D HSI dataset was found in {file_path}.")

    _, cube = max(candidates, key=lambda item: item[1].size)
    # MATLAB v7.3 arrays are commonly stored with reversed dimensions.
    return np.transpose(cube, axes=tuple(range(cube.ndim - 1, -1, -1)))


def load_hsi_file(file_path: Path) -> np.ndarray:
    extension = file_path.suffix.lower()

    if extension == ".npy":
        cube = np.load(file_path)
    elif extension == ".npz":
        with np.load(file_path) as loaded:
            candidates = [loaded[key] for key in loaded.files if loaded[key].ndim == 3]
            if not candidates:
                raise ValueError(f"No 3D array was found in {file_path}.")
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
            cube = load_mat_v73(file_path=file_path, preferred_key=HSI_KEY)
    elif extension in {".pt", ".pth"}:
        loaded = load_torch_checkpoint(file_path, device="cpu")
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
            raise TypeError(f"Unsupported object type in {file_path}: {type(loaded)}")
    else:
        raise ValueError(f"Unsupported HSI extension: {extension}")

    cube = np.asarray(cube, dtype=np.float32)
    cube = np.squeeze(cube)
    if cube.ndim != 3:
        raise ValueError(
            f"Expected a 3D HSI cube in {file_path}, but found {cube.shape}."
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
        f"Found {cube.shape}; expected {hsi_channels} bands."
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
        loaded = load_torch_checkpoint(file_path, device="cpu")
        if isinstance(loaded, torch.Tensor):
            array = loaded.detach().cpu().float().numpy()
        elif isinstance(loaded, np.ndarray):
            array = loaded.astype(np.float32)
        else:
            raise TypeError(f"Unsupported RGB object in {file_path}: {type(loaded)}")
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
            f"Could not convert RGB file {file_path} to CHW. Found {array.shape}."
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
# File discovery and pairing
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
                f"Duplicate {kind} stem '{path.stem}'.\n"
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
            "No paired HSI/RGB files were found. Files must have identical stems."
        )

    pairs = [(hsi_by_stem[stem], rgb_by_stem[stem]) for stem in shared_stems]
    print(
        f"Found {len(pairs)} pairs:\n"
        f"  HSI: {hsi_directory}\n"
        f"  RGB: {rgb_directory}"
    )
    return pairs


# =============================================================================
# Paired transforms and dataset
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
    return F.pad(tensor, (0, pad_width, 0, pad_height), mode="replicate")


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
        hsi[:, top:top + crop_size, left:left + crop_size],
        rgb[:, top:top + crop_size, left:left + crop_size],
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

    hsi = F.pad(hsi, (0, pad_width, 0, pad_height), mode="replicate")
    rgb = F.pad(rgb, (0, pad_width, 0, pad_height), mode="replicate")
    return hsi, rgb, original_height, original_width


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
            raise ValueError("Training requires a crop size.")
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
# Metric helpers
# =============================================================================


def fallback_mrae(target: torch.Tensor, prediction: torch.Tensor) -> torch.Tensor:
    return torch.mean(torch.abs(prediction - target) / (torch.abs(target) + 1e-6))


def fallback_rmse(target: torch.Tensor, prediction: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(torch.mean((prediction - target) ** 2) + 1e-12)


def fallback_psnr(target: torch.Tensor, prediction: torch.Tensor) -> torch.Tensor:
    mse = torch.mean((prediction - target) ** 2)
    data_range = torch.clamp(target.max() - target.min(), min=1e-6)
    return 20.0 * torch.log10(data_range) - 10.0 * torch.log10(mse.clamp_min(1e-12))


def fallback_sam(target: torch.Tensor, prediction: torch.Tensor) -> torch.Tensor:
    # Mean spectral angle in radians over all spatial pixels.
    dot = torch.sum(target * prediction, dim=1)
    target_norm = torch.linalg.vector_norm(target, dim=1)
    prediction_norm = torch.linalg.vector_norm(prediction, dim=1)
    cosine = dot / (target_norm * prediction_norm + 1e-8)
    cosine = torch.clamp(cosine, -1.0 + 1e-7, 1.0 - 1e-7)
    return torch.acos(cosine).mean()


def fallback_ssim(target: torch.Tensor, prediction: torch.Tensor) -> torch.Tensor:
    # Windowed channel-wise SSIM averaged over batch, bands and pixels.
    kernel_size = 11
    padding = kernel_size // 2
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2

    mu_x = F.avg_pool2d(target, kernel_size, stride=1, padding=padding)
    mu_y = F.avg_pool2d(prediction, kernel_size, stride=1, padding=padding)
    mu_x2 = mu_x.square()
    mu_y2 = mu_y.square()
    mu_xy = mu_x * mu_y

    sigma_x2 = (
        F.avg_pool2d(target.square(), kernel_size, stride=1, padding=padding)
        - mu_x2
    ).clamp_min(0.0)
    sigma_y2 = (
        F.avg_pool2d(prediction.square(), kernel_size, stride=1, padding=padding)
        - mu_y2
    ).clamp_min(0.0)
    sigma_xy = (
        F.avg_pool2d(target * prediction, kernel_size, stride=1, padding=padding)
        - mu_xy
    )

    numerator = (2.0 * mu_xy + c1) * (2.0 * sigma_xy + c2)
    denominator = (mu_x2 + mu_y2 + c1) * (sigma_x2 + sigma_y2 + c2)
    return (numerator / denominator.clamp_min(1e-12)).mean()


def _metric_to_float(value: Any, name: str) -> float:
    if not torch.is_tensor(value):
        value = torch.as_tensor(value)
    value = value.detach().float()
    if value.numel() != 1:
        value = value.mean()
    result = float(value.item())
    if not np.isfinite(result) and not (name == "PSNR" and result == float("inf")):
        raise FloatingPointError(f"{name} returned a non-finite value: {result}")
    return result


def calculate_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> Dict[str, float]:
    """Return sums over images; divide by image count outside this function."""
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

    metric_sums = {"mrae": 0.0, "rmse": 0.0, "sam": 0.0, "psnr": 0.0, "ssim": 0.0}

    for index in range(prediction.shape[0]):
        pred = prediction[index:index + 1]
        truth = target[index:index + 1]

        if USING_PROJECT_METRICS:
            values = {
                "mrae": project_mrae(truth, pred),
                "rmse": project_rmse(truth, pred),
                "sam": project_sam(truth, pred),
                "psnr": project_psnr(truth, pred),
                "ssim": project_ssim(truth, pred),
            }
        else:
            values = {
                "mrae": fallback_mrae(truth, pred),
                "rmse": fallback_rmse(truth, pred),
                "sam": fallback_sam(truth, pred),
                "psnr": fallback_psnr(truth, pred),
                "ssim": fallback_ssim(truth, pred),
            }

        for name, value in values.items():
            metric_sums[name] += _metric_to_float(value, name.upper())

    return metric_sums


# =============================================================================
# Model and checkpoint helpers
# =============================================================================


def load_torch_checkpoint(path: str | Path, device: str | torch.device = "cpu"):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def build_model(device: torch.device) -> MSTPlusPlusDGSolver:
    model = MSTPlusPlusDGSolver(
        mst_checkpoint=MST_CHECKPOINT,
        mst_kwargs=MST_MODEL_KWARGS,
        hsi_channels=HSI_CHANNELS,
        image_size=IMAGE_SIZE,
        dim=BASE_DIM,
        init_dim=INIT_DIM,
        dim_mults=DIM_MULTS,
        timesteps=NUM_DIFFUSION_TIMESTEPS,
        sampling_timesteps=SAMPLING_TIMESTEPS,
        delta_end=DELTA_END,
        sum_scale=SUM_SCALE,
        loss_type=LOSS_TYPE,
        freeze_mst=True,
        strict_mst_loading=STRICT_MST_LOADING,
    )
    return model.to(device)


def trainable_parameters(model: nn.Module) -> List[nn.Parameter]:
    return [parameter for parameter in model.parameters() if parameter.requires_grad]


def diffusion_state_dict(model: MSTPlusPlusDGSolver) -> Dict[str, torch.Tensor]:
    # MST++ is already stored in MST_CHECKPOINT; do not duplicate it here.
    return {
        key: value.detach().cpu()
        for key, value in model.diffusion.state_dict().items()
    }


def save_checkpoint(
    path: Path,
    model: MSTPlusPlusDGSolver,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: GradScaler,
    epoch: int,
    best_mrae: float,
    train_metrics: Dict[str, float],
    validation_metrics: Dict[str, float],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "diffusion_state_dict": diffusion_state_dict(model),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "best_mrae": best_mrae,
            "train_metrics": train_metrics,
            "validation_metrics": validation_metrics,
            "mst_checkpoint": MST_CHECKPOINT,
            "model_config": {
                "hsi_channels": HSI_CHANNELS,
                "image_size": IMAGE_SIZE,
                "dim": BASE_DIM,
                "init_dim": INIT_DIM,
                "dim_mults": DIM_MULTS,
                "timesteps": NUM_DIFFUSION_TIMESTEPS,
                "sampling_timesteps": SAMPLING_TIMESTEPS,
                "delta_end": DELTA_END,
                "sum_scale": SUM_SCALE,
                "loss_type": LOSS_TYPE,
            },
        },
        path,
    )


def load_diffusion_checkpoint(
    model: MSTPlusPlusDGSolver,
    checkpoint_path: str | Path,
) -> dict:
    checkpoint = load_torch_checkpoint(checkpoint_path, device="cpu")
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Checkpoint must be a dictionary: {checkpoint_path}")

    state_dict = checkpoint.get("diffusion_state_dict")
    if state_dict is None:
        # Convenient fallbacks for manually saved checkpoints.
        state_dict = checkpoint.get("model_state_dict", checkpoint.get("state_dict"))
    if not isinstance(state_dict, dict):
        raise KeyError("Could not find diffusion_state_dict/model_state_dict/state_dict.")

    model.diffusion.load_state_dict(state_dict, strict=True)
    return checkpoint


@contextmanager
def sampling_parameter_freeze(model: MSTPlusPlusDGSolver):
    """Disable parameter gradients while preserving gradients w.r.t. x_t for UPS."""
    parameters = list(model.diffusion.model.parameters())
    original_flags = [parameter.requires_grad for parameter in parameters]
    try:
        for parameter in parameters:
            parameter.requires_grad_(False)
        yield
    finally:
        for parameter, flag in zip(parameters, original_flags):
            parameter.requires_grad_(flag)


def sample_refinement(
    model: MSTPlusPlusDGSolver,
    rgb: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return frozen MST++ coarse HSI and DGSolver refined HSI.

    DGSolver UPS uses autograd with respect to the current diffusion state, so
    this helper deliberately enables gradients even during validation.
    """
    with sampling_parameter_freeze(model):
        with torch.enable_grad():
            outputs = model.sample(rgb.float(), return_coarse=True)
    coarse = outputs["coarse_hsi"].detach()
    refined = outputs["refined_hsi"].detach()
    return coarse, refined


# =============================================================================
# Training and validation
# =============================================================================


def train_one_epoch(
    model: MSTPlusPlusDGSolver,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
    use_amp: bool,
) -> Dict[str, float]:
    model.train()
    parameters = trainable_parameters(model)

    loss_sum = 0.0
    sample_count = 0

    progress = tqdm(loader, desc="Training", leave=False)
    for batch_index, (hsi, rgb) in enumerate(progress, start=1):
        hsi = hsi.to(device, non_blocking=True)
        rgb = rgb.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        with autocast_context(device, use_amp):
            outputs = model(rgb=rgb, ground_truth=hsi)
            loss = outputs["residual_loss"]

        if not torch.isfinite(loss):
            raise FloatingPointError(
                f"Non-finite training loss at batch {batch_index}: "
                f"{float(loss.detach())}"
            )

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        gradient_norm = nn.utils.clip_grad_norm_(
            parameters,
            max_norm=GRADIENT_CLIP_NORM,
            error_if_nonfinite=True,
        )
        scaler.step(optimizer)
        scaler.update()

        batch_size = hsi.shape[0]
        loss_sum += float(loss.detach()) * batch_size
        sample_count += batch_size

        if batch_index % PRINT_EVERY == 0 or batch_index == len(loader):
            progress.set_postfix(
                loss=f"{loss_sum / sample_count:.6f}",
                grad=f"{float(gradient_norm):.3f}",
            )

    if sample_count == 0:
        raise RuntimeError("Training DataLoader produced no samples.")
    return {"loss": loss_sum / sample_count}


def validate_one_epoch(
    model: MSTPlusPlusDGSolver,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool,
) -> Dict[str, float]:
    model.eval()

    loss_sum = 0.0
    refined_sums = {"mrae": 0.0, "rmse": 0.0, "sam": 0.0, "psnr": 0.0, "ssim": 0.0}
    coarse_sums = {"mrae": 0.0, "rmse": 0.0, "sam": 0.0, "psnr": 0.0, "ssim": 0.0}
    sample_count = 0

    progress = tqdm(loader, desc="Validation", leave=False)
    for hsi, rgb in progress:
        if VALIDATION_MAX_IMAGES is not None:
            remaining = VALIDATION_MAX_IMAGES - sample_count
            if remaining <= 0:
                break
            hsi = hsi[:remaining]
            rgb = rgb[:remaining]

        hsi = hsi.to(device, non_blocking=True)
        rgb = rgb.to(device, non_blocking=True)
        batch_size = hsi.shape[0]

        # Random-timestep residual-prediction validation loss.
        with torch.no_grad():
            with autocast_context(device, use_amp):
                outputs = model(rgb=rgb, ground_truth=hsi)
                residual_loss = outputs["residual_loss"]
        loss_sum += float(residual_loss.detach()) * batch_size

        # Final reconstruction metrics. Sampling is kept in float32 because UPS
        # computes gradients of its consistency norm.
        coarse, refined = sample_refinement(model, rgb)
        refined = refined.to(dtype=torch.float32)
        coarse = coarse.to(dtype=torch.float32)
        target = hsi.to(dtype=torch.float32)

        refined_batch = calculate_metrics(refined, target)
        coarse_batch = calculate_metrics(coarse, target)
        for key in refined_sums:
            refined_sums[key] += refined_batch[key]
            coarse_sums[key] += coarse_batch[key]

        sample_count += batch_size
        progress.set_postfix(
            loss=f"{loss_sum / sample_count:.6f}",
            mrae=f"{refined_sums['mrae'] / sample_count:.5f}",
        )

    if sample_count == 0:
        raise RuntimeError("Validation DataLoader produced no samples.")

    metrics: Dict[str, float] = {
        "loss": loss_sum / sample_count,
        "evaluated_images": float(sample_count),
    }
    for key in refined_sums:
        metrics[key] = refined_sums[key] / sample_count
        metrics[f"coarse_{key}"] = coarse_sums[key] / sample_count
    return metrics


def prepare_pairs() -> Tuple[List[Tuple[Path, Path]], List[Tuple[Path, Path]]]:
    train_pairs = pair_hsi_rgb_files(TRAIN_HSI_DIR, TRAIN_RGB_DIR)
    validation_pairs = pair_hsi_rgb_files(
        VALIDATION_HSI_DIR,
        VALIDATION_RGB_DIR,
    )
    return train_pairs, validation_pairs


def prepare_validation_pairs() -> List[Tuple[Path, Path]]:
    return pair_hsi_rgb_files(VALIDATION_HSI_DIR, VALIDATION_RGB_DIR)


def run_training(
    train_pairs: Sequence[Tuple[Path, Path]],
    validation_pairs: Sequence[Tuple[Path, Path]],
    device: torch.device,
    use_amp: bool,
) -> None:
    if TRAIN_CROP_SIZE % MODEL_PAD_MULTIPLE != 0:
        raise ValueError(
            f"TRAIN_CROP_SIZE={TRAIN_CROP_SIZE} must be divisible by "
            f"MODEL_PAD_MULTIPLE={MODEL_PAD_MULTIPLE}."
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
        drop_last=(len(train_dataset) >= BATCH_SIZE),
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
    parameters = trainable_parameters(model)
    optimizer = torch.optim.AdamW(
        parameters,
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=NUM_EPOCHS,
        eta_min=MIN_LEARNING_RATE,
    )

    amp_dtype = get_amp_dtype(device)
    scaler = GradScaler(
        enabled=(use_amp and amp_dtype == torch.float16),
        init_scale=FP16_INITIAL_SCALE,
        growth_interval=FP16_GROWTH_INTERVAL,
    )

    start_epoch = 1
    best_mrae = float("inf")

    if RESUME_CHECKPOINT is not None:
        checkpoint = load_diffusion_checkpoint(model, RESUME_CHECKPOINT)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if "scaler_state_dict" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler_state_dict"])
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        best_mrae = float(checkpoint.get("best_mrae", float("inf")))
        print(f"Resumed from {RESUME_CHECKPOINT} at epoch {start_epoch}.")

    frozen_count = sum(parameter.numel() for parameter in model.mst_plus_plus.parameters())
    trainable_count = sum(parameter.numel() for parameter in parameters)
    print(
        f"\nDevice: {device}\n"
        f"AMP: {use_amp} ({amp_dtype if use_amp else 'float32'})\n"
        f"Metric source: {'project loss modules' if USING_PROJECT_METRICS else 'local fallback'}\n"
        f"Training pairs: {len(train_pairs)}\n"
        f"Validation pairs: {len(validation_pairs)}\n"
        f"Frozen MST++ parameters: {frozen_count:,}\n"
        f"Trainable DGSolver parameters: {trainable_count:,}"
    )

    for epoch in range(start_epoch, NUM_EPOCHS + 1):
        print(f"\n{'=' * 80}\nEpoch {epoch}/{NUM_EPOCHS}\n{'=' * 80}")

        train_metrics = train_one_epoch(
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
            f"Epoch {epoch:03d} | LR={optimizer.param_groups[0]['lr']:.2e} | "
            f"train residual={train_metrics['loss']:.6f} | "
            f"val residual={validation_metrics['loss']:.6f}"
        )
        print(
            "Refined validation metrics "
            f"({int(validation_metrics['evaluated_images'])} images) | "
            f"MRAE={validation_metrics['mrae']:.6f} | "
            f"RMSE={validation_metrics['rmse']:.6f} | "
            f"SAM={validation_metrics['sam']:.6f} | "
            f"PSNR={validation_metrics['psnr']:.4f} | "
            f"SSIM={validation_metrics['ssim']:.4f}"
        )
        print(
            "Frozen MST++ baseline | "
            f"MRAE={validation_metrics['coarse_mrae']:.6f} | "
            f"RMSE={validation_metrics['coarse_rmse']:.6f} | "
            f"SAM={validation_metrics['coarse_sam']:.6f} | "
            f"PSNR={validation_metrics['coarse_psnr']:.4f} | "
            f"SSIM={validation_metrics['coarse_ssim']:.4f}"
        )

        save_checkpoint(
            path=LAST_CHECKPOINT,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            epoch=epoch,
            best_mrae=best_mrae,
            train_metrics=train_metrics,
            validation_metrics=validation_metrics,
        )

        if validation_metrics["mrae"] < best_mrae:
            best_mrae = validation_metrics["mrae"]
            save_checkpoint(
                path=BEST_CHECKPOINT,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                epoch=epoch,
                best_mrae=best_mrae,
                train_metrics=train_metrics,
                validation_metrics=validation_metrics,
            )
            print(
                f"Saved new best checkpoint: {BEST_CHECKPOINT} | "
                f"MRAE={best_mrae:.6f}"
            )


# =============================================================================
# Five-image full-resolution visualization
# =============================================================================


def rgb_tensor_to_display(rgb: torch.Tensor) -> np.ndarray:
    array = rgb.detach().float().cpu().numpy().transpose(1, 2, 0)
    return np.clip(array, 0.0, 1.0)


def hsi_triplet_to_display(
    target: torch.Tensor,
    coarse: torch.Tensor,
    refined: torch.Tensor,
    bands: Tuple[int, int, int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    target_np = target.detach().float().cpu().numpy()
    coarse_np = coarse.detach().float().cpu().numpy()
    refined_np = refined.detach().float().cpu().numpy()

    for band in bands:
        if not 0 <= band < target_np.shape[0]:
            raise ValueError(
                f"Visualization band {band} is outside [0, {target_np.shape[0] - 1}]."
            )

    def select(cube: np.ndarray) -> np.ndarray:
        return np.stack([cube[band] for band in bands], axis=-1)

    target_rgb = select(target_np)
    coarse_rgb = select(coarse_np)
    refined_rgb = select(refined_np)

    # Use one target-derived scale for all three HSI panels.
    minimum = target_rgb.min(axis=(0, 1), keepdims=True)
    maximum = target_rgb.max(axis=(0, 1), keepdims=True)
    scale = maximum - minimum + 1e-8

    return (
        np.clip((target_rgb - minimum) / scale, 0.0, 1.0),
        np.clip((coarse_rgb - minimum) / scale, 0.0, 1.0),
        np.clip((refined_rgb - minimum) / scale, 0.0, 1.0),
    )


def metric_text(metrics: Dict[str, float]) -> str:
    return (
        f"MRAE {metrics['mrae']:.4f} | RMSE {metrics['rmse']:.4f}\n"
        f"SAM {metrics['sam']:.4f} | PSNR {metrics['psnr']:.2f} | "
        f"SSIM {metrics['ssim']:.4f}"
    )


def load_model_for_visualization(
    checkpoint_path: str | Path,
    device: torch.device,
) -> MSTPlusPlusDGSolver:
    model = build_model(device)
    load_diffusion_checkpoint(model, checkpoint_path)
    model.eval()
    print(f"Loaded DGSolver checkpoint: {checkpoint_path}")
    return model


def run_visualization(
    model: MSTPlusPlusDGSolver,
    validation_pairs: Sequence[Tuple[Path, Path]],
    device: torch.device,
) -> Path:
    model.eval()
    if not validation_pairs:
        raise RuntimeError("Validation pair list is empty.")

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
        "Frozen MST++ coarse HSI\n(pseudo-RGB)",
        "DGSolver-refined HSI\n(pseudo-RGB)",
    )
    figure, axes = plt.subplots(
        number_to_select,
        4,
        figsize=(17, 4.5 * number_to_select),
        squeeze=False,
    )

    for row, dataset_index in enumerate(selected_indices):
        hsi, rgb, hsi_path_string, _ = dataset[dataset_index]
        padded_hsi, padded_rgb, original_height, original_width = pad_pair_to_multiple(
            hsi=hsi,
            rgb=rgb,
            multiple=MODEL_PAD_MULTIPLE,
        )

        rgb_batch = padded_rgb.unsqueeze(0).to(device)
        target_batch = padded_hsi.unsqueeze(0).to(device)
        coarse_batch, refined_batch = sample_refinement(model, rgb_batch)

        coarse = coarse_batch[0, :, :original_height, :original_width].cpu()
        refined = refined_batch[0, :, :original_height, :original_width].cpu()
        target = target_batch[0, :, :original_height, :original_width].cpu()
        rgb_original = rgb[:, :original_height, :original_width]

        rgb_display = rgb_tensor_to_display(rgb_original)
        target_display, coarse_display, refined_display = hsi_triplet_to_display(
            target=target,
            coarse=coarse,
            refined=refined,
            bands=VISUALIZATION_BANDS,
        )

        coarse_metrics = calculate_metrics(
            coarse.unsqueeze(0),
            target.unsqueeze(0),
        )
        refined_metrics = calculate_metrics(
            refined.unsqueeze(0),
            target.unsqueeze(0),
        )
        stem = Path(hsi_path_string).stem

        panels = (rgb_display, target_display, coarse_display, refined_display)
        for column, panel in enumerate(panels):
            axis = axes[row, column]
            axis.imshow(panel)
            axis.axis("off")
            if row == 0:
                axis.set_title(column_titles[column], fontsize=12, fontweight="bold")

        axes[row, 0].set_ylabel(
            stem,
            fontsize=10,
            rotation=0,
            labelpad=55,
            va="center",
            fontweight="bold",
        )
        axes[row, 2].text(
            0.5,
            -0.08,
            metric_text(coarse_metrics),
            transform=axes[row, 2].transAxes,
            ha="center",
            va="top",
            fontsize=8,
        )
        axes[row, 3].text(
            0.5,
            -0.08,
            metric_text(refined_metrics),
            transform=axes[row, 3].transAxes,
            ha="center",
            va="top",
            fontsize=8,
        )

    figure.suptitle(
        "Random full-resolution validation examples: MST++ coarse estimate and DGSolver refinement",
        fontsize=16,
        fontweight="bold",
        y=0.997,
    )
    figure.tight_layout(rect=(0.05, 0.02, 1.0, 0.985), h_pad=3.0)

    VISUALIZATION_DIR.mkdir(parents=True, exist_ok=True)
    figure.savefig(VISUALIZATION_FILE, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved visualization to: {VISUALIZATION_FILE}")
    return VISUALIZATION_FILE


# =============================================================================
# Mode parser and main
# =============================================================================


def parse_mode() -> str:
    parser = argparse.ArgumentParser(
        description="Train or visualize the frozen-MST++ DGSolver model."
    )
    parser.add_argument(
        "--mode",
        required=True,
        choices=("train", "visualize", "train_visualize"),
        help=(
            "train: train only; visualize: load VISUALIZATION_CHECKPOINT and "
            "render five random validation images; train_visualize: train and "
            "then visualize the best checkpoint."
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
        train_pairs, validation_pairs = prepare_pairs()
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
            BEST_CHECKPOINT if mode == "train_visualize"
            else VISUALIZATION_CHECKPOINT
        )
        model = load_model_for_visualization(checkpoint_path, device)
        run_visualization(
            model=model,
            validation_pairs=validation_pairs,
            device=device,
        )


if __name__ == "__main__":
    main()
