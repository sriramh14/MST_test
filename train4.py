"""Full-resolution training and inference for a frozen MST++ + ResShift pipeline.

The pretrained MST++ model produces a coarse HSI and remains completely frozen.
A ResShift diffusion model transfers the residual to refine the image.

Modules for ResShift are imported from the external `reshift_mst.py` file.

Edit the configuration section, then run one of:
    python train_mstpp_resshift.py --mode train
    python train_mstpp_resshift.py --mode infer
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

# Import the base MST++ model
from model.MST_Plus_Plus import MST_Plus_Plus

# Import the ResShift modules from your separate file
# Note: Ensure the filename matches your actual ResShift file (e.g., reshift_mst.py)
from resshift import ResShiftDenoiser, MSTPlusPlusResShift

# ============================================================
# Configuration
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

# Frozen MST++ fallback settings
MSTPP_STAGES = 3
MSTPP_FEATURES = 31

# ResShift settings
RESSHIFT_T = 15
RESSHIFT_P = 0.3
RESSHIFT_KAPPA = 2.0
RESSHIFT_FEATURES = 31
RESSHIFT_BODY_DEPTH = 3
RESSHIFT_MST_STAGE = 2
RESSHIFT_NUM_BLOCKS = (1, 1, 1)

# Training settings
EPOCHS = 75
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

# ResShift Loss weights
RECONSTRUCTION_LOSS_WEIGHT = 1.0
SPECTRAL_LOSS_WEIGHT = 0.1

VALIDATION_MODE = "endpoint"    # "endpoint" or "sample"
VALIDATION_STOCHASTIC = False

# Stable metric settings
MRAE_EPSILON = 1e-3
SAM_EPSILON = 1e-8
METRIC_DATA_RANGE = 1.0
SSIM_WINDOW_SIZE = 11
SSIM_SIGMA = 1.5
REPORT_SAM_IN_DEGREES = False
WARN_ON_RANGE_MISMATCH = True

RESUME_CHECKPOINT: Optional[str] = None

# Inference settings
INFERENCE_RESSHIFT_CHECKPOINT = "./mstpp_resshift_checkpoints/best_resshift.pth"
INFERENCE_OUTPUT_DIR = "./mstpp_resshift_results"
CLAMP_INFERENCE_OUTPUT = True
INFERENCE_STOCHASTIC = False
HEATMAP_REDUCTION = "mae"       

# Fast corrupt-file filtering.
VALIDATION_CACHE = Path(OUTPUT_DIR) / "hsi_validation_cache.pth"
INVALID_FILE_LOG = Path(OUTPUT_DIR) / "invalid_hsi_files.txt"
FORCE_REVALIDATE = False

HSI_EXTENSIONS = {".mat", ".npy", ".npz", ".pt", ".pth"}
RGB_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".npy", ".pt", ".pth"}


# ============================================================
# General utilities and File Pairings
# ============================================================

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def extract_cube(data: dict, path: Path) -> np.ndarray:
    if HSI_KEY in data:
        value = data[HSI_KEY]
        if isinstance(value, torch.Tensor): value = value.detach().cpu().numpy()
        if isinstance(value, np.ndarray) and value.ndim == 3: return value
    candidates = [v.detach().cpu().numpy() if isinstance(v, torch.Tensor) else v 
                  for k, v in data.items() if not str(k).startswith("__") and isinstance(v, (torch.Tensor, np.ndarray))]
    candidates = [c for c in candidates if c.ndim == 3]
    if not candidates: raise ValueError(f"No three-dimensional HSI cube found in {path}")
    return max(candidates, key=lambda array: array.size)

def load_hdf5_mat(path: Path) -> np.ndarray:
    candidates = []
    with h5py.File(str(path), "r") as file:
        file.visititems(lambda n, o: candidates.append(np.asarray(o)) if isinstance(o, h5py.Dataset) and o.ndim == 3 else None)
    if not candidates: raise ValueError(f"No three-dimensional HSI cube found in {path}")
    cube = max(candidates, key=lambda array: array.size)
    return np.transpose(cube, tuple(range(cube.ndim - 1, -1, -1)))

def load_hsi(path: Path) -> np.ndarray:
    ext = path.suffix.lower()
    if ext == ".npy": cube = np.load(path)
    elif ext == ".npz":
        loaded = np.load(path)
        cube = max([loaded[k] for k in loaded.files if loaded[k].ndim == 3], key=lambda a: a.size)
    elif ext == ".mat":
        try: cube = extract_cube(sio.loadmat(path), path)
        except (NotImplementedError, ValueError): cube = load_hdf5_mat(path)
    elif ext in {".pt", ".pth"}:
        loaded = torch.load(path, map_location="cpu")
        if isinstance(loaded, dict): cube = extract_cube(loaded, path)
        else: cube = loaded.detach().cpu().numpy() if isinstance(loaded, torch.Tensor) else loaded
    else: raise ValueError(f"Unsupported HSI extension: {ext}")
    
    cube = np.asarray(cube, dtype=np.float32).squeeze()
    if cube.shape[0] == HSI_CHANNELS: pass
    elif cube.shape[-1] == HSI_CHANNELS: cube = cube.transpose(2, 0, 1)
    elif cube.shape[1] == HSI_CHANNELS: cube = cube.transpose(1, 0, 2)
    return np.ascontiguousarray(cube)

def load_rgb(path: Path) -> np.ndarray:
    ext = path.suffix.lower()
    if ext in RGB_EXTENSIONS and ext not in {".npy", ".pt", ".pth"}:
        image = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
        return np.ascontiguousarray(image.transpose(2, 0, 1))
    image = np.load(path) if ext == ".npy" else torch.load(path, map_location="cpu").numpy()
    image = np.asarray(image, dtype=np.float32).squeeze()
    if image.shape[-1] == 3: image = image.transpose(2, 0, 1)
    if image.max() > 1.0: image = image / 255.0
    return np.ascontiguousarray(image)

def normalize_hsi(cube: np.ndarray) -> np.ndarray:
    if NORMALIZATION == "none": return cube
    if NORMALIZATION == "minmax": return (cube - cube.min()) / (cube.max() - cube.min() + 1e-8)
    raise ValueError(f"Unknown NORMALIZATION: {NORMALIZATION}")

def build_pairs() -> List[Tuple[Path, Path]]:
    rgb_by_stem = {p.stem: p for p in Path(RGB_DATA_DIR).rglob("*") if p.is_file() and p.suffix.lower() in RGB_EXTENSIONS}
    pairs = [(p, rgb_by_stem[p.stem]) for p in sorted(Path(HSI_DATA_DIR).rglob("*")) if p.is_file() and p.suffix.lower() in HSI_EXTENSIONS and p.stem in rgb_by_stem]
    if not pairs: raise RuntimeError("No paired files found.")
    return pairs

# ============================================================
# Dataset & Metrics
# ============================================================

class RGBHSIDataset(Dataset):
    def __init__(self, pairs: List[Tuple[Path, Path]], training: bool):
        self.pairs = pairs
        self.training = training

    def __len__(self) -> int: return len(self.pairs)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        hsi_path, rgb_path = self.pairs[index]
        rgb = torch.from_numpy(load_rgb(rgb_path)).float()
        hsi = torch.from_numpy(normalize_hsi(load_hsi(hsi_path))).float()
        if self.training and USE_AUGMENTATION:
            if random.random() < 0.5: rgb, hsi = torch.flip(rgb, [1]), torch.flip(hsi, [1])
            if random.random() < 0.5: rgb, hsi = torch.flip(rgb, [2]), torch.flip(hsi, [2])
            rots = random.randint(0, 3)
            if rots: rgb, hsi = torch.rot90(rgb, rots, [1, 2]), torch.rot90(hsi, rots, [1, 2])
        return rgb.contiguous(), hsi.contiguous()

def stable_mrae_per_sample(p: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    return ((p.float() - t.float()).abs() / t.float().abs().clamp_min(MRAE_EPSILON)).mean(dim=(1, 2, 3))

def calculate_metric_tensors(p: torch.Tensor, t: torch.Tensor) -> Dict[str, torch.Tensor]:
    mse = (p.float() - t.float()).square().mean(dim=(1, 2, 3))
    dot = (p * t).sum(dim=1)
    norm = p.square().sum(dim=1).sqrt() * t.square().sum(dim=1).sqrt()
    angle = torch.acos((dot / norm.clamp_min(SAM_EPSILON)).clamp(-1, 1))
    
    return {
        "mrae": stable_mrae_per_sample(p, t),
        "rmse": mse.sqrt(),
        "sam": angle.sum(dim=(1, 2)) / (norm > SAM_EPSILON).sum(dim=(1, 2)).clamp_min(1),
        "psnr": 10.0 * torch.log10((METRIC_DATA_RANGE ** 2) / mse.clamp_min(1e-12))
    }

class ResidualHeatmap(nn.Module):
    def __init__(self, reduction="mae"):
        super().__init__()
        self.reduction = reduction
    def forward(self, p: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        res = p - t
        return res.square().mean(dim=1, keepdim=True).sqrt() if self.reduction == "rmse" else res.abs().mean(dim=1, keepdim=True)

# ============================================================
# Training / Evaluation Loops
# ============================================================

def build_pipeline(device: torch.device) -> MSTPlusPlusResShift:
    # Initialize the base frozen MST++ model
    coarse = MST_Plus_Plus(3, HSI_CHANNELS, MSTPP_FEATURES, MSTPP_STAGES)
    
    # Load weights into the frozen base model
    ckpt = torch.load(MSTPP_CHECKPOINT, map_location="cpu")
    state = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    coarse.load_state_dict({k.replace("module.", ""): v for k,v in state.items()}, strict=True)
    
    # Initialize imported ResShift components
    denoiser = ResShiftDenoiser(
        HSI_CHANNELS, 3, RESSHIFT_FEATURES, RESSHIFT_BODY_DEPTH, 
        RESSHIFT_MST_STAGE, RESSHIFT_NUM_BLOCKS
    )
    
    # Wrap in imported MST++ ResShift architecture
    model = MSTPlusPlusResShift(coarse, denoiser, RESSHIFT_T, RESSHIFT_P, RESSHIFT_KAPPA)
    return model.to(device)

def train_one_epoch(model, loader, optimizer, scaler, device):
    model.train()
    total_loss, total_mrae = 0.0, 0.0
    
    for b_idx, (rgb, hsi) in enumerate(loader):
        rgb, hsi = rgb.to(device), hsi.to(device)
        optimizer.zero_grad()
        
        with torch.amp.autocast(device_type=device.type, enabled=USE_AMP):
            out = model(rgb, hsi)
            
        with torch.amp.autocast(device_type=device.type, enabled=False):
            recon_loss = F.l1_loss(out["predicted_x0"].float(), out["ground_truth"].float())
            loss = RECONSTRUCTION_LOSS_WEIGHT * recon_loss
            
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
        mrae = stable_mrae_per_sample(out["predicted_x0"], hsi).mean().item()
        total_mrae += mrae
        
        if b_idx % PRINT_EVERY == 0:
            print(f"  Train Batch {b_idx}/{len(loader)} | Loss: {loss.item():.4f} | MRAE: {mrae:.4f}")
            
    return total_loss / len(loader), total_mrae / len(loader)

def train():
    set_seed(SEED)
    device = get_device()
    
    pairs = build_pairs()
    random.shuffle(pairs)
    val_split = max(1, int(len(pairs) * VALIDATION_FRACTION))
    train_loader = DataLoader(RGBHSIDataset(pairs[val_split:], True), batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    
    model = build_pipeline(device)
    optimizer = torch.optim.AdamW(model.denoiser.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scaler = torch.amp.GradScaler('cuda', enabled=USE_AMP)
    
    best_mrae = float('inf')
    Path(OUTPUT_DIR).mkdir(exist_ok=True)
    
    for epoch in range(1, EPOCHS + 1):
        print(f"\nEpoch {epoch}/{EPOCHS}")
        l, m = train_one_epoch(model, train_loader, optimizer, scaler, device)
        print(f"Epoch Summary -> Loss: {l:.4f} | MRAE: {m:.4f}")
        
        if m < best_mrae:
            best_mrae = m
            torch.save({"denoiser": model.denoiser.state_dict(), "epoch": epoch}, Path(OUTPUT_DIR) / "best_resshift.pth")
            print("Saved Best Checkpoint!")

# ============================================================
# Inference Routine (Selecting 5 Random images)
# ============================================================

@torch.no_grad()
def infer():
    device = get_device()
    model = build_pipeline(device)
    
    ckpt = torch.load(INFERENCE_RESSHIFT_CHECKPOINT, map_location="cpu")
    model.denoiser.load_state_dict(ckpt["denoiser"])
    model.eval()
    
    pairs = build_pairs()
    sample_pairs = random.sample(pairs, min(5, len(pairs)))
    
    save_dir = Path(INFERENCE_OUTPUT_DIR)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    heatmap_module = ResidualHeatmap(HEATMAP_REDUCTION).to(device)

    print(f"Randomly selected {len(sample_pairs)} images for ResShift Inference.")
    for rgb_path, hsi_path in sample_pairs:
        stem = rgb_path.stem
        print(f"\nProcessing: {stem}")
        
        rgb = torch.from_numpy(load_rgb(rgb_path)).unsqueeze(0).to(device)
        target = torch.from_numpy(normalize_hsi(load_hsi(hsi_path))).unsqueeze(0).to(device)
        
        with torch.amp.autocast(device_type=device.type, enabled=USE_AMP):
            coarse_hsi, refined_hsi = model.sample(rgb, clip_denoised=CLAMP_INFERENCE_OUTPUT)
            
        c_m = calculate_metric_tensors(coarse_hsi.float(), target.float())
        r_m = calculate_metric_tensors(refined_hsi.float(), target.float())
        
        print(f"  Coarse MST++ | MRAE: {c_m['mrae'].mean().item():.4f} | PSNR: {c_m['psnr'].mean().item():.4f}")
        print(f"  Refined ResShift | MRAE: {r_m['mrae'].mean().item():.4f} | PSNR: {r_m['psnr'].mean().item():.4f}")
        
        # Save Heatmaps
        heatmap = heatmap_module(refined_hsi.float(), target.float())
        array = heatmap[0, 0].cpu().numpy()
        plt.figure(figsize=(6, 5))
        img = plt.imshow(array, cmap="inferno")
        plt.colorbar(img, label="Residual magnitude")
        plt.title(f"ResShift Refined MAE Heatmap: {stem}")
        plt.axis("off")
        plt.savefig(save_dir / f"{stem}_heatmap.png", dpi=150, bbox_inches="tight")
        plt.close()
        
        np.save(save_dir / f"{stem}_refined.npy", refined_hsi[0].cpu().numpy())

def parse_mode() -> str:
    parser = argparse.ArgumentParser("ResShift wrapper for frozen MST++")
    parser.add_argument("--mode", choices=["train", "infer"], required=True)
    return parser.parse_args().mode

if __name__ == "__main__":
    if parse_mode() == "train":
        train()
    else:
        infer()
