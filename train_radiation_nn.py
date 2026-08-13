"""
train_radiation_nn.py

Trains a neural network emulator for the medium-induced radiation intensity
distribution using precomputed training data.

Features:
- Loads training data from HDF5 file
- Normalizes inputs and log-transforms outputs
- Uses importance sampling weights for unbiased training
- Enforces soft physics constraints
- Saves trained model for deployment
- Demonstrates batch inference

Usage:
    python train_radiation_nn.py                    # Train the model
    python train_radiation_nn.py --inference-only   # Demo inference with saved model
"""

import argparse
import math
import numpy as np
import h5py
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from dataclasses import dataclass
from typing import Optional, Tuple, Dict
import json
import signal
import os
import time
from pathlib import Path

# Set use of float32 for matmuls to improve performance. Trial feature -- may impact accuracy, depending on problem.
torch.set_float32_matmul_precision('high')
torch.serialization.add_safe_globals([dict])  # save to load a nested dictionary with weights_only

def _collate_passthrough(x):
    """Passthrough collate function for DataLoader. Must be module-level to be picklable."""
    return x

# ==============================================================================
# Configuration
# ==============================================================================
@dataclass
class TrainingConfig:
    """
    Configuration for neural network training.
    """

    # Data
    data_file: str = "data/radiation_training_data.h5"
    train_fraction: float = 0.8
    transform: str = "arcsinh"
    transform_f0: float | None = None  # Scale for transformation, if applicable. None uses std of y_data.

    # Architecture
    hidden_dim: int = 256
    n_layers: int = 5
    activation: str = "silu"  # silu, relu, tanh, gelu

    # Training
    batch_size: int = 4096
    learning_rate: float = 2.72e-3
    weight_decay: float = 1e-4
    dropout_p: float = 0.1
    n_epochs: int = 500
    patience: int = 20  # Early stopping patience

    # Physics constraints
    lambda_uv: float = 0.0  # Weight for UV decay loss term
    lambda_uv_power: float = 0.0  # Weight for UV power-law loss term
    # UV threshold: only penalise points where kt^2 > uv_kt2_threshold (in GeV^2).
    # Should be set comfortably above mu_D^2 ~ g^2 T^2 ~ (2*GeV)^2*(0.3GeV)^2 ~ 0.36 GeV^2.
    # A safe default is 10.0 GeV^2 (kperp > ~3.16 GeV).
    uv_kt_threshold: float = 10.0

    # Output
    model_file: str = "data/radiation_emulator.pt"
    normalization_file: str = "data/radiation_normalization.json"
    checkpoint_file: str = "data/radiation_training_checkpoint.pt"  # Resume checkpoint
    checkpoint_interval: int = 1  # Save resume checkpoint every N epochs

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers: int = 4  # DataLoader worker processaes
    compile: bool = False  # Whether or not to compile the NN during training. Will not work on older Discovery GPUs

    # Utilities
    run_lr_finder: bool = False


# ==============================================================================
# Dataset
# ==============================================================================
class RadiationDataset(Dataset):
    """
    RAM-backed dataset. All valid data is loaded into memory once at construction.

    Design
    ------
    All HDF5 I/O happens in __init__ via _scan_file. After construction, no
    disk access occurs. __getitem__ is a pure array index + lightweight
    normalization — essentially free compared to HDF5 scalar reads.

    Memory layout
    -------------
    self.X_data : np.ndarray, shape (N_valid, 9), float32   -- input features
    self.y_data : np.ndarray, shape (N_valid,),   float32   -- raw intensity
    self.w_data : np.ndarray, shape (N_valid,),   float32   -- importance weights
    self.phi_data : np.ndarray, shape (N_valid,), float32   -- phi values for loss computation
    """

    FEATURE_NAMES = ['x', 'k_perp', 'E', 'z0', 'u_perp', 'T', 'g']
    RAW_FEATURE_NAMES = ['x', 'kx', 'ky', 'E', 'z0', 'u_perp', 'T', 'g']
    N_FEATURES = len(FEATURE_NAMES)

    def __init__(
            self,
            data_file: str,
            transform_output: str = "arcsinh",
    ):
        self.data_file = data_file
        self.transform = transform_output
        self.epsilon = 1e-10

        print(f"Loading {data_file} into RAM ...")
        self._scan_file()

        print(f"Dataset ready  |  valid={self.n_valid:,}  transform={self.transform}")

    # ------------------------------------------------------------------
    # Full load into RAM — no copies, no HDF5 handles kept open
    # ------------------------------------------------------------------
    def _scan_file(self):
        with h5py.File(self.data_file, 'r') as f:
            n_raw = int(f['I'].shape[0])
            print(f"  Reading {n_raw:,} rows from HDF5 ...")

            # Read raw (unmodified) columns straight from disk
            # Allocate arrays, then use h5py to read into them
            raw_cols = {}
            for name in self.RAW_FEATURE_NAMES:
                buf = np.empty(n_raw, dtype=np.float32)
                f[name].read_direct(buf)
                raw_cols[name] = buf

            y_raw = np.empty(n_raw, dtype=np.float32)
            f['I'].read_direct(y_raw)

            w_raw = np.empty(n_raw, dtype=np.float32)
            f['weight'].read_direct(w_raw)

            i_err = np.empty(n_raw, dtype=np.float32)
            f['I_err'].read_direct(i_err)

        # Compute k_perp and phi from kx, ky. phi is kept for loss reconstruction
        # only — it is never part of the network input.
        kx_raw = raw_cols['kx']
        ky_raw = raw_cols['ky']
        k_perp_raw = np.sqrt(kx_raw ** 2 + ky_raw ** 2).astype(np.float32)
        phi_raw = np.arctan2(ky_raw, kx_raw).astype(np.float32)

        # Stack the raw inputs
        X_raw = np.column_stack([
            raw_cols['x'], k_perp_raw, raw_cols['E'], raw_cols['z0'],
            raw_cols['u_perp'], raw_cols['T'], raw_cols['g'],
        ]).astype(np.float32)

        # Mask off any points where the integration when awry
        print("  Filtering invalid rows ...")
        ok = (
                np.isfinite(y_raw)
                & np.isfinite(i_err)
                & np.isfinite(phi_raw)
                & np.all(np.isfinite(X_raw), axis=1)
        )
        del i_err  # Trash the integration error to save memory

        n_valid = int(ok.sum())
        print(f"  {n_raw:,} raw  →  {n_valid:,} valid")

        # Apply mask to cut arrays
        self.X_data = X_raw[ok]
        self.phi_data = phi_raw[ok]
        del X_raw, raw_cols, kx_raw, ky_raw, k_perp_raw, phi_raw

        self.y_data = y_raw[ok]
        del y_raw

        self.w_data = w_raw[ok]
        del w_raw
        del ok

        self.n_valid = n_valid

        # Compute and print some normalization statistics
        print("  Computing normalization statistics ...")
        X64 = self.X_data.astype(np.float64)
        self.X_mean = X64.mean(axis=0).astype(np.float32)
        self.X_std = (X64.std(axis=0) + 1e-8).astype(np.float32)
        del X64

        self.weight_mean = float(self.w_data.mean())

        if config.transform_f0 is None:
            self.f0 = float(np.std(self.y_data))
        else:
            self.f0 = float(config.transform_f0)
        y_t = self._transform_y(self.y_data)
        self.y_mean = float(y_t.mean())
        self.y_std = float(y_t.std() + 1e-8)

        self.y_norm_data = ((y_t - self.y_mean) / self.y_std).astype(np.float32)
        del y_t

        self.X_norm_data = ((self.X_data - self.X_mean) / self.X_std).astype(np.float32)
        self.w_norm_data = (self.w_data / self.weight_mean).astype(np.float32)

        print(f"  weight_mean={self.weight_mean:.3e}  f0={self.f0:.3e}")

    # ------------------------------------------------------------------
    # Output transform
    # ------------------------------------------------------------------
    def _transform_y(self, y: np.ndarray) -> np.ndarray:
        if self.transform == "arcsinh":
            return np.arcsinh(y / self.f0).astype(np.float32)
        elif self.transform == "log":
            return (np.sign(y) * np.log(np.abs(y) + self.epsilon)).astype(np.float32)
        return y.astype(np.float32)

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------
    # Check the number of points in the dataset
    def __len__(self) -> int:
        return self.n_valid

    # Get a single data point by index
    def __getitem__(self, idx: int):
        x_norm = self.X_norm_data[idx]
        phi = self.phi_data[idx]
        y_norm = self.y_norm_data[idx]
        w_norm = self.w_norm_data[idx]

        return (
            torch.from_numpy(x_norm),
            torch.tensor(phi, dtype=torch.float32),
            torch.tensor(y_norm, dtype=torch.float32),
            torch.tensor(w_norm, dtype=torch.float32),
        )

    # Get a group of points by a list of indices. Uses fancy indexing to avoid loop overhead.
    def __getitems__(self, indices: list[int]) -> list:
        indices = np.asarray(indices, dtype=np.intp)

        X_batch = self.X_norm_data[indices]
        phi_batch = self.phi_data[indices]
        y_batch = self.y_norm_data[indices]
        w_batch = self.w_norm_data[indices]

        # Return four tensors — collate_fn receives one "sample" and passes
        # it straight through without any further stacking.
        return [
            torch.from_numpy(X_batch),
            torch.from_numpy(phi_batch.copy()),
            torch.from_numpy(y_batch.copy()),
            torch.from_numpy(w_batch.copy()),
        ]

    # ------------------------------------------------------------------
    # Normalization export (unchanged)
    # ------------------------------------------------------------------
    def get_normalization_params(self) -> Dict:
        return {
            'X_mean':        self.X_mean.tolist(),
            'X_std':         self.X_std.tolist(),
            'y_mean':        self.y_mean,
            'y_std':         self.y_std,
            'transform':     str(self.transform),
            'f0':            self.f0,
            'epsilon':       self.epsilon,
            'feature_names': self.FEATURE_NAMES,
        }

    # shutdown() is now a no-op — kept for API compatibility
    def shutdown(self):
        pass

class RadiationSubset(Dataset):
    """
    A view into a RadiationDataset restricted to a subset of *physical* indices.

    Parameters
    ----------
    dataset : RadiationDataset
        The parent dataset (already loaded into RAM).
    physical_indices : np.ndarray, dtype=np.intp
        Indices into dataset.X_norm_data / y_norm_data / w_norm_data.
        Must be in [0, dataset.n_valid).
    """

    def __init__(
            self,
            dataset: RadiationDataset,
            physical_indices: np.ndarray,
    ):
        self.ds = dataset
        self.idx = np.asarray(physical_indices, dtype=np.intp)
        self.n_phys = len(self.idx)

    def __len__(self) -> int:
        return self.n_phys

    def __getitems__(self, positions: list[int]) -> list:
        positions = np.asarray(positions, dtype=np.intp)
        phys_idx = self.idx[positions]

        X_batch = self.ds.X_norm_data[phys_idx]
        phi_batch = self.ds.phi_data[phys_idx]
        y_batch = self.ds.y_norm_data[phys_idx]
        w_batch = self.ds.w_norm_data[phys_idx]

        return [
            torch.from_numpy(X_batch.copy()),
            torch.from_numpy(phi_batch.copy()),
            torch.from_numpy(y_batch.copy()),
            torch.from_numpy(w_batch.copy()),
        ]

# ==============================================================================
# Neural Network Model
# ==============================================================================
class RadiationEmulator(nn.Module):
    """
    Neural network emulator for medium-induced radiation intensity.

    Architecture: MLP with skip connections every 2 layers.
    """

    def __init__(
            self,
            input_dim: int = 7,
            hidden_dim: int = 256,
            n_layers: int = 5,
            activation: str = "silu",
            dropout_p: float = 0.1,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        # Select activation function
        activations = {
            'silu': nn.SiLU,
            'relu': nn.ReLU,
            'tanh': nn.Tanh,
            'gelu': nn.GELU,
        }
        act_fn = activations.get(activation, nn.SiLU)

        # Build layers
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.input_act = act_fn()

        # Hidden layers with skip connections
        self.hidden_layers = nn.ModuleList()
        self.hidden_acts = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        for i in range(n_layers - 1):
            self.hidden_layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.hidden_acts.append(act_fn())
            self.dropouts.append(nn.Dropout(p=dropout_p))

        # Output layer -- 3 features: A0, A1, & A2 fourier harmonic factors.
        # Each head predicts the harmonic amplitude scaled by 1/f0 (same scale
        # convention as the arcsinh transform used on the training targets).
        self.output_layer = nn.Linear(hidden_dim, 3)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights using Xavier initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, 7) with normalized features
            [x, k_perp, E, z0, u_perp, T, g]

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch_size, 3): (A0, A1, A2) harmonic
            amplitudes, in units of 1/f0 (physical amplitude = output * f0).
        """
        # Input layer
        h = self.input_act(self.input_layer(x))

        # Hidden layers with skip connections every 2 layers
        h_block_in = h
        for i, (layer, act, drop) in enumerate(zip(self.hidden_layers, self.hidden_acts, self.dropouts)):
            if i % 2 == 0:
                h_block_in = h  # Save block input at the start of each 2-layer block
            h = drop(act(layer(h)))
            if i % 2 == 1:  # End of block: add saved block input
                h = h + h_block_in

        # Output layer
        return self.output_layer(h)


# ==============================================================================
# Training utilities
# ==============================================================================
def combine_harmonics(A_heads: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    """
    Reconstruct I/f0 = A0 + A1*cos(phi) + A2*cos(2*phi) from the 3 network heads.

    A_heads : (B, 3) tensor of (A0, A1, A2), each already in units of 1/f0.
    phi     : (B,)   tensor of azimuthal angle.
    """
    return A_heads[:, 0] + A_heads[:, 1] * torch.cos(phi) + A_heads[:, 2] * torch.cos(2 * phi)


def compute_loss(
        model: nn.Module,
        inputs: torch.Tensor,
        phi: torch.Tensor,
        targets: torch.Tensor,
        weights: torch.Tensor,
        config: TrainingConfig,
        X_mean: torch.Tensor,
        X_std: torch.Tensor,
        y_mean: float,
        y_std: float,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Compute weighted MSE loss with physics constraints.

    The model outputs 3 harmonic heads (A0, A1, A2). These are combined via
    A0 + A1*cos(phi) + A2*cos(2*phi) to reconstruct I/f0, then arcsinh-transformed
    and standardized to match the (pre-transformed) targets.

    Returns total loss and dictionary of individual loss components.
    """
    # Get head outputs and compute predicted values at phi points
    A_heads = model(inputs)                       # (B, 3)
    I_over_f0_pred = combine_harmonics(A_heads, phi)
    pred_transformed = torch.arcsinh(I_over_f0_pred)
    predictions = (pred_transformed - y_mean) / y_std

    """
    Weighted MSE loss
    """
    mse = (weights * (predictions - targets) ** 2).mean()

    """
    UV decay enforcement loss

    Enforces each harmonic head -> 0 as k_perp -> inf. Applied per-head since
    every harmonic amplitude should vanish independently in the UV.
    """
    if config.lambda_uv > 0.0:
        max_frac = 0.05
        rng = np.random.default_rng()
        n_uv_samples = 64

        # Get shape and device of inputs
        B = inputs.shape[0]
        device = inputs.device

        # Tile the other 7 parameters from random rows of the training batch
        idx = torch.randint(0, B, (n_uv_samples,), device=device)
        uv_params = inputs[idx].clone()  # (n_uv_samples, 7)

        E_mean = X_mean[2].to(device)   # index 2 == 'E'
        E_std = X_std[2].to(device)
        energy_params = (uv_params[:, 2] * E_std + E_mean).cpu().numpy()

        # Sample k_perp log-uniformly in the UV region, depending on energy of sample point
        log_k = []
        for i in np.arange(0, len(energy_params)):
            # 2 * log(energy) = log(energy^2) -- Use minimum of uv_kt_threshold to avoid penalizing structure at low pT
            log_lo = np.amax([math.log(max_frac) + 2 * math.log(energy_params[i]),
                               math.log(config.uv_kt_threshold)])
            log_hi = 2 * math.log(energy_params[i])
            if log_lo < log_hi:
                log_k.append(rng.uniform(log_lo, log_hi))
            else:
                log_k.append(log_lo)
        log_k = torch.tensor(log_k, device=device)
        k_perp_uv = torch.exp(log_k)

        uv_params[:, 1] = k_perp_uv  # index 1 == 'k_perp'

        uv_heads = model(uv_params)  # shape (n_uv_samples, 3)

        # Use log weight to penalize nonzero result at larger k_perp values more
        log_esqr = torch.tensor(math.log(max_frac) + 2 * np.log(energy_params), device=device)
        log_weight = 2 * log_k - 2 * log_esqr

        # Penalize all three heads
        uv_decay = (log_weight.unsqueeze(1) * uv_heads ** 2).mean()
    else:
        uv_decay = torch.tensor(0.0, device=inputs.device)

    """
    UV power law behavior enforcement loss

    Enforces A_n(alpha * k_perp) / A_n(k_perp) ~ alpha^{-power} for each head.
    """
    if config.lambda_uv_power > 0.0:
        B = inputs.shape[0]
        device = inputs.device
        alpha = 4.0  # scale_factor between compared points -- compare f(k) vs f(scale_factor * k)
        n_uv_power_law_samples = 64
        k_perp_base = 2.0  # base UV scale
        power = 4.0  # expected UV power law exponent

        idx = torch.randint(0, B, (n_uv_power_law_samples,), device=device)
        base_params = inputs[idx].clone()

        # Randomise k_perp_base point in a moderate UV range
        k_perp_sample = k_perp_base * torch.exp(
            torch.empty(n_uv_power_law_samples, device=device).uniform_(0, 3)
        )

        # Low-k point
        base_params[:, 1] = k_perp_sample
        heads_low = model(base_params)  # (n, 3)

        # High-k point (same direction, same other params)
        high_params = base_params.clone()
        high_params[:, 1] = alpha * k_perp_sample
        heads_high = model(high_params)  # (n, 3)

        expected_ratio = alpha ** (-power)
        log_ratio = torch.log(torch.abs(heads_high) + 1e-30) - torch.log(torch.abs(heads_low) + 1e-30)
        target_log_ratio = torch.full_like(log_ratio, math.log(expected_ratio))

        uv_power_law = nn.functional.mse_loss(log_ratio, target_log_ratio)
    else:
        uv_power_law = torch.tensor(0.0, device=inputs.device)

    # Total loss
    total_loss = (
            mse
            + config.lambda_uv * uv_decay
            + config.lambda_uv_power * uv_power_law
    )

    components = {
        'mse': mse.item(),
        'uv_decay': uv_decay.item(),
        'uv_power_law': uv_power_law.item(),
        'total': total_loss.item(),
    }

    return total_loss, components


def train_epoch(
        model: nn.Module,
        dataloader: DataLoader,
        optimizer: torch.optim.Optimizer,
        config: TrainingConfig,
        X_mean: torch.Tensor,
        X_std: torch.Tensor,
        y_mean: float,
        y_std: float,
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    total_mse = 0.0
    n_batches = 0

    t0 = time.time()
    for i, (inputs, phi, targets, weights) in enumerate(dataloader):
        # if i % 10000 == 0 and i != 0:
        #     print(f"  Batch {i}/{len(dataloader)}  [avg {(time.time() - t0)/i:.3f}s/batch]")

        # Send tensors to device
        inputs = inputs.to(config.device)
        phi = phi.to(config.device)
        targets = targets.to(config.device)
        weights = weights.to(config.device)

        # Compute loss and step optimizer
        optimizer.zero_grad()
        loss, components = compute_loss(model, inputs, phi, targets, weights, config, X_mean, X_std, y_mean, y_std)
        loss.backward()
        optimizer.step()

        # Add to running sum of loss and MSE
        total_loss += components['total']
        total_mse += components['mse']
        n_batches += 1

    # Return loss and MSE
    return {
        'loss': total_loss / n_batches,
        'mse': total_mse / n_batches,
    }


def validate(
        model: nn.Module,
        dataloader: DataLoader,
        config: TrainingConfig,
        X_mean: torch.Tensor,
        X_std: torch.Tensor,
        y_mean: float,
        y_std: float,
) -> Dict[str, float]:
    """Validate the model."""
    model.eval()
    total_loss = 0.0
    total_mse = 0.0
    n_batches = 0

    with torch.no_grad():
        for inputs, phi, targets, weights in dataloader:
            # Send tensors to device
            inputs = inputs.to(config.device)
            phi = phi.to(config.device)
            targets = targets.to(config.device)
            weights = weights.to(config.device)

            # Compute loss
            _, components = compute_loss(model, inputs, phi, targets, weights, config, X_mean, X_std, y_mean, y_std)

            # Add to running sum of loss and MSE
            total_loss += components['total']
            total_mse += components['mse']
            n_batches += 1

    # Return loss and MSE
    return {
        'loss': total_loss / n_batches,
        'mse': total_mse / n_batches,
    }




def save_training_checkpoint(
        path: str,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler,
        epoch: int,
        best_val_loss: float,
        patience_counter: int,
):
    """Save a full training state for resumption."""
    raw_state_dict = model.state_dict()
    fixed_state_dict = {
        k.removeprefix('_orig_mod.').removeprefix('module.'): v
        for k, v in raw_state_dict.items()
    }
    torch.save({
        'epoch': epoch,
        'model_state_dict': fixed_state_dict,
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_val_loss': best_val_loss,
        'patience_counter': patience_counter,
    }, path)
    print(f"  [checkpoint] Saved training state at epoch {epoch + 1} → {path}")


def load_training_checkpoint(
        path: str,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler,
        device: str,
):
    """
    Load a training checkpoint.  Returns (start_epoch, best_val_loss, patience_counter).
    """
    print(f"Resuming from checkpoint: {path}")
    ckpt = torch.load(path, map_location=device, weights_only=True)
    # Strip both torch.compile's '_orig_mod.' and DataParallel's 'module.' prefixes.
    # Checkpoints are always stored in bare (unwrapped) format, so we load into
    # the bare model first, then let DataParallel/compile wrap it afterwards.
    bare_state_dict = {
        k.removeprefix('_orig_mod.').removeprefix('module.'): v
        for k, v in ckpt['model_state_dict'].items()
    }

    # Unwrap the model to load into the bare RadiationEmulator, then re-wrap.
    bare_model = model.module if isinstance(model, nn.DataParallel) else model
    bare_model = getattr(bare_model, '_orig_mod', bare_model)  # unwrap torch.compile
    bare_model.load_state_dict(bare_state_dict)

    # Force model to device passed to this function
    bare_model.to(device)

    # Setup optimizer and scheduler from the checkpoint.
    optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    scheduler.load_state_dict(ckpt['scheduler_state_dict'])

    # Set counters
    start_epoch    = ckpt['epoch'] + 1          # resume at the *next* epoch
    best_val_loss  = ckpt['best_val_loss']
    patience_counter = ckpt['patience_counter']

    print(f"  Resumed from epoch {ckpt['epoch'] + 1}  |  best val loss: {best_val_loss:.4e}")
    return start_epoch, best_val_loss, patience_counter

def setup_device(config: TrainingConfig) -> tuple[bool, int]:
    """
    Initialise the correct device and (optionally) the distributed process group.

    Returns
    -------
    distributed : bool
        True if running under torchrun / SLURM with multiple ranks.
    local_rank : int
        The rank of this process on the current node (0 if not distributed).
    """
    if "LOCAL_RANK" not in os.environ:
        # Not launched via torchrun — single process
        local_rank = 0
        if torch.cuda.is_available():
            config.device = "cuda:0"
            print("Single-GPU training")
        else:
            config.device = "cpu"
            print("CPU training")
        return False, local_rank

        # Launched via torchrun — always treat as distributed
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    if world_size > 1 and torch.cuda.is_available():
        # MUST set device before init_process_group
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
        config.device = f"cuda:{local_rank}"
        if local_rank == 0:
            print(f"Distributed training: {dist.get_world_size()} GPUs")
        return True, local_rank
    else:
        # torchrun with 1 process, or no GPU
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            config.device = f"cuda:{local_rank}"
            print("Single-GPU training (via torchrun)")
        else:
            config.device = "cpu"
            print("CPU training")
        return False, local_rank

# ==============================================================================
# Main training function
# ==============================================================================
def train_model(config: TrainingConfig):
    """Train the radiation emulator model."""
    print(f"[rank {os.environ.get('LOCAL_RANK', '?')}] "
          f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'NOT SET')} "
          f"device_count={torch.cuda.device_count()}")

    print("=" * 70)
    print("RADIATION EMULATOR TRAINING")
    print("=" * 70)
    print(f"Device: {config.device}")
    print(f"Data file: {config.data_file}")
    print()

    is_distributed, local_rank = setup_device(config)
    is_main = (local_rank == 0)  # gate all printing/saving on this flag

    # Load dataset
    dataset = RadiationDataset(config.data_file, transform_output=config.transform)

    # Save normalization parameters
    norm_params = dataset.get_normalization_params()
    if is_main:
        print("Saving normalization...")
        with open(config.normalization_file, 'w') as f:
            json.dump(norm_params, f, indent=2)

    # Split on *physical* indices, so each subset gets a
    # contiguous, cache-friendly slice of the underlying arrays.
    n_phys_train = int(dataset.n_valid * config.train_fraction)

    rng_split = np.random.default_rng(42)
    perm = rng_split.permutation(dataset.n_valid).astype(np.intp)

    train_idx = perm[:n_phys_train]
    val_idx = perm[n_phys_train:]

    # Split training and validation datasets
    train_dataset = RadiationSubset(dataset, train_idx)
    val_dataset = RadiationSubset(dataset, val_idx)

    # print(f"Training samples: {n_train}")
    # print(f"Validation samples: {n_val}")

    # Create dataloaders
    if is_distributed:
        train_sampler = DistributedSampler(train_dataset, shuffle=True)
        shuffle_train = False
    else:
        train_sampler = None
        shuffle_train = True
    if config.device == "cpu":
        train_loader = DataLoader(
            train_dataset, batch_size=config.batch_size, sampler=train_sampler,
            shuffle=shuffle_train,
            num_workers=config.num_workers,
            persistent_workers=config.num_workers > 0,
            collate_fn=_collate_passthrough
        )
        val_loader = DataLoader(
            val_dataset, batch_size=config.batch_size * 4, shuffle=False,
            num_workers=0,
            collate_fn=_collate_passthrough
        )
    else:
        train_loader = DataLoader(
            train_dataset, batch_size=config.batch_size, sampler=train_sampler,
            shuffle=shuffle_train,
            num_workers=config.num_workers,
            persistent_workers=config.num_workers > 0,
            pin_memory=True,
            collate_fn=_collate_passthrough
        )
        val_loader = DataLoader(
            val_dataset, batch_size=config.batch_size * 4, shuffle=False,
            num_workers=0,  # No workers needed: no backward pass to overlap with
            pin_memory=True,
            collate_fn=_collate_passthrough
        )

    # Create model
    model = RadiationEmulator(
        input_dim=len(dataset.FEATURE_NAMES),
        hidden_dim=config.hidden_dim,
        n_layers=config.n_layers,
        activation=config.activation,
        dropout_p=config.dropout_p,
    ).to(config.device)

    # Optimizer -- applies various strategies that change the way we use our neurons
    # Controls usage of dropout and L2 regularization to encourage generalization instead of memorization
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    # Scheduler -- controls the variation of learning rate over training epochs
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10,
        min_lr=6.24e-5,
    )

    # ── Resume from checkpoint if one exists ──────────────────────────────
    if os.path.exists(config.checkpoint_file):
        start_epoch, best_val_loss, patience_counter = load_training_checkpoint(
            config.checkpoint_file, model, optimizer, scheduler, config.device
        )
    else:
        print("No checkpoint found – starting from scratch.")
        best_val_loss = float('inf')
        patience_counter = 0
        start_epoch = 0

    # --- Wrap model in appropriate parallelization method ---
    if is_distributed:
        print(f"[rank {local_rank}] model device: {next(model.parameters()).device}")
        model = DDP(model, device_ids=[local_rank])
        print(f"[rank {local_rank}] DDP model ready")
    elif torch.cuda.device_count() > 1:
        # Local machine with multiple GPUs but not launched via torchrun —
        # DataParallel is the fallback here, or you can just use one GPU.
        print(f"Multiple GPUs available but not running under torchrun. "
              f"Using DataParallel.")
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    # LR finder -- to be run before the main training loop, finds optimal learning rate
    # Looks for minima in the loss as function of learning rate, returns rate just before minima in loss function
    if config.run_lr_finder:
        print("\nRunning LR range test...")
        print("-" * 70)
        lrs, losses, raw_losses = find_learning_rate(model, train_loader, optimizer, config)
        suggested_lr = plot_lr_finder(lrs, losses, raw_losses)
        print(f"\nRe-run with --learning-rate {suggested_lr / 3:.2e} (1/3 of suggested)")
        return model, dataset.get_normalization_params()

    # Training loop
    if hasattr(torch, 'compile') and config.compile:
        model = torch.compile(model)

    # ── SIGTERM handler: save checkpoint before SLURM kills the job ──────
    _sigterm_received = [False]

    def _sigterm_handler(signum, frame):
        print("\n[SLURM] SIGTERM received – saving checkpoint before exit...")
        save_training_checkpoint(
            config.checkpoint_file, model, optimizer, scheduler,
            current_epoch[0], best_val_loss, patience_counter
        )
        _sigterm_received[0] = True

    current_epoch = [start_epoch]
    signal.signal(signal.SIGTERM, _sigterm_handler)

    # Pre-move normalization tensors to device once, for use in loss computation
    X_mean_t = torch.tensor(dataset.X_mean, dtype=torch.float32).to(config.device)
    X_std_t = torch.tensor(dataset.X_std, dtype=torch.float32).to(config.device)
    y_mean_t = dataset.y_mean
    y_std_t = dataset.y_std

    print("\nStarting training...")
    print("-" * 70)

    try:
        for epoch in range(start_epoch, config.n_epochs):
            current_epoch[0] = epoch

            # Update the epoch in the sampler -- different order each epoch.
            if is_distributed:
                train_sampler.set_epoch(epoch)

            if _sigterm_received[0]:
                print("Exiting cleanly after SIGTERM.")
                break

            # Train
            train_metrics = train_epoch(model, train_loader, optimizer, config, X_mean_t, X_std_t, y_mean_t, y_std_t)

            # Validate
            val_metrics = validate(model, val_loader, config, X_mean_t, X_std_t, y_mean_t, y_std_t)

            # Update scheduler
            scheduler.step(val_metrics['mse'])  # Scheduler tracks mse, not the overall loss.

            # Print progress
            current_lr = optimizer.param_groups[0]['lr']
            if is_main:
                print(
                    f"Epoch {epoch + 1:3d}/{config.n_epochs} | "
                    f"Train Loss: {train_metrics['loss']:.4e} | "
                    f"Val Loss: {val_metrics['loss']:.4e} | "
                    f"Train MSE: {train_metrics['mse']:.4e} | "
                    f"Val MSE: {val_metrics['mse']:.4e} | "
                    f"MSE Ratio: {(train_metrics['mse'] / val_metrics['mse']):.4e} | "
                    f"LR: {current_lr:.2e}"
                )

            # Early stopping check
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                patience_counter = 0

                # Save best model — unwrap torch.compile's OptimizedModule if present
                # so that the checkpoint is always loadable without torch.compile.
                raw_state_dict = model.state_dict()
                fixed_state_dict = {
                    k.removeprefix('_orig_mod.').removeprefix('module.'): v
                    for k, v in raw_state_dict.items()
                }
                if is_main:
                    torch.save({
                        'model_state_dict': fixed_state_dict,
                        'config': {
                            'hidden_dim': config.hidden_dim,
                            'n_layers': config.n_layers,
                            'activation': config.activation,
                            'input_dim': len(dataset.FEATURE_NAMES),
                            'dropout_p': config.dropout_p,
                        },
                        'epoch': epoch,
                        'val_loss': best_val_loss,
                    }, config.model_file)
            else:
                patience_counter += 1
                if patience_counter >= config.patience:
                    print(f"\nEarly stopping at epoch {epoch + 1}")
                    break

            # ── Periodic resume checkpoint ─────────────────────────────
            if (epoch + 1) % config.checkpoint_interval == 0:
                save_training_checkpoint(
                    config.checkpoint_file, model, optimizer, scheduler,
                    epoch, best_val_loss, patience_counter
                )

    except KeyboardInterrupt:
        print("Keyboard interrupt – saving checkpoint...")
        save_training_checkpoint(
            config.checkpoint_file, model, optimizer, scheduler,
            current_epoch[0], best_val_loss, patience_counter
        )
        print("Training stopped.")

    print("-" * 70)
    print(f"Training complete!")
    print(f"Best validation loss: {best_val_loss:.4e}")
    print(f"Model saved to: {config.model_file}")
    print(f"Normalization params saved to: {config.normalization_file}")

    # Shutdown our dataset object
    dataset.shutdown()

    return model, dataset.get_normalization_params()


# ==============================================================================
# Inference utilities
# ==============================================================================
class RadiationEmulatorInference:
    """
    Wrapper for inference with the trained radiation emulator.

    Handles normalization and inverse transforms automatically.
    """

    FEATURE_NAMES = ['x', 'k_perp', 'E', 'z0', 'u_perp', 'T', 'g']

    def __init__(
            self,
            model_file: str = "data/radiation_emulator.pt",
            normalization_file: str = "data/radiation_normalization.json",
            device: str = "cpu",
            compile: bool = False,
            quiet: bool = False,
    ):
        """
        Load trained model and normalization parameters.

        Parameters
        ----------
        model_file : str
            Path to saved model checkpoint
        normalization_file : str
            Path to JSON file with normalization parameters
        device : str
            Device to run inference on ('cpu' or 'cuda')
        """
        self.device = device

        # Load normalization parameters
        with open(normalization_file, 'r') as f:
            self.norm_params = json.load(f)

        self.X_mean = torch.tensor(self.norm_params['X_mean'], dtype=torch.float32)
        self.X_std = torch.tensor(self.norm_params['X_std'], dtype=torch.float32)
        # self.X_max = torch.tensor(self.norm_params['X_max'], dtype=torch.float32)
        # self.X_min = torch.tensor(self.norm_params['X_min'], dtype=torch.float32)
        self.y_mean = self.norm_params['y_mean']
        self.y_std = self.norm_params['y_std']
        self.transform = self.norm_params['transform']
        self.f0 = self.norm_params['f0']
        self.epsilon = self.norm_params['epsilon']

        # Load model
        checkpoint = torch.load(model_file, map_location=device, weights_only=True)
        model_config = checkpoint['config']

        self.model = RadiationEmulator(
            input_dim=model_config['input_dim'],
            hidden_dim=model_config['hidden_dim'],
            n_layers=model_config['n_layers'],
            activation=model_config['activation'],
            dropout_p=model_config.get('dropout_p', 0.0),  # backward compatible
        ).to(device)

        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs")
            self.model = nn.DataParallel(self.model)

        try:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        except RuntimeError:
            # Strip torch.compile's '_orig_mod.' prefix if present (backward compatible)
            state_dict = {
                k.removeprefix('_orig_mod.').removeprefix('module.'): v
                for k, v in checkpoint['model_state_dict'].items()
            }
            self.model.load_state_dict(state_dict)
        self.model.eval()

        # Pre-move normalization tensors to the target device once,
        # so repeated predict() calls don't trigger device copies.
        self.X_mean = self.X_mean.to(device)
        self.X_std = self.X_std.to(device)

        # Optionally compile the model for faster repeated inference
        # (requires PyTorch >= 2.0; falls back silently on older versions)
        if hasattr(torch, 'compile') and compile:
            self.model = torch.compile(self.model)

        if not quiet:
            print(f"Loaded model from {model_file}")
            print(f"  Validation loss: {checkpoint['val_loss']:.4e}")
            print(f"  Trained for {checkpoint['epoch'] + 1} epochs")

    def predict(
            self,
            x: np.ndarray,
            k_perp: np.ndarray,
            phi: np.ndarray,
            E: np.ndarray,
            z0: np.ndarray,
            u_perp: np.ndarray,
            T: np.ndarray,
            g: np.ndarray,
    ) -> np.ndarray:
        """
        Physical-facing entry point -- returns the combined scalar intensity
        """
        A0, A1, A2 = self.predict_harmonics(x, k_perp, E, z0, u_perp, T, g)
        predictions = A0 + A1 * np.cos(phi) + A2 * np.cos(2 * phi)
        return predictions

    def predict_kxky(
            self,
            x: np.ndarray,
            kx: np.ndarray,
            ky: np.ndarray,
            E: np.ndarray,
            z0: np.ndarray,
            u_perp: np.ndarray,
            T: np.ndarray,
            g: np.ndarray,
    ) -> np.ndarray:
        """
        Legacy entry point: still accepts kx, ky (as callers expect),
        derives k_perp/phi internally, and returns the combined scalar
        intensity — same public contract as before.
        """
        k_perp = np.sqrt(kx ** 2 + ky ** 2)
        phi = np.arctan2(ky, kx)

        A0, A1, A2 = self.predict_harmonics(x, k_perp, E, z0, u_perp, T, g)
        predictions = A0 + A1 * np.cos(phi) + A2 * np.cos(2 * phi)
        return predictions

    def predict_harmonics(
            self,
            x: np.ndarray,
            k_perp: np.ndarray,
            E: np.ndarray,
            z0: np.ndarray,
            u_perp: np.ndarray,
            T: np.ndarray,
            g: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns physical (A0, A1, A2) harmonic amplitudes for the given
        (x, k_perp, ...) points — no phi dependence, so callers can evaluate
        this on a much coarser grid than the full 3D (kx, ky, kz) grid.
        """
        inputs = np.column_stack([x, k_perp, E, z0, u_perp, T, g]).astype(
            np.float32, order='C', copy=False
        )
        A_heads = self.predict_harmonics_raw(inputs)
        return A_heads[:, 0], A_heads[:, 1], A_heads[:, 2]

    def predict_harmonics_raw(self, inputs: np.ndarray) -> np.ndarray:
        """
        Fast entry-point for pre-stacked (N, 7) float32 arrays in feature
        order [x, k_perp, E, z0, u_perp, T, g]. Returns physical (N, 3)
        array of (A0, A1, A2) harmonic amplitudes (phi not yet applied).
        """
        inputs_tensor = torch.from_numpy(
            np.asarray(inputs, dtype=np.float32, order='C')
        ).to(self.device, non_blocking=True)

        inputs_norm = (inputs_tensor - self.X_mean) / self.X_std

        with torch.no_grad():
            A_heads = self.model(inputs_norm)  # (N, 3), units of 1/f0

        A_heads = A_heads if self.device == "cpu" else A_heads.cpu()
        return self.f0 * A_heads.numpy()

    def predict_dict(self, inputs: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Predict from a dictionary of inputs.

        Parameters
        ----------
        inputs : dict
            Dictionary with keys matching FEATURE_NAMES

        Returns
        -------
        np.ndarray
            Predicted radiation intensity
        """
        return self.predict(
            x=inputs['x'],
            k_perp=inputs['k_perp'],
            phi=inputs['phi'],
            E=inputs['E'],
            z0=inputs['z0'],
            u_perp=inputs['u_perp'],
            T=inputs['T'],
            g=inputs['g'],
        )

    def compute_dNd3k_grid(self,
                           E: float,
                           z0: float,
                           u_perp: float,
                           T: float,
                           g: float,
                           kz_values: np.ndarray,
                           k_perp_values: np.ndarray,
                           phi_values: np.ndarray,
                           ) -> (np.ndarray):
        """
        Computes a complete grid of dN/d^3k shaped as (k_perp, phi, kz).

        Since x = x(k_perp, kz) has no phi dependence, the network is
        evaluated only on the 2D (k_perp, kz) grid; phi dependence is
        reconstructed analytically via A0 + A1*cos(phi) + A2*cos(2*phi).
        """
        # create a 2D meshgrid in the only two variables the network needs
        kperp_grid2d, kz_grid2d = np.meshgrid(k_perp_values, kz_values, indexing='ij')  # (n_kperp, n_kz)
        x_grid2d = (1 / (E ** 2 * np.sqrt(2))) * (
                E * kz_grid2d + np.sqrt(E ** 2 * (kz_grid2d ** 2 + kperp_grid2d ** 2))
        )

        # Build input grid once
        n_pts = x_grid2d.size
        grid_inputs = np.column_stack([
            x_grid2d.ravel(), kperp_grid2d.ravel(),
            np.full(n_pts, E), np.full(n_pts, z0),
            np.full(n_pts, u_perp), np.full(n_pts, T), np.full(n_pts, g),
        ]).astype(np.float32)

        # Single batched network call over the (k_perp, kz) grid -- no phi dependence yet
        A_flat = self.predict_harmonics_raw(grid_inputs)  # (n_pts, 3)
        A0_2d = A_flat[:, 0].reshape(x_grid2d.shape)
        A1_2d = A_flat[:, 1].reshape(x_grid2d.shape)
        A2_2d = A_flat[:, 2].reshape(x_grid2d.shape)

        # Reconstruct full angular dependence via broadcasting -- shape (n_kperp, n_phi, n_kz)
        cos_phi = np.cos(phi_values)[None, :, None]
        cos_2phi = np.cos(2 * phi_values)[None, :, None]
        I_nn = A0_2d[:, None, :] + A1_2d[:, None, :] * cos_phi + A2_2d[:, None, :] * cos_2phi

        x_grid3d = np.broadcast_to(x_grid2d[:, None, :], I_nn.shape)
        kperp_grid3d = np.broadcast_to(kperp_grid2d[:, None, :], I_nn.shape)

        # Set any unphysical x coordinate values to zero
        mask = x_grid3d > 1.0
        if np.amax(mask) > 0:
            print("Oh no!!! (x > 1)!!!")
            I_nn[x_grid3d > 1.0] = 0

        # Compute dN/dxd^2k_perp by dividing out energy of each grid point
        N_nn = I_nn / (E * x_grid3d)  # Still needs casimir factor

        # Convert to dN/d^3k via the x -> kz Jacobian
        dkz_dx = (1 / np.sqrt(2)) * (E + kperp_grid3d ** 2 / (2 * x_grid3d ** 2 * E))
        jacobian = 1.0 / dkz_dx
        N_nn = N_nn * jacobian

        # Returned grid is (k_perp, phi, kz), NOT (kx, ky, kz)
        return N_nn

    def sample_emission(self,
                        E: np.ndarray,
                        z0: np.ndarray,
                        u_perp: np.ndarray,
                        T: np.ndarray,
                        g: np.ndarray,
                        N_samples: int = 1,
                        rng: np.random.Generator = np.random.default_rng()
                        ) -> tuple[float, np.ndarray]:
        """
        Computes a grid in (x, k_perp, phi) and returns an inverse-CDF
        sample, converted to a Cartesian (kx, ky, kz) momentum vector.
        """

        # Grid of x, k_perp, phi values
        max_k_perp = 5  # Maybe should be dependent on energy, needs testing.
        x_values = np.logspace(-4, 0, 10)
        k_perp_values = np.linspace(0, max_k_perp, 50)
        phi_values = np.linspace(0, 2 * np.pi, 64, endpoint=False)

        # 2D grid in the only two variables the network needs
        x_grid2d, kperp_grid2d = np.meshgrid(x_values, k_perp_values, indexing='ij')  # (n_x, n_kperp)
        n_pts = x_grid2d.size
        grid_inputs = np.column_stack([
            x_grid2d.ravel(), kperp_grid2d.ravel(),
            np.full(n_pts, E), np.full(n_pts, z0),
            np.full(n_pts, u_perp), np.full(n_pts, T), np.full(n_pts, g),
        ]).astype(np.float32)

        # Single batched network call over the (x, k_perp) grid -- no phi dependence yet
        A_flat = self.predict_harmonics_raw(grid_inputs)
        A0_2d = A_flat[:, 0].reshape(x_grid2d.shape)
        A1_2d = A_flat[:, 1].reshape(x_grid2d.shape)
        A2_2d = A_flat[:, 2].reshape(x_grid2d.shape)

        # Reconstruct full (x, k_perp, phi) grid via broadcasting
        cos_phi = np.cos(phi_values)[None, None, :]
        cos_2phi = np.cos(2 * phi_values)[None, None, :]
        I_nn = A0_2d[:, :, None] + A1_2d[:, :, None] * cos_phi + A2_2d[:, :, None] * cos_2phi
        # shape: (n_x, n_kperp, n_phi)

        # -------------------------------------------------------
        # 1. INTEGRATION
        #    d^2k_perp = k_perp dk_perp dphi -- the k_perp Jacobian must be
        #    folded in explicitly since we're integrating in polar coords.
        # -------------------------------------------------------
        I_x_kperp = np.trapezoid(I_nn, phi_values, axis=2)  # integrate over phi -> (n_x, n_kperp)
        I_x = np.trapezoid(I_x_kperp * k_perp_values[None, :], k_perp_values, axis=1)  # -> (n_x,)
        # Integrate over x in log-space (accounts for log-spaced grid) -- includes Jacobian, factor of x
        total_integral = np.trapezoid(I_x * x_values, np.log(x_values))  # scalar

        # -------------------------------------------------------
        # 2. SAMPLING
        #    Treat I_nn (weighted by the k_perp Jacobian) as an
        #    (unnormalized) 3D probability density over (x, k_perp, phi)
        #    and draw N_samples points from it.
        # -------------------------------------------------------
        weight_grid = I_nn * k_perp_values[None, :, None]  # polar area element
        weight_flat = np.clip(weight_grid, 0, None).ravel()  # ensure non-negative
        pdf = weight_flat / weight_flat.sum()  # normalize to sum to 1
        cdf = np.cumsum(pdf)  # build CDF
        cdf[-1] = 1.0  # Force exact upper bound — removes all floating point slop

        # Draw uniform samples and find where they land in the CDF
        uniform_samples = rng.uniform(size=N_samples)
        flat_indices = np.searchsorted(cdf, uniform_samples)  # shape: (N_samples,)

        # Convert flat indices back to 3D grid indices
        ix, ikperp, iphi = np.unravel_index(flat_indices, I_nn.shape)

        # Look up the corresponding coordinate values
        sampled_x = x_values[ix]
        sampled_kperp = k_perp_values[ikperp]
        sampled_phi = phi_values[iphi]  # full 0..2pi range -- no sign-flip hack needed

        # Convert back to Cartesian transverse components
        sampled_kx = sampled_kperp * np.cos(sampled_phi)
        sampled_ky = sampled_kperp * np.sin(sampled_phi)

        # Actual longitudinal component of the momentum vector
        # We know $k^+ = x p^+$, so we apply the lightcone non-diagonal metric and the on-shell condition, $k^2 = 0$
        sampled_kz = (1 / np.sqrt(2)) * (sampled_x * E - (sampled_kperp ** 2) / (2 * sampled_x * E))

        emission_momentum = np.column_stack([sampled_kx, sampled_ky, sampled_kz])

        if N_samples == 1:
            return total_integral, np.reshape(emission_momentum, 3)
        else:
            return total_integral, emission_momentum


# ==============================================================================
# Optimization tools
# ==============================================================================
def find_learning_rate(
        model: nn.Module,
        dataloader: DataLoader,
        optimizer: torch.optim.Optimizer,
        config: TrainingConfig,
        start_lr: float = 1e-4,    # narrower range start
        end_lr: float = 1e-1,      # narrower range end
        n_steps: int = 150,        # far more steps for resolution
        smoothing: float = 0.9,   # heavy EMA, standard for LR finders
        diverge_threshold: float = 4.0,  # stop if loss exceeds this × best
) -> Tuple[list, list, list]:
    """
    Learning rate range test (Smith 2015).

    Sweeps LR exponentially from start_lr to end_lr over n_steps batches,
    recording the smoothed loss at each step.

    Returns
    -------
    lrs : list of float
        Learning rates tested
    losses : list of float
        Smoothed loss at each learning rate
    """
    import copy
    original_model_state     = copy.deepcopy(model.state_dict())
    original_optimizer_state = copy.deepcopy(optimizer.state_dict())

    for pg in optimizer.param_groups:
        pg['lr'] = start_lr

    lr_multiplier = (end_lr / start_lr) ** (1.0 / n_steps)

    lrs:    list[float] = []
    losses: list[float] = []
    raw_losses: list[float] = []
    smoothed_loss: Optional[float] = None
    best_loss = float('inf')

    model.train()
    data_iter = iter(dataloader)

    print(f"  LR range: {start_lr:.1e} → {end_lr:.1e}  |  "
          f"steps: {n_steps}  |  "
          f"LR multiplier/step: {lr_multiplier:.4f}")
    print(f"  {'Step':>5}  {'LR':>10}  {'Raw Loss':>12}  {'Smoothed':>12}")
    print(f"  {'-'*5}  {'-'*10}  {'-'*12}  {'-'*12}")

    for step in range(n_steps):
        try:
            inputs, targets, weights = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            inputs, targets, weights = next(data_iter)

        inputs  = inputs.to(config.device)
        targets = targets.to(config.device)
        weights = weights.to(config.device)

        optimizer.zero_grad()
        loss, components = compute_loss(model, inputs, targets, weights, config)
        loss.backward()
        optimizer.step()

        raw_loss = components['mse']

        # Bias-corrected EMA — prevents the first few steps from being
        # artificially low just because smoothed_loss started at zero.
        if smoothed_loss is None:
            smoothed_loss = raw_loss
        else:
            smoothed_loss = smoothing * smoothed_loss + (1.0 - smoothing) * raw_loss
        bias_correction = 1.0 - smoothing ** (step + 1)
        loss_debiased = smoothed_loss / bias_correction

        current_lr = optimizer.param_groups[0]['lr']
        lrs.append(current_lr)
        losses.append(loss_debiased)
        raw_losses.append(raw_loss)

        if step % 10 == 0:
            print(f"  {step:>5}  {current_lr:>10.2e}  "
                  f"{raw_loss:>12.4e}  {loss_debiased:>12.4e}")

        # Track best and stop on divergence
        if loss_debiased < best_loss:
            best_loss = loss_debiased
        if loss_debiased > diverge_threshold * best_loss:
            print(f"  Loss diverged at step {step}, LR={current_lr:.2e} — stopping early.")
            break

        for pg in optimizer.param_groups:
            pg['lr'] *= lr_multiplier

    # Restore everything
    model.load_state_dict(original_model_state)
    optimizer.load_state_dict(original_optimizer_state)

    return lrs, losses, raw_losses


def plot_lr_finder(lrs: list, losses: list, raw_losses: Optional[list] = None):
    """Plot the LR finder curve and print the suggested learning rate."""
    import matplotlib.pyplot as plt

    lrs    = np.array(lrs)
    losses = np.array(losses)

    # Suggested LR: steepest negative gradient on the smoothed curve,
    # but only in the region before the minimum (ignore the diverging tail)
    min_idx  = np.argmin(losses)
    # Clamp the window: ignore the first 10% of steps (EMA not yet settled)
    # and everything after the minimum
    start_idx = max(1, len(lrs) // 10)
    lrs_w    = lrs[start_idx : min_idx + 1]
    losses_w = losses[start_idx : min_idx + 1]

    if len(lrs_w) > 1:
        gradients     = np.gradient(losses_w, np.log10(lrs_w))
        suggested_idx = np.argmin(gradients) + start_idx
        suggested_lr  = lrs[suggested_idx]
    else:
        suggested_idx = max(0, int(min_idx - 1))
        suggested_lr  = lrs[suggested_idx]

    print(f"\nLR Finder Results:")
    print(f"  Minimum loss at LR:              {lrs[min_idx]:.2e}")
    print(f"  Suggested LR (steepest descent): {suggested_lr:.2e}")
    print(f"  Recommended starting LR (1/3):   {suggested_lr / 3:.2e}")

    fig, ax = plt.subplots(figsize=(9, 5))

    # Smoothed curve (primary)
    ax.plot(lrs, losses, linewidth=2.5, label='Smoothed loss', color='steelblue')

    # Raw curve (secondary)
    ax.plot(lrs, raw_losses, ls=":", lw=1, label='Raw loss', color='steelblue')

    # Vertical lines
    ax.axvline(lrs[min_idx], color='gray', linestyle=':', linewidth=1.2,
               label=f'Loss minimum  ({lrs[min_idx]:.2e})')
    ax.axvline(suggested_lr, color='red', linestyle='--', linewidth=1.5,
               label=f'Suggested LR  ({suggested_lr:.2e})')
    ax.axvline(suggested_lr / 3, color='orange', linestyle='--', linewidth=1.5,
               label=f'Recommended (÷3)  ({suggested_lr / 3:.2e})')

    ax.set_xscale('log')
    ax.set_xlabel('Learning Rate (log scale)')
    ax.set_ylabel('Smoothed Loss (bias-corrected EMA)')
    ax.set_title('Learning Rate Range Test')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('lr_finder.png', dpi=150)
    print(f"  Plot saved to lr_finder.png")
    plt.show()

    return suggested_lr


# ==============================================================================
# Demo inference
# ==============================================================================
def demo_inference(config: TrainingConfig):
    """Demonstrate how to load and use the trained model."""

    print("=" * 70)
    print("RADIATION EMULATOR INFERENCE DEMO")
    print("=" * 70)

    # Load the trained model
    emulator = RadiationEmulatorInference(
        model_file=config.model_file,
        normalization_file=config.normalization_file,
        device="cpu",  # Use CPU for inference demo
    )

    # Generate some test points
    n_points = 1000
    rng = np.random.default_rng(123)

    test_inputs = {
        'x': rng.uniform(0.1, 0.9, n_points),
        'kx': rng.uniform(-3.0, 3.0, n_points),
        'ky': rng.uniform(-3.0, 3.0, n_points),
        'E': rng.uniform(10.0, 80.0, n_points),
        'z0': rng.uniform(0.0, 3.0, n_points),
        'u_perp': rng.uniform(0.0, 0.5, n_points),
        'T': rng.uniform(0.2, 0.4, n_points),
        'g': rng.uniform(1.8, 2.2, n_points),
    }

    # Run inference
    print(f"\nRunning inference on {n_points} test points...")

    import time
    t0 = time.time()
    predictions = emulator.predict_dict(test_inputs)
    dt = time.time() - t0

    print(f"Inference time: {dt * 1000:.2f} ms ({dt / n_points * 1e6:.2f} µs per point)")
    print(f"Prediction shape: {predictions.shape}")
    print(f"Prediction range: [{predictions.min():.4e}, {predictions.max():.4e}]")

    # Example: Apply Casimir factor for quarks
    CF_QUARK = 4 / 3
    CF_GLUON = 3

    I_quark = CF_QUARK * predictions
    I_gluon = CF_GLUON * predictions

    print(f"\nWith Casimir factors applied:")
    print(f"  Quark (CF=4/3): [{I_quark.min():.4e}, {I_quark.max():.4e}]")
    print(f"  Gluon (CF=3):   [{I_gluon.min():.4e}, {I_gluon.max():.4e}]")

    # Example: Single point query
    print("\n" + "-" * 70)
    print("Single point query example:")
    print("-" * 70)

    single_pred = emulator.predict(
        x=np.array([0.3]),
        kx=np.array([1.0]),
        ky=np.array([0.5]),
        E=np.array([50.0]),
        z0=np.array([0.0]),
        u_perp=np.array([0.3]),
        T=np.array([0.3]),
        g=np.array([2.0]),
    )

    print(f"Input: x=0.3, kx=1.0, ky=0.5, E=50, z0=0, zf=5, u_perp=0.3, T=0.3, g=2.0")
    print(f"Predicted I (no CF): {single_pred:.4e}")
    print(f"Predicted I (quark): {CF_QUARK * single_pred:.4e}")
    print(f"Predicted I (gluon): {CF_GLUON * single_pred:.4e}")


# ==============================================================================
# Main entry point
# ==============================================================================
if __name__ == "__main__":
    default_config = TrainingConfig()  # Load default values from class description
    parser = argparse.ArgumentParser(description="Train or run inference with radiation emulator")
    parser.add_argument("--data-file", type=str, default=default_config.data_file, help="Training data file")
    parser.add_argument("--model-file", type=str, default=default_config.model_file, help="Model output file")
    parser.add_argument("--normalization-file", type=str, default=default_config.normalization_file, help="Model normalization output file")
    parser.add_argument("--transform", type=str, default=default_config.transform, help="Type of transform on data")
    parser.add_argument("--epochs", type=int, default=default_config.n_epochs, help="Number of training epochs")
    parser.add_argument("--hidden-dim", type=int, default=default_config.hidden_dim, help="Hidden layer dimension")
    parser.add_argument("--n-layers", type=int, default=default_config.n_layers, help="Number of hidden layers")
    parser.add_argument("--learning-rate", type=float, default=default_config.learning_rate,
                        help="Initial learning rate")
    parser.add_argument("--find-lr", action="store_true", help="Run LR range test and exit")
    parser.add_argument("--n-workers", type=int, default=default_config.num_workers,
                        help="Number of workers for data loading")
    parser.add_argument("--checkpoint-file", type=str, default=default_config.checkpoint_file,
                        help="Path for the resume checkpoint (read + written during training)")
    parser.add_argument("--checkpoint-interval", type=int, default=default_config.checkpoint_interval,
                        help="Save a resume checkpoint every N epochs")
    parser.add_argument("--no-resume", action="store_true",
                        help="Ignore any existing checkpoint and start from scratch")
    args = parser.parse_args()

    config = TrainingConfig(
        data_file=args.data_file,
        model_file=args.model_file,
        normalization_file=args.normalization_file,
        n_epochs=args.epochs,
        hidden_dim=args.hidden_dim,
        n_layers=args.n_layers,
        learning_rate=args.learning_rate,
        run_lr_finder=args.find_lr,
        transform=args.transform,
        num_workers=args.n_workers,
        checkpoint_file=args.checkpoint_file,
        checkpoint_interval=args.checkpoint_interval,
    )

    # --no-resume: delete checkpoint so training starts from scratch
    if args.no_resume and os.path.exists(config.checkpoint_file):
        os.remove(config.checkpoint_file)
        print(f"Deleted existing checkpoint: {config.checkpoint_file}")

    # # Example overfitting-style training config
    # config = TrainingConfig(
    #     data_file=args.data_file,
    #     model_file=args.model_file,
    #     train_fraction=0.99,  # Use almost all data for training
    #     weight_decay=0.0,  # No regularization
    #     n_epochs=500,  # Train longer
    #     patience=500,  # Don't early stop
    #     hidden_dim=512,  # Larger model
    #     n_layers=8,  # Deeper
    # )

    # Make data directory, if not present
    Path(config.model_file).parent.mkdir(exist_ok=True)

    # Train the model
    train_model(config)
