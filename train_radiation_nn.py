"""
train_radiation_nn.py

Trains a neural network emulator for the medium-induced radiation intensity
distribution using precomputed training data.

Features:
- Loads training data from HDF5 file
- Normalizes inputs and log-transforms outputs
- Uses importance sampling weights for unbiased training
- Enforces soft physics constraints (positivity, k_y symmetry)
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
from torch.utils.data import Dataset, DataLoader, random_split
from dataclasses import dataclass
from typing import Optional, Tuple, Dict
import json
import signal
import os
import time
from pathlib import Path
import threading
import queue

# Set use of float32 for matmuls to improve performance. Trial feature -- may impact accuracy, depending on problem.
torch.set_float32_matmul_precision('high')

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
    mirror: bool = True  # Whether to mirror data in ky.

    # Architecture
    hidden_dim: int = 256
    n_layers: int = 5
    activation: str = "silu"  # silu, relu, tanh, gelu

    # Training
    batch_size: int = 256
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    dropout_p: float = 0.1
    n_epochs: int = 500
    patience: int = 20  # Early stopping patience

    # Physics constraints
    lambda_positivity: float = 0.0  # Weight for positivity penalty -- 0 so that we don't enforce positivity
    lambda_ky_symmetry: float = 0.0  # Weight for k_y symmetry penalty -- data enforces it -- data is mirrored.
    lambda_uv: float = 0.01  # Weight for UV decay loss term
    lambda_uv_power: float = 0.0  # Weight for UV power-law loss term
    # UV threshold: only penalise points where kt^2 > uv_kt2_threshold (in GeV^2).
    # Should be set comfortably above mu_D^2 ~ g^2 T^2 ~ (2*GeV)^2*(0.3GeV)^2 ~ 0.36 GeV^2.
    # A safe default is 4.0 GeV^2 (kt > 2 GeV).
    uv_kt_threshold: float = 10.0

    # Output
    model_file: str = "data/radiation_emulator.pt"
    normalization_file: str = "data/radiation_normalization.json"
    checkpoint_file: str = "data/radiation_training_checkpoint.pt"  # Resume checkpoint
    checkpoint_interval: int = 1  # Save resume checkpoint every N epochs

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers: int = 4  # DataLoader worker processes
    compile: bool = False  # Whether or not to compile the NN during training. Will not work on older Discovery GPUs

    # Utilities
    run_lr_finder: bool = False


# ==============================================================================
# Dataset
# ==============================================================================
class RadiationDataset(Dataset):
    """
    HDF5-backed dataset. Each worker process opens its own file handle.

    Design
    ------
    No background threads are used. The dataset stores only:
      - valid_indices: the HDF5 row numbers that passed the finite-value filter
      - normalization statistics (X_mean, X_std, y_mean, y_std, etc.)

    The HDF5 file is opened lazily in __getitem__ on first access within each
    DataLoader worker process. Because each worker is a *separate process*
    (fork), there is no shared HDF5 file handle and no locking contention.

    For num_workers=0 (single process), the handle is opened once and reused.

    Mirroring in k_y is virtual: logical index i >= n_valid returns the same
    HDF5 row as i - n_valid, with the ky column negated. No data is duplicated.

    RAM usage
    ---------
    Only valid_indices (int64 array, ~8 bytes/row) lives in RAM permanently.
    For 240M valid rows that is ~1.9 GB. If this is still too much, pass
    subsample_indices=True to store every Nth index, though this is rarely needed.
    """

    FEATURE_NAMES = ['x', 'kx', 'ky', 'E', 'z0', 'zf', 'u_perp', 'T', 'g']
    N_FEATURES    = len(FEATURE_NAMES)
    KY_IDX        = 2

    def __init__(
        self,
        data_file: str,
        transform_output: str = "arcsinh",
        chunk_rows: int = 65_536,
        mirror: bool = True,
    ):
        self.data_file  = data_file
        self.transform  = transform_output
        self.chunk_rows = chunk_rows
        self.mirror     = mirror
        self.epsilon    = 1e-10

        # Per-worker file handle — None until first __getitem__ call in that worker
        self._hdf5_file = None

        print(f"Scanning {data_file} ...")
        self._scan_file()

        n_logical = 2 * self.n_valid if self.mirror else self.n_valid
        print(f"Dataset ready  |  valid={self.n_valid:,}  "
              f"logical={n_logical:,}  transform={self.transform}")

    # ------------------------------------------------------------------
    # One-time scan for valid indices + normalization stats
    # ------------------------------------------------------------------
    def _scan_file(self):
        CHUNK     = self.chunk_rows
        valid_parts: list[np.ndarray] = []
        X_mean    = np.zeros(self.N_FEATURES, dtype=np.float64)
        X_M2      = np.zeros(self.N_FEATURES, dtype=np.float64)
        n_welford = 0
        MAX_Y     = 4_000_000
        y_res: list[np.ndarray] = []
        y_res_n   = 0
        w_sum     = np.float64(0)
        w_count   = 0

        with h5py.File(self.data_file, 'r') as f:
            n_raw = int(f['I'].shape[0])
            for start in range(0, n_raw, CHUNK):
                end = min(start + CHUNK, n_raw)
                sl  = slice(start, end)

                y_c    = f['I'][sl].astype(np.float32)
                yerr_c = f['I_err'][sl].astype(np.float32)
                X_c    = np.empty((end - start, self.N_FEATURES), dtype=np.float32)
                for j, name in enumerate(self.FEATURE_NAMES):
                    X_c[:, j] = f[name][sl]

                ok = (np.isfinite(y_c) & np.isfinite(yerr_c) &
                      np.all(np.isfinite(X_c), axis=1))
                valid_parts.append(np.where(ok)[0] + start)

                X_v = X_c[ok].astype(np.float64)
                y_v = y_c[ok]
                w_v = f['weight'][sl][ok].astype(np.float32)

                m = len(X_v)
                if m > 0:
                    bm  = X_v.mean(axis=0)
                    bM2 = ((X_v - bm) ** 2).sum(axis=0)
                    cn  = n_welford + m
                    d   = bm - X_mean
                    X_mean  = (n_welford * X_mean + m * bm) / cn
                    X_M2   += bM2 + d ** 2 * n_welford * m / cn
                    n_welford = cn

                if y_res_n < MAX_Y:
                    y_res.append(y_v)
                    y_res_n += len(y_v)

                w_sum   += w_v.sum()
                w_count += len(w_v)

        self.valid_indices = np.concatenate(valid_parts)   # shape (N_valid,), int64
        self.n_valid       = len(self.valid_indices)

        self.X_mean  = X_mean.astype(np.float32)
        self.X_std   = (np.sqrt(np.maximum(X_M2 / max(n_welford - 1, 1), 0))
                        + 1e-8).astype(np.float32)
        self.weight_mean = float(w_sum / max(w_count, 1))

        y_sample    = np.concatenate(y_res) if y_res else np.array([0.0], dtype=np.float32)
        self.f0     = float(np.std(y_sample)) or 1.0
        y_t         = self._transform_y(y_sample)
        self.y_mean = float(np.mean(y_t))
        self.y_std  = float(np.std(y_t) + 1e-8)

        print(f"  {n_raw:,} raw  →  {self.n_valid:,} valid"
              f"  |  weight_mean={self.weight_mean:.3e}  f0={self.f0:.3e}")

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
    # Lazy per-worker file handle
    # ------------------------------------------------------------------
    def _get_file(self) -> "h5py.File":
        """Return the open HDF5 handle for this process, opening it if needed."""
        if self._hdf5_file is None:
            self._hdf5_file = h5py.File(self.data_file, 'r')
        return self._hdf5_file

    # ------------------------------------------------------------------
    # DataLoader worker initialisation hook
    # ------------------------------------------------------------------
    def worker_init(self, worker_id: int):
        """
        Call this as DataLoader's worker_init_fn to reset the file handle in
        each forked worker process.  Without this, forked workers inherit the
        parent's open handle, which causes HDF5 corruption / hangs.

        Usage:
            DataLoader(..., worker_init_fn=dataset.worker_init)
        """
        # Each forked worker must open its own handle — reset the inherited one.
        self._hdf5_file = None

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return 2 * self.n_valid if self.mirror else self.n_valid

    def __getitem__(self, idx: int):
        mirrored = self.mirror and (idx >= self.n_valid)
        raw_idx  = idx - self.n_valid if mirrored else idx
        hdf_idx  = int(self.valid_indices[raw_idx])

        f = self._get_file()
        x_raw = np.array([f[name][hdf_idx] for name in self.FEATURE_NAMES],
                         dtype=np.float32)
        y_raw = float(f['I'][hdf_idx])
        w_raw = float(f['weight'][hdf_idx])

        x_norm = (x_raw - self.X_mean) / self.X_std
        y_norm = float((self._transform_y(np.array([y_raw], dtype=np.float32))[0]
                        - self.y_mean) / self.y_std)
        w_norm = w_raw / self.weight_mean

        if mirrored:
            x_norm[self.KY_IDX] = -x_norm[self.KY_IDX]

        return (
            torch.from_numpy(x_norm),
            torch.tensor(y_norm, dtype=torch.float32),
            torch.tensor(w_norm, dtype=torch.float32),
        )

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------
    def shutdown(self):
        """Close the HDF5 file handle if open. Call after training."""
        if self._hdf5_file is not None:
            try:
                self._hdf5_file.close()
            except Exception:
                pass
            self._hdf5_file = None

    # ------------------------------------------------------------------
    # Normalization export
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
            input_dim: int = 9,
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

        # Output layer
        self.output_layer = nn.Linear(hidden_dim, 1)

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
            Input tensor of shape (batch_size, 9) with normalized features

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch_size, 1) with normalized log-intensity
        """
        # Input layer
        h = self.input_act(self.input_layer(x))

        # Hidden layers with skip connections every 2 layers
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
def compute_loss(
        model: nn.Module,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        weights: torch.Tensor,
        config: TrainingConfig,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Compute weighted MSE loss with physics constraints.

    Returns total loss and dictionary of individual loss components.
    """
    predictions = model(inputs).squeeze(-1)

    """
    Weighted MSE loss
    """
    mse = (weights * (predictions - targets) ** 2).mean()

    """
    Positivity enforcement loss

    (in normalized space, this is approximate)
    We don't use this, it's just here as a sample loss term
    """
    positivity = torch.relu(-predictions - 2.0).mean()  # Threshold at -2 std

    """
    k_y symmetry enforcement loss 

    k_y symmetry is already built into the training data (mirrored samples).
    This is an optional soft loss penalty on top.
    """
    if config.lambda_ky_symmetry > 0.0:
        # Create inputs with flipped k_y (index 2 in feature list)
        inputs_ky_flipped = inputs.clone()
        inputs_ky_flipped[:, 2] = -inputs_ky_flipped[:, 2]  # ky -> -ky
        predictions_flipped = model(inputs_ky_flipped).squeeze(-1)
        ky_symmetry = ((predictions - predictions_flipped) ** 2).mean()
    else:
        ky_symmetry = torch.tensor(0.0, device=inputs.device)

    """
    UV decay enforcement loss

    Enforces f(params, k_x, k_y) -> 0 as k_perp -> inf.
    """
    if config.lambda_uv > 0.0:
        k_perp_max = 1e4  # Maximum k_perp value to consider
        n_uv_samples = 64  # Number of samples to draw from UV region

        # Get shape and device of inputs
        B = inputs.shape[0]
        device = inputs.device

        # Sample k_perp log-uniformly in the UV region
        log_k = torch.empty(n_uv_samples, device=device).uniform_(
            math.log(config.uv_kt_threshold),
            math.log(k_perp_max),
        )
        k_perp = torch.exp(log_k)

        # Isotropic: random azimuthal angle phi
        phi = torch.empty(n_uv_samples, device=device).uniform_(0, 2 * math.pi)
        k_x_uv = k_perp * torch.cos(phi)
        if config.mirror:
            k_y_uv = k_perp * torch.sin(phi)  # k_y positive or negative
        else:
            k_y_uv = np.abs(k_perp * torch.sin(phi))  # make k_y always positive

        # Tile the other 7 parameters from random rows of the training batch
        idx = torch.randint(0, B, (n_uv_samples,), device=device)
        uv_params = inputs[idx].clone()  # (n_uv_samples, 9)
        uv_params[:, 1] = k_x_uv  # overwrite k_x
        uv_params[:, 2] = k_y_uv  # overwrite k_y

        uv_output = model(uv_params).squeeze(-1)  # (n_uv_samples,) or (n_uv_samples, 1)

        # Use log weight to penalize nonzero result at larger k_perp values more
        log_weight = 2 * log_k - 2 * math.log(config.uv_kt_threshold)
        uv_decay = (log_weight * uv_output ** 2).mean()
    else:
        uv_decay = torch.tensor(0.0, device=inputs.device)

    """
    UV power law behavior enforcement loss

    Enforces f(alpha * k_perp) / f(k_perp) ~ alpha^{-power}.
    Robust to overall normalisation; constrains the UV shape directly.
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
        k_perp = k_perp_base * torch.exp(
            torch.empty(n_uv_power_law_samples, device=device).uniform_(0, 3)
        )
        phi = torch.empty(n_uv_power_law_samples, device=device).uniform_(0, 2 * torch.pi)

        # Low-k point
        base_params[:, 1] = k_perp * torch.cos(phi)
        base_params[:, 2] = k_perp * torch.sin(phi)
        f_low = model(base_params).squeeze()

        # High-k point (same direction, same other params)
        high_params = base_params.clone()
        high_params[:, 1] = (alpha * k_perp) * torch.cos(phi)
        high_params[:, 2] = (alpha * k_perp) * torch.sin(phi)
        f_high = model(high_params).squeeze()

        # Target ratio: f(alpha*k) / f(k) = alpha^{-power}
        expected_ratio = alpha ** (-power)
        # Use log-ratio loss for numerical stability; avoids division-by-zero
        log_ratio = torch.log(torch.abs(f_high) + 1e-30) - torch.log(torch.abs(f_low) + 1e-30)
        target_log_ratio = torch.full_like(log_ratio, torch.log(torch.tensor(expected_ratio)))

        uv_power_law = nn.functional.mse_loss(log_ratio, target_log_ratio)
    else:
        uv_power_law = torch.tensor(0.0, device=inputs.device)

    # Total loss
    total_loss = (
            mse
            + config.lambda_positivity * positivity
            + config.lambda_ky_symmetry * ky_symmetry
            + config.lambda_uv * uv_decay
            + config.lambda_uv_power * uv_power_law
    )

    # Return loss components for logging
    components = {
        'mse': mse.item(),
        'positivity': positivity.item(),
        'ky_symmetry': ky_symmetry.item(),
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
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    total_mse = 0.0
    n_batches = 0

    for inputs, targets, weights in dataloader:
        inputs = inputs.to(config.device)
        targets = targets.to(config.device)
        weights = weights.to(config.device)

        optimizer.zero_grad()
        loss, components = compute_loss(model, inputs, targets, weights, config)
        loss.backward()
        optimizer.step()

        total_loss += components['total']
        total_mse += components['mse']
        n_batches += 1

    return {
        'loss': total_loss / n_batches,
        'mse': total_mse / n_batches,
    }


def validate(
        model: nn.Module,
        dataloader: DataLoader,
        config: TrainingConfig,
) -> Dict[str, float]:
    """Validate the model."""
    model.eval()
    total_loss = 0.0
    total_mse = 0.0
    n_batches = 0

    with torch.no_grad():
        for inputs, targets, weights in dataloader:
            inputs = inputs.to(config.device)
            targets = targets.to(config.device)
            weights = weights.to(config.device)

            _, components = compute_loss(model, inputs, targets, weights, config)

            total_loss += components['total']
            total_mse += components['mse']
            n_batches += 1

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
    fixed_state_dict = {k.removeprefix('_orig_mod.'): v for k, v in raw_state_dict.items()}
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
    ckpt = torch.load(path, map_location=device)
    state_dict = {k.removeprefix('_orig_mod.'): v for k, v in ckpt['model_state_dict'].items()}
    model.load_state_dict(state_dict)
    optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    scheduler.load_state_dict(ckpt['scheduler_state_dict'])
    start_epoch    = ckpt['epoch'] + 1          # resume at the *next* epoch
    best_val_loss  = ckpt['best_val_loss']
    patience_counter = ckpt['patience_counter']
    print(f"  Resumed from epoch {ckpt['epoch'] + 1}  |  best val loss: {best_val_loss:.4e}")
    return start_epoch, best_val_loss, patience_counter

# ==============================================================================
# Main training function
# ==============================================================================
def train_model(config: TrainingConfig):
    """Train the radiation emulator model."""

    print("=" * 70)
    print("RADIATION EMULATOR TRAINING")
    print("=" * 70)
    print(f"Device: {config.device}")
    print(f"Data file: {config.data_file}")
    print()

    # Load dataset
    dataset = RadiationDataset(config.data_file, transform_output=config.transform)

    # Save normalization parameters
    print("Saving normalization...")
    norm_params = dataset.get_normalization_params()
    with open(config.normalization_file, 'w') as f:
        json.dump(norm_params, f, indent=2)

    # Split into train/validation
    n_train = int(len(dataset) * config.train_fraction)
    n_val = len(dataset) - n_train
    train_dataset, val_dataset = random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )

    print(f"Training samples: {n_train}")
    print(f"Validation samples: {n_val}")

    # Create dataloaders
    if config.device == "cuda":
        train_loader = DataLoader(
            train_dataset, batch_size=config.batch_size, shuffle=True,
            num_workers=config.num_workers,
            persistent_workers=config.num_workers > 0,
            pin_memory=True,
            worker_init_fn=dataset.worker_init,
        )
        val_loader = DataLoader(
            val_dataset, batch_size=config.batch_size, shuffle=False,
            num_workers=config.num_workers,
            persistent_workers=config.num_workers > 0,
            pin_memory=True,
            worker_init_fn=dataset.worker_init,
        )
    elif config.device == "cpu":
        train_loader = DataLoader(
            train_dataset, batch_size=config.batch_size, shuffle=True,
            num_workers=config.num_workers,
            persistent_workers=config.num_workers > 0,
            worker_init_fn=dataset.worker_init,
        )
        val_loader = DataLoader(
            val_dataset, batch_size=config.batch_size, shuffle=False,
            num_workers=config.num_workers,
            persistent_workers=config.num_workers > 0,
            worker_init_fn=dataset.worker_init,
        )

    # Create model
    model = RadiationEmulator(
        input_dim=len(dataset.FEATURE_NAMES),
        hidden_dim=config.hidden_dim,
        n_layers=config.n_layers,
        activation=config.activation,
        dropout_p=config.dropout_p,
    ).to(config.device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    # Optimizer -- applies various strategies that change the way we use our neurons
    # Controls usage of dropout and L2 regularization to encourage generalization instead of memorization
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    # LR finder -- to be run before the main training loop, finds optimal learning rate
    # Looks for minima in the loss as function of learning rate, returns rate just before minima in loss function
    if config.run_lr_finder:
        print("\nRunning LR range test...")
        print("-" * 70)
        lrs, losses = find_learning_rate(model, train_loader, optimizer, config)
        suggested_lr = plot_lr_finder(lrs, losses)
        print(f"\nRe-run with --learning-rate {suggested_lr / 3:.2e} (1/3 of suggested)")
        return model, dataset.get_normalization_params()

    # Scheduler -- controls the variation of learning rate over training epochs
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10,
        min_lr=6.24e-5,
    )

    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0
    start_epoch = 0

    # ── Resume from checkpoint if one exists ──────────────────────────────
    if os.path.exists(config.checkpoint_file):
        start_epoch, best_val_loss, patience_counter = load_training_checkpoint(
            config.checkpoint_file, model, optimizer, scheduler, config.device
        )
    else:
        print("No checkpoint found – starting from scratch.")

    if hasattr(torch, 'compile') and config.compile == True:
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

    print("\nStarting training...")
    print("-" * 70)

    try:
        for epoch in range(start_epoch, config.n_epochs):
            current_epoch[0] = epoch

            if _sigterm_received[0]:
                print("Exiting cleanly after SIGTERM.")
                break

            # Train
            print("Training...")
            train_metrics = train_epoch(model, train_loader, optimizer, config)

            # Validate
            print("Validating...")
            val_metrics = validate(model, val_loader, config)

            # Update scheduler
            scheduler.step(val_metrics['mse'])  # Scheduler tracks mse, not the overall loss.

            # Print progress
            current_lr = optimizer.param_groups[0]['lr']
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
                    k.removeprefix('_orig_mod.'): v
                    for k, v in raw_state_dict.items()
                }

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

    FEATURE_NAMES = ['x', 'kx', 'ky', 'E', 'z0', 'zf', 'u_perp', 'T', 'g']

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
        checkpoint = torch.load(model_file, map_location=device)
        model_config = checkpoint['config']

        self.model = RadiationEmulator(
            input_dim=model_config['input_dim'],
            hidden_dim=model_config['hidden_dim'],
            n_layers=model_config['n_layers'],
            activation=model_config['activation'],
            dropout_p=model_config.get('dropout_p', 0.0),  # backward compatible
        ).to(device)

        try:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        except RuntimeError:
            # Strip torch.compile's '_orig_mod.' prefix if present (backward compatible)
            state_dict = {
                k.removeprefix('_orig_mod.'): v
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
        if hasattr(torch, 'compile') and compile == True:
            self.model = torch.compile(self.model)

        if not quiet:
            print(f"Loaded model from {model_file}")
            print(f"  Validation loss: {checkpoint['val_loss']:.4e}")
            print(f"  Trained for {checkpoint['epoch'] + 1} epochs")

    def predict(
            self,
            x: np.ndarray,  # unitless
            kx: np.ndarray,  # in GeV
            ky: np.ndarray,  # in GeV
            E: np.ndarray,  # in GeV
            z0: np.ndarray,  # in invGeV
            zf: np.ndarray,  # in invGeV
            u_perp: np.ndarray,  # unitless
            T: np.ndarray,  # in GeV
            g: np.ndarray,  # unitless
    ) -> np.ndarray:
        # Stack inputs directly into a contiguous float32 C-array,
        # then wrap with from_numpy (zero-copy) before sending to device.
        inputs = np.column_stack([x, kx, ky, E, z0, zf, u_perp, T, g]).astype(
            np.float32, order='C', copy=False
        )
        inputs_tensor = torch.from_numpy(inputs).to(self.device, non_blocking=True)

        # Normalize (X_mean / X_std are already on self.device)
        inputs_norm = (inputs_tensor - self.X_mean) / self.X_std

        with torch.no_grad():
            predictions_norm = self.model(inputs_norm).squeeze()

        # Stay on CPU as a numpy array; avoid an extra .cpu() call
        # by using the tensor directly when device is already CPU.
        pn = predictions_norm if self.device == "cpu" else predictions_norm.cpu()
        predictions_transformed = pn.numpy() * self.y_std + self.y_mean

        if self.transform == "log":
            predictions = np.exp(np.abs(predictions_transformed)) - self.epsilon
            predictions = np.sign(predictions_transformed) * predictions
        elif self.transform == "arcsinh":
            predictions = self.f0 * np.sinh(predictions_transformed)
        else:
            predictions = predictions_transformed

        return predictions

    def predict_raw(self, inputs: np.ndarray) -> np.ndarray:
        """
        Faster entry-point when the caller can supply a pre-stacked
        (N, 9) float32 array in feature order [x, kx, ky, E, z0, zf, u_perp, T, g].

        Skips the np.column_stack overhead, which matters when predict()
        is called thousands of times with the same grid layout.
        """
        inputs_tensor = torch.from_numpy(
            np.asarray(inputs, dtype=np.float32, order='C')
        ).to(self.device, non_blocking=True)

        inputs_norm = (inputs_tensor - self.X_mean) / self.X_std

        with torch.no_grad():
            predictions_norm = self.model(inputs_norm).squeeze()

        pn = predictions_norm if self.device == "cpu" else predictions_norm.cpu()
        predictions_transformed = pn.numpy() * self.y_std + self.y_mean

        if self.transform == "log":
            predictions = np.exp(np.abs(predictions_transformed)) - self.epsilon
            predictions = np.sign(predictions_transformed) * predictions
        elif self.transform == "arcsinh":
            predictions = self.f0 * np.sinh(predictions_transformed)
        else:
            predictions = predictions_transformed

        return predictions

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
            kx=inputs['kx'],
            ky=inputs['ky'],
            E=inputs['E'],
            z0=inputs['z0'],
            zf=inputs['zf'],
            u_perp=inputs['u_perp'],
            T=inputs['T'],
            g=inputs['g'],
        )

    def compute_dNd3k_grid(self,
                           E: float,
                           z0: float,
                           zf: float,
                           u_perp: float,
                           T: float,
                           g: float,
                           kz_values: np.ndarray,
                           kx_values: np.ndarray,
                           ky_values: np.ndarray
                           ) -> (np.ndarray):
        """
        Computes a complete grid of dN/(dxdkxdky) shaped as (kx, ky, kz) and returns the grid, sans CR
        """

        # Create a meshgrid and compute x values for each grid point
        kx_grid, ky_grid, kz_grid = np.meshgrid(kx_values, ky_values, kz_values, indexing='ij')
        x_grid = (1 / ((E**2 )*np.sqrt(2))) * (E*kz_grid + np.sqrt(E**2 * (kz_grid**2 + kx_grid**2 + ky_grid**2)))
        n_pts = x_grid.size

        # Build input grid once
        grid_inputs = np.column_stack([
            x_grid.ravel(), kx_grid.ravel(), ky_grid.ravel(),
            np.full(n_pts, E), np.full(n_pts, z0), np.full(n_pts, zf),
            np.full(n_pts, u_perp), np.full(n_pts, T), np.full(n_pts, g),
        ]).astype(np.float32)

        # Get predictions -- returns as flat array of dI/dxd^2k_perp points
        I_nn_flat = self.predict_raw(grid_inputs)

        # Reshape flat predictions back onto the 3D grid
        I_nn = I_nn_flat.reshape(x_grid.shape)  # shape: (n_kx, n_ky, n_x)

        # Set any unphysical x coordinate values to zero
        I_nn[x_grid > 1.0] = 0

        # Compute dN/dxd^2k_perp by dividing out energy of each grid point
        N_nn = I_nn / (E*x_grid)  # Still needs casimir factor

        # Convert to dN/d^3k
        # Compute and apply Jacobian for x -> kz : dN/dkz = (dN/dx) * |dx/dkz|
        # Broadcast shapes: kx/ky -> (m, 1, 1) and (1, m, 1), x -> (1, 1, n)
        dkz_dx = (1 / np.sqrt(2)) * (E + (kx_grid ** 2 + ky_grid ** 2) / (2 * x_grid ** 2 * E))
        jacobian = 1.0 / dkz_dx  # |dx/dkz|, shape broadcasts to (m, m, n)
        N_nn = N_nn * jacobian

        # Return as dN/d^3k
        return N_nn

    def sample_emission(self,
                        E: np.ndarray,
                        z0: np.ndarray,
                        zf: np.ndarray,
                        u_perp: np.ndarray,
                        T: np.ndarray,
                        g: np.ndarray,
                        N_samples: int = 1,
                        rng: np.random.Generator = np.random.default_rng()
                        ) -> (float, np.ndarray):
        """
        Computes a complete grid in x, kx, ky and returns a 3D inverse CDF sample value for x, kx, ky vector.
        """

        # Grid of x, kx, ky values
        max_kx_ky = 5  # Maybe should be dependent on energy, needs testing.
        x_values = np.logspace(-4, 0, 10)
        kx_values = np.linspace(-max_kx_ky, max_kx_ky, 50)
        ky_values = np.linspace(0, max_kx_ky, 50)
        x_grid, kx_grid, ky_grid = np.meshgrid(x_values, kx_values, ky_values, indexing='ij')
        n_pts = x_grid.size

        # Build input grid once
        grid_inputs = np.column_stack([
            x_grid.ravel(), kx_grid.ravel(), ky_grid.ravel(),
            np.full(n_pts, E), np.full(n_pts, z0), np.full(n_pts, zf),
            np.full(n_pts, u_perp), np.full(n_pts, T), np.full(n_pts, g),
        ]).astype(np.float32)

        # Get predictions
        I_nn_flat = self.predict_raw(grid_inputs)

        # --- Reshape flat predictions back onto the 3D grid ---
        I_nn = I_nn_flat.reshape(x_grid.shape)  # shape: (n_x, n_kx, n_ky)

        # -------------------------------------------------------
        # 1. INTEGRATION
        #    Integrate over ky first, then kx, then x (in log-space).
        #    np.trapz(y, x) integrates y along the last axis by default.
        # -------------------------------------------------------
        # Integrate over ky (axis 2)
        I_kx_x = 2 * np.trapezoid(I_nn, ky_values,
                                  axis=2)  # shape: (n_x, n_kx), multiply by two for symmetric -ky half of grid
        # Integrate over kx (axis 1)
        I_x = np.trapezoid(I_kx_x, kx_values, axis=1)  # shape: (n_x,)
        # Integrate over x in log-space (accounts for log-spaced grid) -- includes Jacobian, factor of x
        total_integral = np.trapezoid(I_x * x_values, np.log(x_values))  # scalar

        # print(f"Total integral: {total_integral:.6e}")

        # -------------------------------------------------------
        # 2. SAMPLING
        #    Treat I_nn as an (unnormalized) 3D probability density
        #    and draw N_samples points (x, kx, ky) from it.
        # -------------------------------------------------------

        # Build a normalized flat PDF, then a CDF
        I_flat = I_nn.ravel()
        I_flat_pos = np.clip(I_flat, 0, None)  # ensure non-negative
        pdf = I_flat_pos / I_flat_pos.sum()  # normalize to sum to 1
        cdf = np.cumsum(pdf)  # build CDF
        cdf[-1] = 1.0  # Force exact upper bound — removes all floating point slop

        # Draw uniform samples and find where they land in the CDF
        uniform_samples = rng.uniform(size=N_samples)  # Should use the global RNG...
        sign_samples = rng.choice([-1, 1], size=N_samples)
        flat_indices = np.searchsorted(cdf, uniform_samples)  # shape: (N_samples,)

        # Convert flat indices back to 3D grid indices
        ix, ikx, iky = np.unravel_index(flat_indices, I_nn.shape)

        # Look up the corresponding coordinate values
        sampled_x = x_values[ix]
        sampled_kx = kx_values[ikx]
        sampled_ky = sign_samples * ky_values[
            iky]  # Apply a random sign to the ky values to simulate symmetric -ky values

        # Actual longitudinal component of the momentum vector
        # We know $k^+ = x p^+$, so we apply the lightcone non-diagonal metric and the on-shell condition, $k^2 = 0$
        sampled_kz = (1 / np.sqrt(2)) * (sampled_x * E - ((sampled_kx ** 2 + sampled_ky ** 2) / (2 * sampled_x * E)))

        # sampled_kz = 0  # Just give zero kz for now
        # mag = np.sqrt(sampled_kx**2 + sampled_ky**2 + sampled_kz**2)
        # emission_momentum = total_integral * np.column_stack([sampled_kx/mag, sampled_ky/mag, sampled_kz/mag])
        # emission_momentum =  np.column_stack([sampled_x*E * sampled_kx / mag, sampled_x*E * sampled_ky / mag, sampled_x*E * sampled_kz / mag])
        emission_momentum = np.column_stack([sampled_kx, sampled_ky, sampled_kz])

        # # Apply random uniform jitter to point, to simulate continuous sampling
        # sampled_x += np.random.uniform(-dx / 2, dx / 2, size=N_samples)
        # sampled_kx += np.random.uniform(-dkx / 2, dkx / 2, size=N_samples)
        # sampled_ky += np.random.uniform(-dky / 2, dky / 2, size=N_samples)

        # print(f"Sampled {N_samples} points.")
        # print(f"  x  range: [{sampled_x.min():.4e},  {sampled_x.max():.4e}]")
        # print(f"  kx range: [{sampled_kx.min():.3f}, {sampled_kx.max():.3f}]")
        # print(f"  ky range: [{sampled_ky.min():.3f}, {sampled_ky.max():.3f}]")

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
        start_lr: float = 1e-5,
        end_lr: float = 3e-2,
        n_steps: int = 20,
        smoothing: float = 0.3,
) -> Tuple[list, list]:
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
    # Save original model and optimizer state so we can restore after the test
    import copy
    original_model_state = copy.deepcopy(model.state_dict())
    original_optimizer_state = copy.deepcopy(optimizer.state_dict())

    # Set starting LR
    for pg in optimizer.param_groups:
        pg['lr'] = start_lr

    lr_multiplier = (end_lr / start_lr) ** (1.0 / n_steps)

    lrs = []
    losses = []
    smoothed_loss = None
    best_loss = float('inf')

    model.train()
    data_iter = iter(dataloader)

    for step in range(n_steps):
        # Get next batch, cycling through dataloader if needed
        try:
            inputs, targets, weights = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            inputs, targets, weights = next(data_iter)

        inputs = inputs.to(config.device)
        targets = targets.to(config.device)
        weights = weights.to(config.device)

        optimizer.zero_grad()
        loss, components = compute_loss(model, inputs, targets, weights, config)
        loss.backward()
        optimizer.step()

        raw_loss = components['mse']

        # Exponential smoothing to reduce noise
        if smoothed_loss is None:
            smoothed_loss = raw_loss
        else:
            smoothed_loss = smoothing * raw_loss + (1.0 - smoothing) * smoothed_loss

        current_lr = optimizer.param_groups[0]['lr']
        lrs.append(current_lr)
        losses.append(smoothed_loss)

        # Stop early if loss has exploded (5x best seen so far)
        if smoothed_loss < best_loss:
            best_loss = smoothed_loss
        if smoothed_loss > 5.0 * best_loss:
            print(f"  Loss diverged at LR={current_lr:.2e}, stopping early.")
            break

        # Increase LR for next step
        for pg in optimizer.param_groups:
            pg['lr'] *= lr_multiplier

    # Restore model and optimizer to pre-test state
    model.load_state_dict(original_model_state)
    optimizer.load_state_dict(original_optimizer_state)

    return lrs, losses


def plot_lr_finder(lrs: list, losses: list):
    """Plot the LR finder curve and print the suggested learning rate."""
    import matplotlib.pyplot as plt
    import numpy as np

    lrs = np.array(lrs)
    losses = np.array(losses)

    # Only consider the region before the loss minimum (descending slope)
    min_loss_idx = np.argmin(losses)
    lrs_descending = lrs[:min_loss_idx + 1]
    losses_descending = losses[:min_loss_idx + 1]

    # Suggest the LR at the point of steepest negative gradient on the descending slope
    if len(lrs_descending) > 1:
        gradients = np.gradient(losses_descending, np.log10(lrs_descending))
        suggested_idx = np.argmin(gradients)
        suggested_lr = lrs_descending[suggested_idx]
    else:
        # Fallback: use the LR just before the minimum
        suggested_idx = max(0, min_loss_idx - 1)
        suggested_lr = lrs[suggested_idx]

    print(f"\nLR Finder Results:")
    print(f"  Suggested LR (steepest descent): {suggested_lr:.2e}")
    print(f"  Loss at suggested LR:            {losses[suggested_idx]:.4e}")
    print(f"  (Consider using ~1/3 to 1/10 of this as your starting LR)")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(lrs, losses, linewidth=2)
    ax.axvline(suggested_lr, color='red', linestyle='--',
               label=f'Suggested LR: {suggested_lr:.2e}')
    ax.set_xscale('log')
    ax.set_xlabel('Learning Rate (log scale)')
    ax.set_ylabel('Smoothed Loss')
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
        'zf': rng.uniform(3.0, 8.0, n_points),
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
        zf=np.array([5.0]),
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
    parser.add_argument("--inference-only", action="store_true", help="Skip training, demo inference only")
    parser.add_argument("--data-file", type=str, default=default_config.data_file, help="Training data file")
    parser.add_argument("--model-file", type=str, default=default_config.model_file, help="Model output file")
    parser.add_argument("--transform", type=str, default=default_config.transform, help="Whether to log transform data")
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

    if args.inference_only:
        # Demo inference only
        demo_inference(config)
    else:
        # Train the model
        train_model(config)

        # Then demo inference
        print("\n")
        demo_inference(config)
