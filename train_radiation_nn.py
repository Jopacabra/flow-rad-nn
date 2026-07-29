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
    mirror: bool = True  # Whether to mirror data in ky.

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
    self.X_data : np.ndarray, shape (N_valid, 9), float32   — input features
    self.y_data : np.ndarray, shape (N_valid,),   float32   — raw intensity
    self.w_data : np.ndarray, shape (N_valid,),   float32   — importance weights

    For 127M valid rows at float32:
      X_data  ~  4.6 GB  (127M × 9 × 4 bytes)
      y_data  ~  0.5 GB  (127M × 4 bytes)
      w_data  ~  0.5 GB  (127M × 4 bytes)
      total   ~  5.6 GB

    Mirroring in k_y is virtual (same as before): logical index i >= n_valid
    returns the same row as i - n_valid with the ky column negated. No data
    is duplicated in memory.
    """

    FEATURE_NAMES = ['x', 'kx', 'ky', 'E', 'z0', 'u_perp', 'T', 'g']
    N_FEATURES    = len(FEATURE_NAMES)
    KY_IDX        = 2

    def __init__(
        self,
        data_file: str,
        transform_output: str = "arcsinh",
        mirror: bool = True,
    ):
        self.data_file = data_file
        self.transform = transform_output
        self.mirror    = mirror
        self.epsilon   = 1e-10

        print(f"Loading {data_file} into RAM ...")
        self._scan_file()

        n_logical = 2 * self.n_valid if self.mirror else self.n_valid
        print(f"Dataset ready  |  valid={self.n_valid:,}  "
              f"logical={n_logical:,}  transform={self.transform}")

    # ------------------------------------------------------------------
    # Full load into RAM — no copies, no HDF5 handles kept open
    # ------------------------------------------------------------------
    def _scan_file(self):
        with h5py.File(self.data_file, 'r') as f:
            n_raw = int(f['I'].shape[0])
            print(f"  Reading {n_raw:,} rows from HDF5 ...")

            # ----------------------------------------------------------
            # Step 1 — allocate output arrays ONCE, read directly into them.
            # h5py supports out= for direct no-copy reads.
            # ----------------------------------------------------------
            X_raw = np.empty((n_raw, self.N_FEATURES), dtype=np.float32)
            for j, name in enumerate(self.FEATURE_NAMES):
                f[name].read_direct(X_raw, dest_sel=np.s_[:, j])

            y_raw  = np.empty(n_raw, dtype=np.float32)
            f['I'].read_direct(y_raw)

            w_raw  = np.empty(n_raw, dtype=np.float32)
            f['weight'].read_direct(w_raw)

            i_err  = np.empty(n_raw, dtype=np.float32)
            f['I_err'].read_direct(i_err)

        # ----------------------------------------------------------
        # Step 2 — validity mask; filter once, then immediately
        # let the raw arrays go out of scope so GC can free them.
        # ----------------------------------------------------------
        print("  Filtering invalid rows ...")
        ok = (
            np.isfinite(y_raw)
            & np.isfinite(i_err)
            & np.all(np.isfinite(X_raw), axis=1)
        )
        # i_err is only needed for the validity check — drop it now
        del i_err

        n_valid = int(ok.sum())
        print(f"  {n_raw:,} raw  →  {n_valid:,} valid")

        # Fancy-index once per array to keep only valid rows.
        # This creates new contiguous arrays; the originals are then deleted.
        self.X_data = X_raw[ok]   # (N_valid, 9)  float32
        del X_raw

        self.y_data = y_raw[ok]   # (N_valid,)    float32
        del y_raw

        self.w_data = w_raw[ok]   # (N_valid,)    float32
        del w_raw

        # Boolean mask is no longer needed either
        del ok

        self.n_valid = n_valid

        # ----------------------------------------------------------
        # Step 3 — normalization stats (no extra data copies needed;
        # computed directly from self.X_data / self.y_data)
        # ----------------------------------------------------------
        print("  Computing normalization statistics ...")

        # Input features: mean and std in float64 for numerical precision,
        # then cast back to float32 for storage.
        X64 = self.X_data.astype(np.float64)          # temporary; freed below
        self.X_mean = X64.mean(axis=0).astype(np.float32)
        self.X_std  = (X64.std(axis=0) + 1e-8).astype(np.float32)
        del X64

        # Importance weight normalization scalar
        self.weight_mean = float(self.w_data.mean())

        # Output transform + normalization
        self.f0 = float(np.std(self.y_data)) or 1.0
        y_t = self._transform_y(self.y_data)  # returns a new array
        self.y_mean = float(y_t.mean())
        self.y_std = float(y_t.std() + 1e-8)

        # Pre-compute fully normalized outputs and store them.
        # This avoids recomputing _transform_y() on every __getitem__ call.
        self.y_norm_data = ((y_t - self.y_mean) / self.y_std).astype(np.float32)
        del y_t

        # Pre-compute fully normalized inputs and store them.
        # This avoids per-sample subtraction/division in __getitem__.
        self.X_norm_data = ((self.X_data - self.X_mean) / self.X_std).astype(np.float32)

        # Pre-compute normalized weights.
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
    def __len__(self) -> int:
        return 2 * self.n_valid if self.mirror else self.n_valid

    def __getitem__(self, idx: int):
        mirrored = self.mirror and (idx >= self.n_valid)
        raw_idx = idx - self.n_valid if mirrored else idx

        # --- inputs ---
        # Copy the pre-normalized row so we can flip ky if needed.
        x_norm = self.X_norm_data[raw_idx].copy()

        if mirrored:
            x_norm[self.KY_IDX] = -x_norm[self.KY_IDX]

        # --- output (pre-computed, just index) ---
        y_norm = self.y_norm_data[raw_idx]

        # --- weight (pre-computed, just index) ---
        w_norm = self.w_norm_data[raw_idx]

        return (
            torch.from_numpy(x_norm),  # zero-copy after .copy()
            torch.tensor(y_norm, dtype=torch.float32),
            torch.tensor(w_norm, dtype=torch.float32),
        )

    def __getitems__(self, indices: list[int]) -> list:
        indices = np.asarray(indices, dtype=np.intp)

        if self.mirror:
            mirrored_mask = indices >= self.n_valid
            raw_indices = np.where(mirrored_mask, indices - self.n_valid, indices)
        else:
            mirrored_mask = np.zeros(len(indices), dtype=bool)
            raw_indices = indices

        X_batch = self.X_norm_data[raw_indices].copy()
        y_batch = self.y_norm_data[raw_indices]
        w_batch = self.w_norm_data[raw_indices]

        if self.mirror and mirrored_mask.any():
            X_batch[mirrored_mask, self.KY_IDX] *= -1

        # ✅ Return three tensors — collate_fn receives one "sample" and passes
        # it straight through without any further stacking.
        return [
            torch.from_numpy(X_batch),
            torch.from_numpy(y_batch.copy()),  # copy() to own the memory
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
    mirror : bool
        Whether to expose a mirrored (ky → -ky) copy of every sample.
        Typically True for training, False for validation.
    """

    def __init__(
        self,
        dataset: RadiationDataset,
        physical_indices: np.ndarray,
        mirror: bool = True,
    ):
        self.ds      = dataset
        self.idx     = np.asarray(physical_indices, dtype=np.intp)
        self.mirror  = mirror
        self.n_phys  = len(self.idx)

    def __len__(self) -> int:
        return 2 * self.n_phys if self.mirror else self.n_phys

    def __getitems__(self, positions: list[int]) -> list:
        positions = np.asarray(positions, dtype=np.intp)

        if self.mirror:
            mirrored_mask = positions >= self.n_phys
            local_idx     = np.where(mirrored_mask, positions - self.n_phys, positions)
        else:
            mirrored_mask = np.zeros(len(positions), dtype=bool)
            local_idx     = positions

        # Map local positions → physical dataset indices (one fancy-index)
        phys_idx = self.idx[local_idx]

        # Single C-level read per array — fully contiguous because self.idx
        # was built from a sorted or pre-shuffled contiguous slice
        X_batch = self.ds.X_norm_data[phys_idx].copy()   # copy needed for ky flip
        y_batch = self.ds.y_norm_data[phys_idx].copy()
        w_batch = self.ds.w_norm_data[phys_idx].copy()

        if self.mirror and mirrored_mask.any():
            X_batch[mirrored_mask, RadiationDataset.KY_IDX] *= -1

        return [
            torch.from_numpy(X_batch),
            torch.from_numpy(y_batch),
            torch.from_numpy(w_batch),
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
            input_dim: int = 8,
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
        max_frac = 0.25  # fraction of kperp^2_max at which to start penalizing.
        rng = np.random.default_rng()
        n_uv_samples = 64  # Number of samples to draw from UV region

        # Get shape and device of inputs
        B = inputs.shape[0]
        device = inputs.device

        # Tile the other 7 parameters from random rows of the training batch
        idx = torch.randint(0, B, (n_uv_samples,), device=device)
        uv_params = inputs[idx].clone()  # (n_uv_samples, 8)
        energy_params = uv_params[:, 3].cpu().numpy()

        # Sample k_perp log-uniformly in the UV region, depending on energy of sample point
        log_k = []
        for i in np.arange(0, len(energy_params)):
            try:
                log_k.append(rng.uniform(math.log(max_frac * (energy_params[i])**2),  # Begin penalizing at 0.25 *E^2
                math.log((energy_params[i])**2),  # use E^2 as maximum k_perp.
                ))
            except:  # If you get something crazy, just use the energy squared.
                log_k.append((energy_params[i])**2)
        log_k = torch.tensor(log_k, device=device)  # Make list into torch tensor
        k_perp = torch.exp(log_k)

        # Isotropic: random azimuthal angle phi
        phi = torch.empty(n_uv_samples, device=device).uniform_(0, 2 * math.pi)
        k_x_uv = k_perp * torch.cos(phi)
        if config.mirror:
            k_y_uv = k_perp * torch.sin(phi)  # k_y positive or negative
        else:
            k_y_uv = np.abs(k_perp * torch.sin(phi))  # make k_y always positive

        uv_params[:, 1] = k_x_uv  # overwrite k_x
        uv_params[:, 2] = k_y_uv  # overwrite k_y

        uv_output = model(uv_params).squeeze(-1)  # (n_uv_samples,) or (n_uv_samples, 1)

        # Use log weight to penalize nonzero result at larger k_perp values more
        log_esqr = torch.tensor(np.log(max_frac * energy_params**2), device=device)
        log_weight = 2 * log_k - 2 * log_esqr
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
        target_log_ratio = torch.full_like(log_ratio, torch.log(torch.tensor(expected_ratio)).item())

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

    t0 = time.time()
    for i, (inputs, targets, weights) in enumerate(dataloader):
        # if i % 10000 == 0 and i != 0:
        #     print(f"  Batch {i}/{len(dataloader)}  [avg {(time.time() - t0)/i:.3f}s/batch]")

        # Send tensors to device
        inputs = inputs.to(config.device)
        targets = targets.to(config.device)
        weights = weights.to(config.device)

        # Compute loss and step optimizer
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

    # Split on *physical* indices before mirroring, so each subset gets a
    # contiguous, cache-friendly slice of the underlying arrays.
    n_phys_train = int(dataset.n_valid * config.train_fraction)

    rng_split = np.random.default_rng(42)
    perm = rng_split.permutation(dataset.n_valid).astype(np.intp)

    train_idx = perm[:n_phys_train]
    val_idx = perm[n_phys_train:]

    # Training subset: mirrored (doubles the effective dataset size).
    # Validation subset: no mirroring — we want unbiased coverage.
    train_dataset = RadiationSubset(dataset, train_idx, mirror=config.mirror)
    val_dataset = RadiationSubset(dataset, val_idx, mirror=False)

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
            # print("Training...")
            train_metrics = train_epoch(model, train_loader, optimizer, config)

            # Validate
            # print("Validating...")
            val_metrics = validate(model, val_loader, config)

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

    FEATURE_NAMES = ['x', 'kx', 'ky', 'E', 'z0', 'u_perp', 'T', 'g']

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
            x: np.ndarray,  # unitless
            kx: np.ndarray,  # in GeV
            ky: np.ndarray,  # in GeV
            E: np.ndarray,  # in GeV
            z0: np.ndarray,  # in invGeV -- zf hardcoded at dtau = 0.1 fm
            u_perp: np.ndarray,  # unitless
            T: np.ndarray,  # in GeV
            g: np.ndarray,  # unitless
    ) -> np.ndarray:
        # Stack inputs directly into a contiguous float32 C-array,
        # then wrap with from_numpy (zero-copy) before sending to device.
        inputs = np.column_stack([x, kx, ky, E, z0, u_perp, T, g]).astype(
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
        # print(f"normed predictions: {np.mean(pn)}")
        # print(pn)
        predictions_transformed = pn.numpy() * self.y_std + self.y_mean
        print(f"denormed: {np.mean(predictions_transformed)}")

        if self.transform == "log":
            predictions = np.exp(np.abs(predictions_transformed)) - self.epsilon
            predictions = np.sign(predictions_transformed) * predictions
        elif self.transform == "arcsinh":
            predictions = self.f0 * np.sinh(predictions_transformed)
        else:
            predictions = predictions_transformed

        print(f"after transform: {np.mean(predictions)}")
        return predictions

    def predict_raw(self, inputs: np.ndarray) -> np.ndarray:
        """
        Faster entry-point when the caller can supply a pre-stacked
        (N, 9) float32 array in feature order [x, kx, ky, E, z0, u_perp, T, g].

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
            np.full(n_pts, E), np.full(n_pts, z0),
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
                        u_perp: np.ndarray,
                        T: np.ndarray,
                        g: np.ndarray,
                        N_samples: int = 1,
                        rng: np.random.Generator = np.random.default_rng()
                        ) -> tuple[float, np.ndarray]:
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
            np.full(n_pts, E), np.full(n_pts, z0),
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

    # Train the model
    train_model(config)
