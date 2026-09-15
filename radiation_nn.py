"""
radiation_nn.py

Trains a neural network emulator to replicate the leading term of the medium-induced radiation intensity distribution
from a fast probe in a transversely flowing quark gluon plasma using precomputed training data.

The transformations and normalization are specialized to reproduce a huge dynamic range of intensity values while
keeping weights small. The model architecture incorporates a physics-informed UV power law envelope that sets in
at large k_perp.

Values at small x are many orders of magnitude larger than those at large x.

Features:
- Loads training data from HDF5 file
- Normalizes input parameters to zero mean and unit variance
- Normalizes input targets to zero mean and unit variance, then transforms input targets
- Trains NN, enforcing some physics constraints via architecture and loss terms
- Saves trained model for deployment
- Provides inference methods to predict radiation intensity distribution from model file

Usage:
    python radiation_nn.py                    # Train the model
"""

import argparse
import math
import numpy as np
import h5py
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
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
# Module level constants -- ensure that this matches the dataset you're training on
HBARC = 0.197327
DELTA_Z = 0.1 / HBARC  # hardcoded 0.1 fm to GeV^{-1}
@dataclass
class TrainingConfig:
    """
    Configuration for neural network training.
    """

    # Data
    data_file: str = "data/sobol_batch_0002.h5"
    train_fraction: float = 0.8
    transform: str = "arcsinh"
    transform_f0: float | None = None  # Scale for transformation, if applicable. None uses small % of data

    # Architecture
    hidden_dim: int = 256
    n_layers: int = 5
    activation: str = "silu"  # silu, relu, tanh, gelu
    n_harmonics: int = 0  # number of cos/sin harmonics of the cosine phase per head (0 = no harmonic architecture)
    n_harmonics_dz: int = 0  # same for sinc phase -- this phase typically varies much more slowly when z0 >> Delta_z

    # Training
    batch_size: int = 4096
    learning_rate: float = 2.72e-3
    weight_decay: float = 1e-4
    dropout_p: float = 0.1
    n_epochs: int = 500
    patience: int = 20  # Early stopping patience

    # Physics constraints
    lambda_A0: float = 1.0  # Weight for MSE of A0 head
    lambda_A1: float = 1.0  # Weight for MSE of A1 head
    lambda_A0_int: float = 1.0  # Weight for integral-importance absolute-error term (A0)
    lambda_A1_int: float = 1.0  # Weight for integral-importance absolute-error term (A1)
    lambda_uv: float = 0.0  # Weight for UV decay loss term
    # UV threshold: only penalise points where kt^2 > uv_kt2_threshold (in GeV^2).
    # Should be set comfortably above mu_D^2 ~ g^2 T^2 ~ (2*GeV)^2*(0.3GeV)^2 ~ 0.36 GeV^2.
    # A safe default is 10.0 GeV^2 (kperp > ~3.16 GeV).
    uv_kt_threshold: float = 10.0

    # Output
    model_file: str = "data/radiation_emulator.pt"
    checkpoint_file: str = "data/radiation_training_checkpoint.pt"  # Resume checkpoint
    checkpoint_interval: int = 1  # Save resume checkpoint every N epochs

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers: int = 4  # DataLoader worker processaes
    compile: bool = False  # Whether or not to compile the NN during training. Will not work on older Discovery GPUs

    # Utilities
    run_lr_finder: bool = False


#########################
# Input feature control #
#########################
def compute_input_features(x: np.ndarray, k_perp: np.ndarray, E: np.ndarray, z0: np.ndarray, u_perp: np.ndarray,
                           mu: np.ndarray):
    """
    Single source for computing NN-input features.
    Returns a dict of {name: array}, in the order they should be fed to the network.

    Everywhere should call this function to collate inputs for a NN pass or otherwise deal with input feature arrays.

    E: hard parton energy (GeV)
    x: momentum fraction of the emitted gluon
    k_perp: transverse momentum of the emitted gluon (GeV)
    z0: slab entry point (GeV)
    u_perp: transverse collective flow velocity (unitless fraction of c)
    mu: DeBye screening mass -- the medium scale

    Constructed features are frequencies and widths of the phase factors in the spectrum.

    """
    # Oscillatory phases
    omega_k = (k_perp**2) / (2 * x * E)
    omega_k_dz = omega_k * DELTA_Z / 2
    omega_k_midz = omega_k * (z0 + DELTA_Z / 2)
    omega_width = (mu**2) * DELTA_Z / (2 * x * E)

    # Create dictionary of computed input features
    input_dict = {
        'ln(x)': np.log(x),
        'ln(k_perp)': np.log(k_perp),
        'ln(E)': np.log(E),
        'z0': z0,
        'u_perp': u_perp,
        'mu': mu,
        # 'sinc(omega_k_dz)': np.sinc(omega_k_dz / np.pi) * np.pi,  # Numpy sinc is signal-processing normalized sinc
        # 'cos(omega_k_midz)': np.cos(omega_k_midz),
        # 'arcsinh(omega_width)': np.arcsinh(omega_width),
    }

    return input_dict


# ==============================================================================
# Dataset
# ==============================================================================
class RadiationDataset(Dataset):
    """
    All valid data is loaded into memory once at construction.

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
    """
    # Get the NN input feature names and number of features
    FEATURE_NAMES = list(
        compute_input_features(x=np.array([]), k_perp=np.array([]), E=np.array([]), z0=np.array([]),
                               u_perp=np.array([]), mu=np.array([])).keys())
    N_FEATURES = len(FEATURE_NAMES)
    print(f"  Using {N_FEATURES} features for NN input:")
    print(f"    {FEATURE_NAMES}")

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

    ######################
    # Load Data into RAM #
    ######################
    def _scan_file(self):
        with h5py.File(self.data_file, 'r') as f:
            # Get dataset feature names from the file
            self.RAW_FEATURE_NAMES = sorted(key for key in f.keys() if isinstance(f[key], h5py.Dataset))
            print(f"  Found {len(self.RAW_FEATURE_NAMES)} features in dataset:")
            print(f"    {self.RAW_FEATURE_NAMES}")

            n_raw = int(f['A0'].shape[0])
            print(f"  Reading {n_raw:,} rows from HDF5 ...")

            # Read raw (unmodified) columns straight from disk into pre-allocated arrays
            # Allocate arrays, then use h5py to read into them
            raw_cols = {}
            for name in self.RAW_FEATURE_NAMES:
                buf = np.empty(n_raw, dtype=np.float32)
                f[name].read_direct(buf)
                raw_cols[name] = buf

            A0_raw = np.empty(n_raw, dtype=np.float32)
            A1_raw = np.empty(n_raw, dtype=np.float32)
            f['A0'].read_direct(A0_raw)
            f['A1'].read_direct(A1_raw)

            w_raw = np.empty(n_raw, dtype=np.float32)
            f['weight'].read_direct(w_raw)

            A0_err = np.empty(n_raw, dtype=np.float32)
            A1_err = np.empty(n_raw, dtype=np.float32)
            f['A0_err'].read_direct(A0_err)
            f['A1_err'].read_direct(A1_err)

        # Compute the input parameter dictionary
        X_raw_dict = compute_input_features(raw_cols['x'], raw_cols['k_perp'], raw_cols['E'], raw_cols['z0'],
            raw_cols['u_perp'], raw_cols['mu'])

        # Stack into a an array of input arrays shape (N_points, N_FEATURES)
        X_raw = np.column_stack(list(X_raw_dict.values())).astype(np.float32)

        # Mask off any points where the integration when awry
        print("  Filtering invalid rows ...")
        ok = (
                np.isfinite(A0_raw)
                & np.isfinite(A1_raw)
                & np.isfinite(A0_err)
                & np.isfinite(A1_err)
                & np.all(np.isfinite(X_raw), axis=1)
        )

        n_valid = int(ok.sum())
        print(f"  {n_raw:,} raw  →  {n_valid:,} valid")

        # Apply mask to cut arrays
        self.X_data = X_raw[ok]
        del X_raw

        self.A0_data = A0_raw[ok]
        del A0_raw

        self.A1_data = A1_raw[ok]
        del A1_raw

        self.w_data = w_raw[ok]
        del w_raw

        # Keep the MC integration errors -- We will use these to have a noise-floor-dependent loss
        self.A0_err_data = A0_err[ok]
        self.A1_err_data = A1_err[ok]
        del A0_err, A1_err

        """
        Integral-importance weight: Jacobian from log-uniform sampling in
        (x, k_perp) to the physical measure (x dk_perp^2), i.e. how much
        each point contributes to the deployment-grid integral over
        dx d^2k_perp. 
        
        Important to match this to the actual deployment quadrature weights 
        if the grid isn't log-uniform in both variables.
        """
        x_ok = raw_cols['x'][ok]
        kperp_ok = raw_cols['k_perp'][ok]
        w_int_raw = (x_ok * kperp_ok ** 2).astype(np.float64)
        self.w_int_mean = float(w_int_raw.mean())
        self.w_int_data = (w_int_raw / self.w_int_mean).astype(np.float32)  # normalized to mean 1, for loss-scale stability

        del raw_cols
        del ok

        self.n_valid = n_valid

        # Compute and print some normalization statistics
        print("  Computing normalization statistics ...")
        X64 = self.X_data.astype(np.float64)
        self.X_mean = X64.mean(axis=0).astype(np.float32)
        self.X_std = (X64.std(axis=0) + 1e-8).astype(np.float32)
        del X64

        self.weight_mean = float(self.w_data.mean())

        # Transform input data
        if config.transform_f0 is None:
            self.f0 = np.percentile(np.abs(self.A0_data[self.A0_data != 0]), 1)
        else:
            self.f0 = float(config.transform_f0)
        A0_t = self._transform_y(self.A0_data)
        self.A0_mean = float(A0_t.mean())
        self.A0_std = float(A0_t.std() + 1e-8)
        A1_t = self._transform_y(self.A1_data)
        self.A1_mean = float(A1_t.mean())
        self.A1_std = float(A1_t.std() + 1e-8)

        # Normalize transformed input data to zero mean and unit variance
        self.A0_norm_data = ((A0_t - self.A0_mean) / self.A0_std).astype(np.float32)
        del A0_t
        self.A1_norm_data = ((A1_t - self.A1_mean) / self.A1_std).astype(np.float32)
        del A1_t
        self.X_norm_data = ((self.X_data - self.X_mean) / self.X_std).astype(np.float32)
        self.w_norm_data = (self.w_data / self.weight_mean).astype(np.float32)

        # Normalize MC integration error
        A0_deriv = self._transform_deriv(self.A0_data)
        A1_deriv = self._transform_deriv(self.A1_data)
        self.A0_err_norm_data = (self.A0_err_data * A0_deriv / self.A0_std).astype(np.float32)
        self.A1_err_norm_data = (self.A1_err_data * A1_deriv / self.A1_std).astype(np.float32)
        del self.A0_err_data, self.A1_err_data, A0_deriv, A1_deriv

        # Combined weight for the integral-importance absolute-error loss term:
        #   (transform-space residual)^2 * A0_intw  ≈  (physical residual)^2 * w_int
        # i.e. this directly approximates each point's contribution to the
        # *squared error of the total integral estimate*.
        self.A0_intw_data = (
                self.w_int_data.astype(np.float64)
                * (self.A0_std ** 2)
                * (self.f0 ** 2 + self.A0_data.astype(np.float64) ** 2)
        ).astype(np.float32)
        self.A1_intw_data = (
                self.w_int_data.astype(np.float64)
                * (self.A1_std ** 2)
                * (self.f0 ** 2 + self.A1_data.astype(np.float64) ** 2)
        ).astype(np.float32)

        print(f"  weight_mean={self.weight_mean:.3e}  f0={self.f0:.3e}")

    ####################
    # Output transform #
    ####################
    def _transform_y(self, y: np.ndarray) -> np.ndarray:
        if self.transform == "arcsinh":
            return np.arcsinh(y / self.f0).astype(np.float32)
        elif self.transform == "log":
            return (np.sign(y / self.f0) * np.log(np.abs(y / self.f0) + self.epsilon)).astype(np.float32)
        return y.astype(np.float32)

    def _transform_deriv(self, y: np.ndarray) -> np.ndarray:
        """
        Analytic derivative d[transform(y)]/dy, used to propagate the raw MC
        integration error (in physical A0/A1 units) into the transformed
        space the network is trained in, via the delta method:
            sigma_transformed ≈ |d(transform)/dy| * sigma_y
        """
        y64 = y.astype(np.float64)
        if self.transform == "arcsinh":
            # d/dy arcsinh(y/f0) = 1 / sqrt(f0^2 + y^2)
            return (1.0 / np.sqrt(self.f0 ** 2 + y64 ** 2)).astype(np.float32)
        elif self.transform == "log":
            # d/dy [sign(y/f0)*log(|y/f0|+eps)] = 1 / (|y| + f0*eps)
            return (1.0 / (np.abs(y64) + self.f0 * self.epsilon)).astype(np.float32)
        return np.ones_like(y, dtype=np.float32)

    #####################
    # Dataset protocols #
    #####################
    # Check the number of points in the dataset
    def __len__(self) -> int:
        return self.n_valid

    # Get a single data point by index
    def __getitem__(self, idx: int):
        x_norm = self.X_norm_data[idx]
        A0_norm = self.A0_norm_data[idx]
        A1_norm = self.A1_norm_data[idx]
        w_norm = self.w_norm_data[idx]
        A0_err_norm = self.A0_err_norm_data[idx]
        A1_err_norm = self.A1_err_norm_data[idx]
        A0_intw = self.A0_intw_data[idx]
        A1_intw = self.A1_intw_data[idx]

        return (
            torch.from_numpy(x_norm),
            torch.tensor(A0_norm, dtype=torch.float32),
            torch.tensor(A1_norm, dtype=torch.float32),
            torch.tensor(w_norm, dtype=torch.float32),
            torch.tensor(A0_err_norm, dtype=torch.float32),
            torch.tensor(A1_err_norm, dtype=torch.float32),
            torch.tensor(A0_intw, dtype=torch.float32),
            torch.tensor(A1_intw, dtype=torch.float32),
        )

    # Get a group of points by a list of indices. Uses fancy indexing to avoid loop overhead.
    def __getitems__(self, indices: list[int]) -> list:
        indices = np.asarray(indices, dtype=np.intp)

        X_batch = self.X_norm_data[indices]
        A0_batch = self.A0_norm_data[indices]
        A1_batch = self.A1_norm_data[indices]
        w_batch = self.w_norm_data[indices]
        A0_err_batch = self.A0_err_norm_data[indices]
        A1_err_batch = self.A1_err_norm_data[indices]
        A0_intw_batch = self.A0_intw_data[indices]
        A1_intw_batch = self.A1_intw_data[indices]

        return [
            torch.from_numpy(X_batch),
            torch.from_numpy(A0_batch.copy()),
            torch.from_numpy(A1_batch.copy()),
            torch.from_numpy(w_batch.copy()),
            torch.from_numpy(A0_err_batch.copy()),
            torch.from_numpy(A1_err_batch.copy()),
            torch.from_numpy(A0_intw_batch.copy()),
            torch.from_numpy(A1_intw_batch.copy()),
        ]

    ########################
    # Normalization export #
    ########################
    def get_normalization_params(self) -> Dict:
        return {
            'X_mean':        self.X_mean.tolist(),
            'X_std':         self.X_std.tolist(),
            'A0_mean':       self.A0_mean,
            'A0_std':        self.A0_std,
            'A1_mean':       self.A1_mean,
            'A1_std':        self.A1_std,
            'transform':     str(self.transform),
            'f0':            self.f0,
            'epsilon':       self.epsilon,
            'feature_names': self.FEATURE_NAMES,
        }

    # shutdown() is now a no-op — kept for API compatibility
    def shutdown(self):
        pass


@dataclass
class Normalization:
    X_mean: torch.Tensor
    X_std: torch.Tensor
    y_mean: Dict[str, float]   # {'A0': ..., 'A1': ...} -- keyed by target name
    y_std: Dict[str, float]
    feature_index: Dict[str, int]  # e.g. {'x': 0, 'k_perp': 1, 'E': 2, ...}

    @classmethod
    def from_dataset(cls, dataset: RadiationDataset, device) -> "Normalization":
        return cls(
            X_mean=torch.tensor(dataset.X_mean, device=device),
            X_std=torch.tensor(dataset.X_std, device=device),
            y_mean={'A0': dataset.A0_mean, 'A1': dataset.A1_mean},
            y_std={'A0': dataset.A0_std, 'A1': dataset.A1_std},
            feature_index={name: i for i, name in enumerate(dataset.FEATURE_NAMES)},
        )


# ==============================================================================
# Neural Network Model
# ==============================================================================
class RadiationEmulator(nn.Module):
    """
    RadiationEmulator architecture
    ------------------------------
    Two-head MLP (A0, A1) predicting the radiation intensity harmonics in a
    transformed, normalized "z-space". Each head is decomposed as:

        z_head = clamp( background + S_end - S_start ) + log_envelope(k_perp)

    1. Background: raw MLP scalar output per head (slowly-varying part).

    2. Oscillatory part -- physics-injected carrier, MLP-learned envelope:
       The medium-induced phase for a single q_perp is exactly
           Phi(q_perp) = 1 - sinc(omega*dz/2)*cos(omega*(z0+dz/2))
                       = 1 - [sin(theta_end) - sin(theta_start)] / (omega*dz)
       where omega = (k_perp-q_perp)^2/(2xE), and
           theta_end   = omega_midz + omega_dz  = omega*(z0+dz)   (phase at slab exit)
           theta_start = omega_midz - omega_dz  = omega*z0        (phase at slab entry)

       ---
       This difference of sines form turned out to have obvious catastrophic cancellation issues, so the smeared cosine
       and sinc form is preferable in principle. Difference of sines version kept in comment for now.
       ---

       i.e. the true structure is additive in two single-frequency terms sharing
       one instantaneous omega but accumulated over two different path lengths --
       NOT a product of independent cos/sinc series. The q_perp integral that
       produces the observed A0/A1 keeps this frequency (stationary phase at
       q_perp=0) but decoheres theta_end and theta_start at different rates
       (longer path -> more decoherence, LPM-like), so each needs its own
       independently-learned amplitude/phase envelope.

       theta_end, theta_start are recomputed EXACTLY in float64 each forward
       pass from the physical inputs (not inverted from the stored arcsinh
       features, to avoid amplifying float32 rounding by ~omega at small x).
       Two independent truncated Fourier series (n_harmonics, n_harmonics_dz
       terms) are evaluated at theta_end and theta_start respectively, with
       MLP-predicted coefficients (functions of all inputs) acting as the
       slowly-varying envelope amplitude/phase on top of the exact carrier.
       Combined additively (S_end - S_start), matching the exact single-q_perp
       identity (which is recovered exactly at n=1 with the right coefficient).

       Harmonic count must grow as x -> 0 since omega ~ k_perp^2/x oscillates
       arbitrarily fast; coefficients must stay smooth in k_perp for a
       truncated series to track the true envelope.

    3. Z-space soft clamp (tanh, scale Z_CLAMP): prevents float32 sinh overflow
       at huge dynamic range (small-x intensities are many decades above
       large-x). CAUTION: if raw pre-clamp values approach Z_CLAMP, tanh
       saturation clips the oscillatory carrier and injects spurious
       high-frequency distortion harmonics -- check `frac_saturated` (binned by
       x) if oscillation artifacts appear, especially at small x.

    4. UV power-law envelope: learnable exponent p in [P_MIN,P_MAX] and
       learnable transition scale k0^2 (anchored above the kinematic max
       k_perp^2), applied multiplicatively in z-space (additively in log) so
       every head decays smoothly to the UV threshold set by uv_kt_threshold.

    Output is de-normalized (y_mean/y_std) and, at inference, inverse-transformed
    via sinh(z)*f0 to recover the physical A0/A1 amplitudes.
    """

    IDX_K_PERP = RadiationDataset.FEATURE_NAMES.index('ln(k_perp)')
    IDX_MU     = RadiationDataset.FEATURE_NAMES.index('mu')
    IDX_X      = RadiationDataset.FEATURE_NAMES.index('ln(x)')
    IDX_E      = RadiationDataset.FEATURE_NAMES.index('ln(E)')
    IDX_Z0     = RadiationDataset.FEATURE_NAMES.index('z0')

    def __init__(
            self,
            input_dim: int = 7,
            hidden_dim: int = 256,
            n_layers: int = 5,
            activation: str = "silu",
            dropout_p: float = 0.1,
            transform: str = "arcsinh",
            n_harmonics: int = 1,
            n_harmonics_dz: int = 1,
    ):
        super().__init__()
        self.debug = False  # Whether to compute and print debug info during forward pass
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.transform = transform
        self.n_harmonics = n_harmonics
        self.n_harmonics_dz = n_harmonics_dz

        activations = {'silu': nn.SiLU, 'relu': nn.ReLU, 'tanh': nn.Tanh, 'gelu': nn.GELU}
        act_fn = activations.get(activation, nn.SiLU)

        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.input_act = act_fn()

        self.hidden_layers = nn.ModuleList()
        self.hidden_acts = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        for i in range(n_layers - 1):
            self.hidden_layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.hidden_acts.append(act_fn())
            self.dropouts.append(nn.Dropout(p=dropout_p))

        # Output layer -- target 2 features: A0, A1 fourier harmonic factors.
        # Each head predicts the harmonic amplitude scaled by 1/f0
        # Per head: 1 background + (A_n,B_n) per cos(omega_midz) harmonic + (C_m,D_m) per sinc(omega_dz) harmonic
        self._n_out_per_head = 1 + 2 * self.n_harmonics + 2 * self.n_harmonics_dz
        self.output_layer = nn.Linear(hidden_dim, 2 * self._n_out_per_head)

        # Learnable per-head output scale, in the same units as f0 Initialized to 1.0
        # so that at initialization -- with raw ~ O(1) from Xavier init -- the
        # head's behavior matches a plain linear layer (sinh(z) ~ z for small z).
        # Training is then free to learn a different per-head scale if A0, A1, A2
        # have substantially different typical magnitudes.
        # Currently does nothing.
        self.head_scale = nn.Parameter(torch.ones(2))

        # Learnable UV shape parameters
        self.raw_p = nn.Parameter(torch.full((2,), -3.0))  # one exponent per head (A0, A1)
        target_k0_over_mu = math.sqrt(10) / 0.4
        init_val = math.log(math.exp(target_k0_over_mu) - 1.0)  # invert softplus
        self.raw_k0_scale = nn.Parameter(torch.full((2,), init_val))
        self.P_MIN, self.P_MAX = 1.0, 8.0

        # Normalization buffers -- initialized with something close to an identity normalization
        self.register_buffer('X_mean', torch.zeros(input_dim))
        self.register_buffer('X_std',  torch.ones(input_dim))
        self.register_buffer('y_mean', torch.zeros(2))
        self.register_buffer('y_std',  torch.ones(2))
        self.register_buffer('f0',      torch.tensor(1.0))
        self.register_buffer('epsilon', torch.tensor(1e-10))

        self._init_weights()
        self._init_harmonic_scale()

    def _init_harmonic_scale(self):
        """Shrink initial weights feeding the harmonic slots so training starts close to the
        old background-only behavior and phases in oscillatory content gradually."""
        if self.n_harmonics == 0 and self.n_harmonics_dz == 0:
            return
        with torch.no_grad():
            W = self.output_layer.weight.view(2, self._n_out_per_head, -1)
            b = self.output_layer.bias.view(2, self._n_out_per_head)

            idx = 1
            if self.n_harmonics > 0:
                s = 1.0 / math.sqrt(self.n_harmonics)
                W[:, idx: idx + 2 * self.n_harmonics, :] *= s
                b[:, idx: idx + 2 * self.n_harmonics] *= s
                idx += 2 * self.n_harmonics
            if self.n_harmonics_dz > 0:
                s = 1.0 / math.sqrt(self.n_harmonics_dz)
                W[:, idx: idx + 2 * self.n_harmonics_dz, :] *= s
                b[:, idx: idx + 2 * self.n_harmonics_dz] *= s

    @staticmethod
    def _eval_harmonics(phase64: torch.Tensor, coeffs: torch.Tensor, n_max: int) -> torch.Tensor:
        """
        phase64 : (B,) float64          -- exact bare phase, one value per sample
        coeffs  : (B, heads, n_max, 2)  -- (A_n, B_n) pairs per head per harmonic
        n_max   : number of harmonics (0 => returns zeros)

        Returns (B, heads) = sum_n [ A_n*cos(n*phase) + B_n*sin(n*phase) ]
        """
        if n_max == 0:
            return torch.zeros(coeffs.shape[0], coeffs.shape[1], device=coeffs.device, dtype=coeffs.dtype)

        n_range = torch.arange(1, n_max + 1, device=phase64.device, dtype=torch.float64)  # (n_max,)
        phase_n = phase64.unsqueeze(-1) * n_range.unsqueeze(0)  # (B, n_max), float64
        cos_n = torch.cos(phase_n).to(coeffs.dtype)  # (B, n_max)
        sin_n = torch.sin(phase_n).to(coeffs.dtype)  # (B, n_max)

        return (coeffs[..., 0] * cos_n.unsqueeze(1)
                + coeffs[..., 1] * sin_n.unsqueeze(1)).sum(dim=-1)  # (B, heads)

    def _init_weights(self):
        """Initialize weights using Xavier initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def set_normalization(self, X_mean, X_std, y_mean, y_std, f0, epsilon=1e-10):
        """
        Populate normalization buffers from dataset statistics. Call once,
        right after model construction, before training starts.

        y_mean / y_std should be length-3 (A0, A1, A2); pad the unused A2
        slot with (0.0, 1.0) if you're not training a third head.
        """
        self.X_mean.copy_(torch.as_tensor(X_mean, dtype=torch.float32))
        self.X_std.copy_(torch.as_tensor(X_std,  dtype=torch.float32))
        self.y_mean.copy_(torch.as_tensor(y_mean, dtype=torch.float32))
        self.y_std.copy_(torch.as_tensor(y_std,  dtype=torch.float32))
        self.f0.copy_(torch.as_tensor(f0, dtype=torch.float32))
        self.epsilon.copy_(torch.as_tensor(epsilon, dtype=torch.float32))

    def forward(self, x_norm: torch.Tensor) -> torch.Tensor:
        h = self.input_act(self.input_layer(x_norm))
        h_block_in = h
        for i, (layer, act, drop) in enumerate(zip(self.hidden_layers, self.hidden_acts, self.dropouts)):
            if i % 2 == 0:
                h_block_in = h
            h = drop(act(layer(h)))
            if i % 2 == 1:
                h = h + h_block_in

        raw_out = self.output_layer(h).view(-1, 2, self._n_out_per_head)  # (B, heads=2, 1+2N+2M)

        # Get physical parameters
        lnx64 = (x_norm[:, self.IDX_X].double() * self.X_std[self.IDX_X].double() + self.X_mean[
            self.IDX_X].double())
        x64 = torch.exp(lnx64)
        E64 = torch.exp(
            x_norm[:, self.IDX_E].double() * self.X_std[self.IDX_E].double() + self.X_mean[self.IDX_E].double())
        kp64 = torch.exp(x_norm[:, self.IDX_K_PERP].double() * self.X_std[self.IDX_K_PERP].double() + self.X_mean[
            self.IDX_K_PERP].double())
        z064 = (x_norm[:, self.IDX_Z0].double() * self.X_std[self.IDX_Z0].double() + self.X_mean[
            self.IDX_Z0].double())

        x64_safe = x64.clamp(min=1e-12)  # guard only; x is physically bounded away from 0 by kinematic cuts

        background = raw_out[:, :, 0]  # (B, 2)
        if self.n_harmonics > 0:
            idx = 1
            harm_midz_coeffs = raw_out[:, :, idx: idx + 2 * self.n_harmonics].reshape(
                -1, 2, self.n_harmonics, 2)  # (B, heads, N, [A,B])
            idx += 2 * self.n_harmonics
            harm_dz_coeffs = raw_out[:, :, idx: idx + 2 * self.n_harmonics_dz].reshape(
                -1, 2, self.n_harmonics_dz, 2)  # (B, heads, M, [C,D])

            # --- Recompute EXACT bare phases from physical inputs, in float64 ---
            # Deliberately does NOT invert the stored arcsinh(omega_*) features.
            # That inversion amplifies float32 rounding error by a factor of ~omega itself, which
            # is precisely the divergent regime (x -> 0) we need to be most accurate in.
            omega_k64 = kp64 ** 2 / (2.0 * x64_safe * E64)  # (B,)
            omega_dz64 = omega_k64 * DELTA_Z / 2.0  # phase for the sinc(...) factor
            omega_midz64 = omega_k64 * (z064 + DELTA_Z / 2.0)  # phase for the cos(...) factor

            # # Exact slab-endpoint phases -- sinc(omega dz / 2)cos(omega(z0 + dz/2)) -> difference of sines
            # theta_end64 = omega_midz64 + omega_dz64  # omega_k * (z0 + Delta_z)
            # theta_start64 = omega_midz64 - omega_dz64  # omega_k * z0
            #
            # harm_end_sum = self._eval_harmonics(theta_end64, harm_midz_coeffs,  # (B, 2)
            #                                     self.n_harmonics)
            # harm_start_sum = self._eval_harmonics(theta_start64, harm_dz_coeffs,  # (B, 2)
            #                                       self.n_harmonics_dz)
            #
            # raw = background + harm_end_sum - harm_start_sum  # (B, 2) -- this sign is arbitrary -- can be absorbed by NN
            harm_midz_sum = self._eval_harmonics(omega_midz64, harm_midz_coeffs, self.n_harmonics)  # (B, 2)
            harm_dz_sum = self._eval_harmonics(omega_dz64, harm_dz_coeffs, self.n_harmonics_dz)  # (B, 2)

            raw = background + harm_midz_sum + harm_dz_sum  # (B, 2) -- same role as old "raw"
        else:
            raw = background # (B, 2)

        """
        Soft-clamp on the "z-space" raw output -- normalized+transformed
        Tuning this parameter is extremely important for not clipping huge vals at small x !!!

        If you end up with any amount of clipping in the z-space output, 
        you may need to adjust this parameter. This can manifest in many different signals, including apparent high-
        frequency oscillations in the very large value regions of the output. Use the debug print below when plotting
        the output to check for clipping. You should see zero here.

        float32 overflow occurs at sinh(~88.7), so keep this relatively well below 88.7.
        """
        Z_CLAMP = 80.0
        raw_z = Z_CLAMP * torch.tanh(raw / Z_CLAMP)  # (B, 2)

        """
        Envelop on output to enforce decay with k_perp -- learnable power law with learnable transition scale
        """
        # Physical parameters
        E = E64.to(raw.dtype)
        x = x64.to(raw.dtype)
        k_perp = kp64.to(raw.dtype)
        mu = x_norm[:, self.IDX_MU] * self.X_std[self.IDX_MU] + self.X_mean[self.IDX_MU]

        # Learnable exponent, bounded by P_MIN and P_MAX
        p = self.P_MIN + (self.P_MAX - self.P_MIN) * torch.sigmoid(self.raw_p)  # (2,)

        # Learnable transition scale -- anchored comfortably above ((Min[x^2, (1-x)^2]  * E^2) - mu^2),
        # near the physical UV threshold -- at which the envelope really starts to squeeze.
        # K = kperp_max^2 in GeV^2. Clamp for safety.
        K = (torch.minimum(x.unsqueeze(-1) ** 2, (1 - x.unsqueeze(-1)) ** 2) * E.unsqueeze(-1) ** 2) - mu.unsqueeze(
            -1) ** 2
        K = K.clamp(min=1e-6)

        # k0_sq is k0^2 in GeV^2 -- anchored comfortably above kperp_max^2 by the learnable (>=1) factor
        k0_sq = K * torch.nn.functional.softplus(self.raw_k0_scale)  # (B, 2), units GeV^2

        # Should be dimensionless: k_perp^2 [GeV^2] / k0_sq [GeV^2]
        log_ratio_sq = torch.log1p(k_perp.unsqueeze(-1) ** 2 / k0_sq)
        log_envelope = -0.5 * p * log_ratio_sq

        # Additive combination in z-space
        z = raw_z + log_envelope  # No head scale effect -- quick comparison preferred no head scaling
        # z = raw_z + self.head_scale.unsqueeze(0) + log_envelope  # Include head scale effect

        # Some debug prints
        if self.debug:
            with torch.no_grad():
                sat_mask = (raw_z.abs() > 0.95 * Z_CLAMP)  # (B, 2) -- per-head saturation
                sat_any = sat_mask.any(dim=1)  # (B,)   -- saturated on either head

                x_bins = torch.tensor([0.0, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0],
                                      device=x64.device, dtype=x64.dtype)
                bin_idx = torch.bucketize(x64.detach(), x_bins)

                print(f"  clamp saturation frac: {sat_any.float().mean():.3f}  "
                      f"K: {torch.sqrt(K).mean():.3f}  k0: {torch.sqrt(k0_sq).mean():.3f}  "
                      f"p: {p.tolist()}")
                for b in range(1, len(x_bins)):
                    m = bin_idx == b
                    n = int(m.sum())
                    if n > 0:
                        frac0 = sat_mask[m, 0].float().mean().item()
                        frac1 = sat_mask[m, 1].float().mean().item()
                        print(f"    x∈[{x_bins[b - 1]:.1e},{x_bins[b]:.1e})  n={n:6d}  "
                              f"sat(A0)={frac0:.3f}  sat(A1)={frac1:.3f}")

        return (z - self.y_mean) / self.y_std


def set_debug(model, flag: bool):
    """
    Set debug flag in a model object.
    """
    m = model.module if isinstance(model, nn.DataParallel) else model
    m = getattr(m, '_orig_mod', m)  # unwrap torch.compile if present
    m.debug = flag

# ==============================================================================
# Training utilities
# ==============================================================================
def apply_output_transform(y_over_f0: torch.Tensor, transform: str, epsilon: float = 1e-10):
    """
    Shared forward transform
    """
    if transform == "arcsinh":
        return torch.arcsinh(y_over_f0)
    elif transform == "log":
        return torch.sign(y_over_f0) * torch.log(torch.abs(y_over_f0) + epsilon)
    return y_over_f0


def reverse_output_transform(y_over_f0: torch.Tensor, transform: str, epsilon: float = 1e-10):
    """
    Shared reverse transform
    """
    if transform == "arcsinh":
        return torch.sinh(y_over_f0)
    elif transform == "log":
        return np.nan
    return y_over_f0


def compute_loss(
        model: nn.Module,
        inputs: torch.Tensor,
        A0_targets: torch.Tensor,
        A1_targets: torch.Tensor,
        weights: torch.Tensor,
        A0_err: torch.Tensor,
        A1_err: torch.Tensor,
        A0_intw: torch.Tensor,
        A1_intw: torch.Tensor,
        config: TrainingConfig,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Compute weighted MSE loss with physics constraints.

    The model outputs 2 harmonic heads (A0, A1). These are combined via
    A0 + A1*cos(phi) to reconstruct I/f0, then arcsinh-transformed
    and standardized to match the (pre-transformed) targets.

    Returns total loss and dictionary of individual loss components.
    """
    # Find indices of physical features
    names = RadiationDataset.FEATURE_NAMES
    IDX_K_PERP = names.index('ln(k_perp)')
    IDX_E = names.index('ln(E)')

    """
    Weighted MSE loss
    """
    # Get head outputs and compute MSE per each
    NN_heads = model(inputs)  # (B, 2)
    NN_A0 = NN_heads[:, 0]
    NN_A1 = NN_heads[:, 1]

    # Dead-zone (epsilon-insensitive) residual: zero loss contribution for
    # any point where |NN - target| is within that point's MC integration
    # error; only the excess beyond the noise floor is penalized.
    A0_excess = torch.clamp(torch.abs(NN_A0 - A0_targets) - A0_err, min=0.0)
    A1_excess = torch.clamp(torch.abs(NN_A1 - A1_targets) - A1_err, min=0.0)

    # Quadrature subtraction residual: variance decomposition into (modal error)^2 + (MC error)^2
    # Advantage of having no discontinuity at error level, but not as clean of a signal of what the error means
    # A0_excess = torch.clamp(torch.abs(NN_A0 - A0_targets)**2 - A0_err**2, min=0.0)
    # A1_excess = torch.clamp(torch.abs(NN_A1 - A1_targets)**2 - A1_err**2, min=0.0)

    A0_mse = (weights * A0_excess ** 2).mean()
    A1_mse = (weights * A1_excess ** 2).mean()


    """
    Integral importance weighted absolute-error loss
    """
    # A0_intw/A1_intw ≈ (integral quadrature weight) * d(physical)/d(z_norm)^2,
    # so (excess^2 * intw) approximates each point's contribution to the
    # squared error of the deployment-grid integral estimate.
    A0_int_abs = (A0_intw * A0_excess ** 2).mean()
    A1_int_abs = (A1_intw * A1_excess ** 2).mean()

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

        # Unnormalize the whole row back to physical space
        X_mean = model.X_mean.to(device)
        X_std = model.X_std.to(device)
        phys = uv_params * X_std + X_mean  # (n_uv_samples, n_features), physical units

        E_mean = model.X_mean[IDX_E].to(device)
        E_std = model.X_std[IDX_E].to(device)
        energy_params = np.pow(10, (uv_params[:, IDX_E] * E_std + E_mean).cpu().numpy())

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

        # Overwrite k_perp in physical space
        phys[:, IDX_K_PERP] = k_perp_uv

        # Recompute the k_perp-dependent derived features consistently
        IDX_X, IDX_E, IDX_Z0 = names.index('ln(x)'), names.index('ln(E)'), names.index('ln(z0)')
        x_p, E_p, z0_p = torch.exp(phys[:, IDX_X]), torch.exp(phys[:, IDX_E]), torch.exp(phys[:, IDX_Z0])
        mu_p = phys[:, names.index('mu')]
        k_perp_p = torch.exp(phys[:, IDX_K_PERP])
        omega_k = k_perp_p ** 2 / (2 * x_p * E_p)
        phys[:, names.index('omega_k_dz')] = omega_k * DELTA_Z / 2
        phys[:, names.index('omega_k_midz')] = omega_k * (z0_p + DELTA_Z / 2)
        phys[:, names.index('omega_width')] = mu_p ** 2 * DELTA_Z / (2 * x_p * E_p)

        # Re-normalize before feeding to the model
        uv_params = (phys - X_mean) / X_std
        uv_heads = model(uv_params) # shape (n_uv_samples, 3)

        # Use log weight to penalize nonzero result at larger k_perp values more
        log_esqr = torch.tensor(math.log(max_frac) + 2 * np.log(energy_params), device=device)
        log_weight = 2 * log_k - 2 * log_esqr

        # Penalize all three heads
        uv_decay = (log_weight.unsqueeze(1) * uv_heads ** 2).mean()
    else:
        uv_decay = torch.tensor(0.0, device=inputs.device)

    # Total loss
    total_loss = (
            config.lambda_A0 * A0_mse
            + config.lambda_A1 * A1_mse
            + config.lambda_A0_int * A0_int_abs
            + config.lambda_A1_int * A1_int_abs
            + config.lambda_uv * uv_decay
    )

    components = {
        'A0_mse': A0_mse.item(),
        'A1_mse': A1_mse.item(),
        'A0_int_abs': A0_int_abs.item(),
        'A1_int_abs': A1_int_abs.item(),
        'uv_decay': uv_decay.item(),
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
    for i, (inputs, A0_targets, A1_targets, weights, A0_err, A1_err, A0_intw, A1_intw) in enumerate(dataloader):
        # if i % 10000 == 0 and i != 0:
        #     print(f"  Batch {i}/{len(dataloader)}  [avg {(time.time() - t0)/i:.3f}s/batch]")

        # Send tensors to device
        inputs = inputs.to(config.device)
        A0_targets = A0_targets.to(config.device)
        A1_targets = A1_targets.to(config.device)
        weights = weights.to(config.device)
        A0_err = A0_err.to(config.device)
        A1_err = A1_err.to(config.device)

        # Compute loss and step optimizer
        optimizer.zero_grad()
        loss, components = compute_loss(model, inputs, A0_targets, A1_targets, weights, A0_err, A1_err, A0_intw, A1_intw, config)
        if not torch.isfinite(loss):
            print("Non-finite loss detected:", components)
            raise FloatingPointError("Non-finite loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # clip gradients to avoid feedback from exploding gradients
        optimizer.step()

        # Add to running sum of loss and MSE
        total_loss += components['total']
        total_mse += components['A0_mse'] + components['A1_mse']
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
) -> Dict[str, float]:
    """Validate the model."""
    model.eval()
    total_loss = 0.0
    total_mse = 0.0
    n_batches = 0

    with torch.no_grad():
        for inputs, A0_targets, A1_targets, weights, A0_err, A1_err, A0_intw, A1_intw in dataloader:
            # Send tensors to device
            inputs = inputs.to(config.device)
            A0_targets = A0_targets.to(config.device)
            A1_targets = A1_targets.to(config.device)
            weights = weights.to(config.device)
            A0_err = A0_err.to(config.device)
            A1_err = A1_err.to(config.device)

            # Compute loss
            _, components = compute_loss(model, inputs, A0_targets, A1_targets, weights, A0_err, A1_err, A0_intw, A1_intw, config)

            # Add to running sum of loss and MSE
            total_loss += components['total']
            total_mse += components['A0_mse'] + components['A1_mse']
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
    # print(f"  [checkpoint] Saved training state at epoch {epoch + 1} → {path}")


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

    # Create normalization metadata container
    norm = Normalization.from_dataset(dataset, device=config.device)

    # Split on *physical* indices, so each subset gets a
    # contiguous, cache-friendly slice of the underlying arrays.
    n_phys_train = int(dataset.n_valid * config.train_fraction)

    rng_split = np.random.default_rng(42)
    perm = rng_split.permutation(dataset.n_valid).astype(np.intp)

    train_idx = perm[:n_phys_train]
    val_idx = perm[n_phys_train:]

    # Split dataset into training and validation subsets using Pytorch's built in Subset class
    train_dataset = Subset(dataset, train_idx)
    val_dataset = Subset(dataset, val_idx)

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
        transform=config.transform,
        n_harmonics=config.n_harmonics,
        n_harmonics_dz=config.n_harmonics_dz,
    ).to(config.device)

    # Set normalization values of model
    model.set_normalization(
        X_mean=dataset.X_mean,
        X_std=dataset.X_std,
        y_mean=[dataset.A0_mean, dataset.A1_mean],  # 0.0 placeholder for unused A2 head
        y_std=[dataset.A0_std, dataset.A1_std],
        f0=dataset.f0,
        epsilon=dataset.epsilon,
    )

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
            train_metrics = train_epoch(model, train_loader, optimizer, config)

            # Validate
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
                            'transform': config.transform,
                            'n_harmonics': config.n_harmonics,
                            'n_harmonics_dz': config.n_harmonics_dz,
                        },
                        'feature_names': dataset.FEATURE_NAMES,
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

    # Shutdown our dataset object
    dataset.shutdown()

    return model, dataset.get_normalization_params()


# ==============================================================================
# Inference utilities
# ==============================================================================
class RadiationEmulatorInference:
    def __init__(self, model_file: str = "data/radiation_emulator.pt", device="cpu", compile=False, quiet=False):
        checkpoint = torch.load(model_file, map_location=device, weights_only=True)
        model_config = checkpoint['config']

        self.model = RadiationEmulator(
            input_dim=model_config['input_dim'],
            hidden_dim=model_config['hidden_dim'],
            n_layers=model_config['n_layers'],
            activation=model_config['activation'],
            dropout_p=model_config.get('dropout_p', 0.0),
            transform=model_config.get('transform', 'arcsinh'),
            n_harmonics=model_config.get('n_harmonics', 0),
            n_harmonics_dz=model_config.get('n_harmonics_dz', 0),
        ).to(device)
        self.device = device

        saved_names = checkpoint.get('feature_names')
        # if saved_names is not None and saved_names != RadiationDataset.FEATURE_NAMES:
        #     raise ValueError(f"Feature mismatch: model trained with {saved_names}")

        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs")
            self.model = nn.DataParallel(self.model)

        state_dict = checkpoint['model_state_dict']
        try:
            self.model.load_state_dict(state_dict)
        except RuntimeError:
            state_dict = {k.removeprefix('_orig_mod.').removeprefix('module.'): v
                          for k, v in state_dict.items()}
            self.model.load_state_dict(state_dict)   # buffers restored here too

        self.model.eval()

        # Optionally compile the model for faster repeated inference
        # (requires PyTorch >= 2.0; falls back on older versions)
        if hasattr(torch, 'compile') and compile:
            self.model = torch.compile(self.model)

        if not quiet:
            print(f"Loaded model from {model_file}")
            print(f"  Validation loss: {checkpoint['val_loss']:.4e}")
            print(f"  Trained for {checkpoint['epoch'] + 1} epochs")
            set_debug(self.model, True)

    def predict(
            self,
            x: np.ndarray,
            k_perp: np.ndarray,
            phi: np.ndarray,
            E: np.ndarray,
            z0: np.ndarray,
            u_perp: np.ndarray,
            mu: np.ndarray,
    ) -> np.ndarray:
        """
        Physical-facing entry point -- returns the combined scalar intensity
        """
        A0, A1 = self.predict_harmonics(x, k_perp, E, z0, u_perp, mu)
        predictions = A0 + A1 * np.cos(phi)
        return predictions

    def predict_harmonics(
            self,
            x: np.ndarray,
            k_perp: np.ndarray,
            E: np.ndarray,
            z0: np.ndarray,
            u_perp: np.ndarray,
            mu: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns physical (A0, A1, A2) harmonic amplitudes for the given
        (x, k_perp, ...) points — no phi dependence, so callers can evaluate
        this on a much coarser grid than the full 3D (kx, ky, kz) grid.
        """
        inputs = np.column_stack(
            list(compute_input_features(
                x=x, k_perp=k_perp, E=E, z0=z0, u_perp=u_perp, mu=mu).values()
                                 )).astype(np.float32, order='C', copy=False
        )
        A_heads = self.predict_harmonics_raw(inputs)
        return A_heads[:, 0], A_heads[:, 1]

    def predict_harmonics_raw(self, inputs: np.ndarray) -> np.ndarray:
        inputs_tensor = torch.from_numpy(np.asarray(inputs, dtype=np.float32)).to(self.device)

        # Normalize using the model's own buffers -- guaranteed in sync with training
        inputs_norm = (inputs_tensor - self.model.X_mean) / self.model.X_std

        with torch.no_grad():
            z = self.model(inputs_norm)                        # (N, 3), normalized+transformed

        phys_over_f0 = torch.sinh(z * self.model.y_std + self.model.y_mean)
        A_heads = (self.model.f0 * phys_over_f0).cpu().numpy()
        return A_heads

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
            mu=inputs['mu'],
        )

    def compute_dNdxd2k_grid(self,
                             E: float,
                             z0: float,
                             u_perp: float,
                             mu: float,
                             x_values: np.ndarray,
                             k_perp_values: np.ndarray,
                             phi_values: np.ndarray,
                             ) -> (np.ndarray):
        """
        Computes a complete grid of (1/CR) dN/(dx d^2k_perp) shaped as (k_perp, phi, x).

        Note that this is actually differential in (dx d^2k_perp), not (dx d|k_perp|, dphi).
        Apply the Jacobian later when you need it!

        The network is evaluated only on the 2D (k_perp, x) grid; phi dependence
        is reconstructed analytically via A0 + A1*cos(phi) + A2*cos(2*phi).

        Points outside the kinematic region mu/E <= x <= 1 - mu/E and
        mu <= k_perp <= sqrt(E^2 * min(x^2, (1-x)^2) - mu^2) are set to zero.
        """
        # create a 2D meshgrid in the only two variables the network needs
        kperp_grid2d, x_grid2d = np.meshgrid(k_perp_values, x_values, indexing='ij')  # (n_kperp, n_x)

        # Build input grid once
        n_pts = x_grid2d.size
        grid_inputs = np.column_stack([
            x_grid2d.ravel(), kperp_grid2d.ravel(),
            np.full(n_pts, E), np.full(n_pts, z0),
            np.full(n_pts, u_perp), np.full(n_pts, T), np.full(n_pts, g),
        ]).astype(np.float32)

        # Single batched network call over the (k_perp, x) grid -- no phi dependence yet
        A_flat = self.predict_harmonics_raw(grid_inputs)  # (n_pts, 3)
        A0_2d = A_flat[:, 0].reshape(x_grid2d.shape)
        A1_2d = A_flat[:, 1].reshape(x_grid2d.shape)
        A2_2d = A_flat[:, 2].reshape(x_grid2d.shape)

        # Reconstruct full angular dependence via broadcasting -- shape (n_kperp, n_phi, n_x)
        cos_phi = np.cos(phi_values)[None, :, None]
        cos_2phi = np.cos(2 * phi_values)[None, :, None]
        I_nn = A0_2d[:, None, :] + A1_2d[:, None, :] * cos_phi + A2_2d[:, None, :] * cos_2phi

        # Perform kinematic cuts on the 2D (k_perp, x) grid -- the cut is
        # phi-independent, so evaluating it here (instead of on the full 3D
        # grid) avoids redundant sqrt/compare work across the phi axis.
        x_min = mu / E
        x_max = 1.0 - mu / E
        kperp_min = mu
        kperp_max_sq = (E ** 2) * np.minimum(x_grid2d ** 2, (1.0 - x_grid2d) ** 2) - mu ** 2
        kperp_max = np.sqrt(np.clip(kperp_max_sq, 0.0, None))

        valid_2d = (
                (x_grid2d > x_min) & (x_grid2d < x_max) &
                (kperp_grid2d > kperp_min) & (kperp_grid2d < kperp_max)
        )  # shape (n_kperp, n_x), broadcasts over the phi axis below

        I_nn *= valid_2d[:, None, :]

        # Compute dN/dxd^2k_perp by dividing out energy of each grid point
        N_nn = I_nn / (E * x_grid2d[:, None, :])  # Still needs casimir factor

        # Returned grid is (k_perp, phi, x), differential in x and d^2k_perp
        return N_nn


# ==============================================================================
# Optimization tools
# ==============================================================================
def find_learning_rate(
        model: nn.Module,
        dataloader: DataLoader,
        optimizer: torch.optim.Optimizer,
        config: TrainingConfig,
        normalization: Normalization,
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
            inputs, A0_targets, A1_targets, weights = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            inputs, A0_targets, A1_targets, weights = next(data_iter)

        inputs  = inputs.to(config.device)
        A0_targets = A0_targets.to(config.device)
        A1_targets = A1_targets.to(config.device)
        weights = weights.to(config.device)

        optimizer.zero_grad()
        loss, components = compute_loss(model, inputs, A0_targets, A1_targets, weights, config, normalization)
        if not torch.isfinite(loss):
            print("Non-finite loss detected:", components)
            raise FloatingPointError("Non-finite loss")
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
# Main entry point
# ==============================================================================
if __name__ == "__main__":
    default_config = TrainingConfig()  # Load default values from class description
    parser = argparse.ArgumentParser(description="Train or run inference with radiation emulator")
    parser.add_argument("--data-file", type=str, default=default_config.data_file, help="Training data file")
    parser.add_argument("--model-file", type=str, default=default_config.model_file, help="Model output file")
    parser.add_argument("--transform", type=str, default=default_config.transform, help="Type of transform on data")
    parser.add_argument("--epochs", type=int, default=default_config.n_epochs, help="Number of training epochs")
    parser.add_argument("--hidden-dim", type=int, default=default_config.hidden_dim, help="Hidden layer dimension")
    parser.add_argument("--n-layers", type=int, default=default_config.n_layers, help="Number of hidden layers")
    parser.add_argument("--n-harmonics", type=int, default=default_config.n_harmonics,
                        help="Number of cos/sin harmonics of the bare oscillation phase per head")
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
        n_harmonics=args.n_harmonics,
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
