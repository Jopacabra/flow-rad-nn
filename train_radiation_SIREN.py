
"""
train_radiation_nn.py

Trains a SIREN (Sinusoidal Representation Network) emulator for the
medium-induced radiation intensity distribution using precomputed training data.

The SIREN architecture uses periodic (sin) activations throughout, which
directly addresses spectral bias — the tendency of standard MLPs to learn
low-frequency components first and struggle with the high-frequency LPM
oscillations present in the radiation spectra.

Key architectural differences from the standard MLP version:
- All activations are sin(w0 * x), where w0 is a tunable frequency parameter
- Weight initialisation follows the scheme of Sitzmann et al. (2020), which
  preserves a stationary distribution of activations through depth — this is
  handled internally by siren-pytorch
- Dropout is omitted from the SIREN body, as it disrupts the initialisation
  guarantees; regularisation is achieved via weight_decay in AdamW instead
- Skip connections are omitted; SIREN's periodic activations provide
  multi-scale representation intrinsically

Tuning w0 / w0_initial:
  These control the frequency of the sinusoidal activations. A rough guide
  is to set w0_initial to be on the order of the dominant oscillation frequency
  in your normalised input space. For LPM oscillations, the relevant scale is
  k^2 / (2xE); after input normalisation this maps to O(10–100). Start with
  the default of 30 and sweep {30, 60, 120} if the oscillations are not
  resolved. w0 (hidden layers) can usually stay at 30.

Features:
- Loads training data from HDF5 file
- Normalizes inputs and log-transforms outputs
- Uses importance sampling weights for unbiased training
- Enforces soft physics constraints (positivity, k_y symmetry)
- Saves trained model for deployment
- Demonstrates batch inference

Dependencies:
    pip install siren-pytorch

Usage:
    python train_radiation_nn.py                    # Train the model
    python train_radiation_nn.py --inference-only   # Demo inference with saved model
"""

import argparse
import copy
import json
import time

import h5py
import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass
from siren_pytorch import SirenNet
from torch.utils.data import Dataset, DataLoader, random_split
from typing import Optional, Tuple, Dict


# ==============================================================================
# Configuration
# ==============================================================================
@dataclass
class TrainingConfig:
    """
    Configuration for neural network training.
    """

    # Data
    data_file: str = "radiation_training_data.h5"
    train_fraction: float = 0.8
    log_transform: bool = True

    # Architecture
    hidden_dim: int = 256
    n_layers: int = 5
    # w0_initial: frequency for the first SIREN layer.
    #   Start at 30. Increase to 60 or 120 if high-frequency LPM oscillations
    #   are not resolved. Too high risks training instability.
    siren_w0_initial: float = 30.0
    # w0: frequency for all subsequent hidden layers. Usually kept at 30.
    siren_w0: float = 30.0

    # Training
    # Note: a lower learning rate than a standard MLP is recommended for SIRENs.
    # Start with 1e-4 and use the LR finder to confirm.
    batch_size: int = 256
    learning_rate: float = 1e-4
    # weight_decay: float = 1e-4  # counterproductive for SIRENs
    n_epochs: int = 200
    patience: int = 20  # Early stopping patience

    # Physics constraints
    lambda_positivity: float = 0.0  # Weight for positivity penalty -- 0 so that we don't enforce positivity
    lambda_ky_symmetry: float = 0.0  # Weight for k_y symmetry penalty -- data enforces it via mirroring

    # Output
    model_file: str = "radiation_emulator.pt"
    normalization_file: str = "radiation_normalization.json"

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Utilities
    run_lr_finder: bool = False
    run_w0_finder: bool = False


# ==============================================================================
# Dataset
# ==============================================================================
class RadiationDataset(Dataset):
    """Dataset for radiation intensity training data."""

    # Input feature names in order
    FEATURE_NAMES = ['x', 'kx', 'ky', 'E', 'z0', 'zf', 'u_perp', 'T', 'g']

    def __init__(self, data_file: str, log_transform_output: bool = True):
        """
        Load training data from HDF5 file.

        Parameters
        ----------
        data_file : str
            Path to HDF5 file with training data
        log_transform_output : bool
            Whether to apply log(|I| + epsilon) transformation to outputs
        """
        self.log_transform = log_transform_output
        self.epsilon = 1e-10  # Small constant for log stability

        # Load data from HDF5
        with h5py.File(data_file, 'r') as f:
            # Load all features
            features = []
            for name in self.FEATURE_NAMES:
                features.append(f[name][:])

            self.X = np.column_stack(features).astype(np.float32)
            self.y = f['I'][:].astype(np.float32)
            self.y_err = f['I_err'][:].astype(np.float32)
            self.weights = f['weight'][:].astype(np.float32)

            # Store metadata
            self.n_samples = f.attrs.get('n_samples', len(self.y))

        # Filter out any NaN or Inf values
        valid_mask = (
                np.isfinite(self.y) &
                np.isfinite(self.y_err) &
                np.all(np.isfinite(self.X), axis=1)
        )
        self.X = self.X[valid_mask]
        self.y = self.y[valid_mask]
        self.y_err = self.y_err[valid_mask]
        self.weights = self.weights[valid_mask]

        # Compute input normalization (mean/std for each feature)
        self.X_mean = self.X.mean(axis=0)
        self.X_std = self.X.std(axis=0) + 1e-8  # Avoid division by zero

        # Store sign of y before log transform (needed for reconstruction)
        self.y_sign = np.sign(self.y)

        # Compute output normalization
        if self.log_transform:
            # Log transform: y_transformed = sign(y) * log(|y| + epsilon)
            self.y_transformed = self.y_sign * np.log(np.abs(self.y) + self.epsilon)
        else:
            self.y_transformed = self.y

        self.y_mean = self.y_transformed.mean()
        self.y_std = self.y_transformed.std() + 1e-8

        # Normalize
        self.X_normalized = (self.X - self.X_mean) / self.X_std
        self.y_normalized = (self.y_transformed - self.y_mean) / self.y_std

        # Normalize weights to have mean 1
        self.weights = self.weights / self.weights.mean()

        print(f"Loaded {len(self.y)} samples from {data_file}")
        print(f"Input shape: {self.X.shape}")
        print(f"Output range: [{self.y.min():.4e}, {self.y.max():.4e}]")

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.X_normalized[idx]),
            torch.tensor(self.y_normalized[idx]),
            torch.tensor(self.weights[idx]),
            torch.tensor(self.y_sign[idx]),  # For symmetry loss
        )

    def get_normalization_params(self) -> Dict:
        """Return normalization parameters for saving."""
        return {
            'X_mean': self.X_mean.tolist(),
            'X_std': self.X_std.tolist(),
            'y_mean': float(self.y_mean),
            'y_std': float(self.y_std),
            'log_transform': self.log_transform,
            'epsilon': self.epsilon,
            'feature_names': self.FEATURE_NAMES,
        }


# ==============================================================================
# Neural Network Model
# ==============================================================================
class RadiationEmulator(nn.Module):
    """
    SIREN emulator for medium-induced radiation intensity.

    Wraps siren_pytorch.SirenNet for use in this training pipeline.

    All activations are sin(w0 * (Wx + b)), with weights initialised
    following Sitzmann et al. (2020) to preserve a stationary distribution
    of pre-activations through depth. This is handled internally by
    SirenNet and must NOT be overridden.

    Dropout is intentionally omitted from the SIREN body: the initialisation
    scheme relies on every neuron being active, and dropout breaks this
    guarantee. Regularisation is handled by weight_decay in the optimizer.
    """

    def __init__(
            self,
            input_dim: int = 9,
            hidden_dim: int = 256,
            n_layers: int = 5,
            w0: float = 30.0,
            w0_initial: float = 30.0,
    ):
        """
        Parameters
        ----------
        input_dim : int
            Number of input features (9 for this problem).
        hidden_dim : int
            Width of each hidden layer.
        n_layers : int
            Total number of layers (including the final linear output layer).
        w0 : float
            Frequency multiplier for all hidden-layer sinusoidal activations.
        w0_initial : float
            Frequency multiplier for the first layer. Increase this (e.g. to
            60 or 120) if the network fails to resolve the LPM oscillations.
        """
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.w0 = w0
        self.w0_initial = w0_initial

        self.net = SirenNet(
            dim_in=input_dim,
            dim_hidden=hidden_dim,
            dim_out=1,
            num_layers=n_layers,
            w0=w0,
            w0_initial=w0_initial,
            use_bias=True,
            final_activation=nn.Identity(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, input_dim) with normalized features.

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch_size, 1) with normalized log-intensity.
        """
        return self.net(x).squeeze(-1)


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
    Compute weighted MSE loss with optional physics constraints.

    Returns total loss and dictionary of individual loss components.
    """
    predictions = model(inputs).squeeze(-1)

    # Weighted MSE loss
    mse = (weights * (predictions - targets) ** 2).mean()

    # Positivity penalty (in normalized space, this is approximate)
    # We penalize strongly negative predictions
    positivity = torch.relu(-predictions - 2.0).mean()  # Threshold at -2 std

    # k_y symmetry is already built into the training data (mirrored samples).
    # The constraint below can optionally add an extra soft penalty on top.
    # Create inputs with flipped k_y (index 2 in feature list)
    inputs_ky_flipped = inputs.clone()
    inputs_ky_flipped[:, 2] = -inputs_ky_flipped[:, 2]  # ky -> -ky
    predictions_flipped = model(inputs_ky_flipped).squeeze(-1)
    ky_symmetry = ((predictions - predictions_flipped) ** 2).mean()

    # Total loss
    total_loss = (
            mse
            + config.lambda_positivity * positivity
            + config.lambda_ky_symmetry * ky_symmetry
    )

    # Return loss components for logging
    components = {
        'mse': mse.item(),
        'positivity': positivity.item(),
        'ky_symmetry': ky_symmetry.item(),
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

    for inputs, targets, weights, _ in dataloader:
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
        for inputs, targets, weights, _ in dataloader:
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


def _make_checkpoint_config(config: TrainingConfig, dataset: RadiationDataset) -> Dict:
    """Build the config sub-dict that is stored inside each model checkpoint."""
    return {
        'input_dim': len(dataset.FEATURE_NAMES),
        'hidden_dim': config.hidden_dim,
        'n_layers': config.n_layers,
        'siren_w0': config.siren_w0,
        'siren_w0_initial': config.siren_w0_initial,
    }


# ==============================================================================
# Main training function
# ==============================================================================
def train_model(config: TrainingConfig):
    """Train the radiation emulator model."""

    print("=" * 70)
    print("RADIATION EMULATOR TRAINING  (SIREN architecture)")
    print("=" * 70)
    print(f"Device:          {config.device}")
    print(f"Data file:       {config.data_file}")
    print(f"w0_initial:      {config.siren_w0_initial}  (first-layer frequency)")
    print(f"w0:              {config.siren_w0}  (hidden-layer frequency)")
    print()

    # Load dataset
    dataset = RadiationDataset(config.data_file, log_transform_output=config.log_transform)

    # Split into train/validation
    n_train = int(len(dataset) * config.train_fraction)
    n_val = len(dataset) - n_train
    train_dataset, val_dataset = random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )

    print(f"Training samples:   {n_train}")
    print(f"Validation samples: {n_val}")

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=0
    )

    # Create model
    model = RadiationEmulator(
        input_dim=len(dataset.FEATURE_NAMES),
        hidden_dim=config.hidden_dim,
        n_layers=config.n_layers,
        w0=config.siren_w0,
        w0_initial=config.siren_w0_initial,
    ).to(config.device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    # Optimizer.
    # AdamW with weight_decay is the primary regulariser for SIREN (no dropout).
    # A lower learning rate than a standard MLP is typically needed.
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
    )

    # LR finder -- sweep LR exponentially to find the optimal starting value.
    # Looks for the point of steepest loss descent; suggests 1/3 of that value.
    if config.run_lr_finder:
        print("\nRunning LR range test...")
        print("-" * 70)
        lrs, losses = find_learning_rate(model, train_loader, optimizer, config)
        suggested_lr = plot_lr_finder(lrs, losses)
        print(f"\nRe-run with --learning-rate {suggested_lr / 3:.2e} (1/3 of suggested)")
        return model, dataset.get_normalization_params()

    # Sweep omega0, training a new model for each value (it's an initialization parameter!), finding optimal point
    if config.run_w0_finder:
        w0_vals, losses = find_w0(dataset, config)
        best_w0 = plot_w0_finder(w0_vals, losses)
        return model, dataset.get_normalization_params()

    # # Scheduler: reduce LR when validation loss plateaus.
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #     optimizer, mode='min', factor=0.5, patience=10,
    #     min_lr=1e-9,
    # )

    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0
    checkpoint_config = _make_checkpoint_config(config, dataset)

    print("\nStarting training...")
    print("-" * 70)

    for epoch in range(config.n_epochs):
        # Train
        train_metrics = train_epoch(model, train_loader, optimizer, config)

        # Validate
        val_metrics = validate(model, val_loader, config)

        # Update scheduler
        # scheduler.step(val_metrics['loss'])  # No scheduling of LR, possibly counterproductive for SIRENs.

        # Print progress
        if (epoch + 1) % 10 == 0 or epoch == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(
                f"Epoch {epoch + 1:3d}/{config.n_epochs} | "
                f"Train Loss: {train_metrics['loss']:.4e} | "
                f"Val Loss: {val_metrics['loss']:.4e} | "
                f"Val MSE: {val_metrics['mse']:.4e} | "
                f"LR: {current_lr:.2e}"
            )

        # Early stopping / checkpoint
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            patience_counter = 0

            torch.save({
                'model_state_dict': model.state_dict(),
                'config': checkpoint_config,
                'epoch': epoch,
                'val_loss': best_val_loss,
            }, config.model_file)
        else:
            patience_counter += 1
            if patience_counter >= config.patience:
                print(f"\nEarly stopping at epoch {epoch + 1}")
                break

    # Save normalization parameters
    norm_params = dataset.get_normalization_params()
    with open(config.normalization_file, 'w') as f:
        json.dump(norm_params, f, indent=2)

    print("-" * 70)
    print(f"Training complete!")
    print(f"Best validation loss: {best_val_loss:.4e}")
    print(f"Model saved to: {config.model_file}")
    print(f"Normalization params saved to: {config.normalization_file}")

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
            model_file: str = "radiation_emulator.pt",
            normalization_file: str = "radiation_normalization.json",
            device: str = "cpu",
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
        self.y_mean = self.norm_params['y_mean']
        self.y_std = self.norm_params['y_std']
        self.log_transform = self.norm_params['log_transform']
        self.epsilon = self.norm_params['epsilon']

        # Load model
        checkpoint = torch.load(model_file, map_location=device)
        model_config = checkpoint['config']

        self.model = RadiationEmulator(
            input_dim=model_config['input_dim'],
            hidden_dim=model_config['hidden_dim'],
            n_layers=model_config['n_layers'],
            w0=model_config.get('siren_w0', 30.0),
            w0_initial=model_config.get('siren_w0_initial', 30.0),
        ).to(device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        print(f"Loaded model from {model_file}")
        print(f"  Validation loss: {checkpoint['val_loss']:.4e}")
        print(f"  Trained for {checkpoint['epoch'] + 1} epochs")
        print(f"  w0_initial: {model_config.get('siren_w0_initial', 30.0)}, "
              f"w0: {model_config.get('siren_w0', 30.0)}")

    def predict(
            self,
            x: np.ndarray,
            kx: np.ndarray,
            ky: np.ndarray,
            E: np.ndarray,
            z0: np.ndarray,
            zf: np.ndarray,
            u_perp: np.ndarray,
            T: np.ndarray,
            g: np.ndarray,
    ) -> np.ndarray:
        """
        Predict radiation intensity for a batch of input points.

        All inputs should be 1D numpy arrays of the same length.

        Parameters
        ----------
        x : array-like
            Momentum fraction
        kx : array-like
            Transverse momentum k_x (GeV)
        ky : array-like
            Transverse momentum k_y (GeV)
        E : array-like
            Parton energy (GeV)
        z0 : array-like
            Initial longitudinal position (fm)
        zf : array-like
            Final longitudinal position (fm)
        u_perp : array-like
            Transverse flow velocity magnitude
        T : array-like
            Temperature (GeV)
        g : array-like
            Coupling constant

        Returns
        -------
        np.ndarray
            Predicted radiation intensity I (NOT including CF factor)
        """
        # Stack inputs
        inputs = np.column_stack([x, kx, ky, E, z0, zf, u_perp, T, g]).astype(np.float32)

        # Convert to tensor and normalize
        inputs_tensor = torch.from_numpy(inputs).to(self.device)
        inputs_norm = (inputs_tensor - self.X_mean.to(self.device)) / self.X_std.to(self.device)

        # Predict
        with torch.no_grad():
            predictions_norm = self.model(inputs_norm).squeeze(-1)

        # Denormalize
        predictions_transformed = predictions_norm.cpu().numpy() * self.y_std + self.y_mean

        # Inverse log transform: y_transformed = sign(y) * log(|y| + epsilon)
        if self.log_transform:
            predictions = np.exp(np.abs(predictions_transformed)) - self.epsilon
            predictions = np.sign(predictions_transformed) * predictions
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


# ==============================================================================
# Optimization tools
# ==============================================================================
def find_learning_rate(
        model: nn.Module,
        dataloader: DataLoader,
        optimizer: torch.optim.Optimizer,
        config: TrainingConfig,
        start_lr: float = 1e-10,
        end_lr: float = 1e-2,
        n_steps: int = 50,
        smoothing: float = 0.3,
) -> Tuple[list, list]:
    """
    Learning rate range test (Smith 2015).

    Sweeps LR exponentially from start_lr to end_lr over n_steps batches,
    recording the smoothed loss at each step. Default range is narrower than
    for a standard MLP to reflect SIREN's lower optimal learning rate.

    Returns
    -------
    lrs : list of float
        Learning rates tested
    losses : list of float
        Smoothed loss at each learning rate
    """
    # Save original model and optimizer state so we can restore after the test
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
            inputs, targets, weights, _ = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            inputs, targets, weights, _ = next(data_iter)

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
    ax.set_title('Learning Rate Range Test (SIREN)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('lr_finder.png', dpi=150)
    print(f"  Plot saved to lr_finder.png")
    plt.show()

    return suggested_lr

def find_w0(
        dataset: RadiationDataset,
        config: TrainingConfig,
        w0_values: list = None,
        n_epochs_per_w0: int = 20,
        smoothing: float = 0.3,
) -> Tuple[list, list]:
    """
    w0 range test for SIREN: trains a fresh model for each candidate w0_initial
    and records the final smoothed validation loss.

    Unlike the LR range test, w0 is a model architecture parameter, so each
    candidate requires a new model instantiation and a short training run.
    w0 (hidden layers) is kept fixed at config.siren_w0; only w0_initial is swept.

    Parameters
    ----------
    dataset : RadiationDataset
        The full dataset (will be split internally).
    config : TrainingConfig
        Base training config. learning_rate, batch_size, weight_decay are reused.
    w0_values : list of float, optional
        Candidate w0_initial values to test. Defaults to a log-spaced sweep
        from 1 to 300.
    n_epochs_per_w0 : int
        Number of epochs to train each candidate. 20 is usually enough to
        distinguish good from bad initialisations.
    smoothing : float
        Exponential smoothing factor applied to the per-epoch val loss curve.

    Returns
    -------
    w0_values : list of float
        The w0_initial values tested.
    losses : list of float
        Smoothed final validation loss for each w0_initial.
    """
    if w0_values is None:
        w0_values = [1.0, 1/3.0, 1/10.0, 1/30.0, 1/60.0, 1/100.0, 1/200.0, 1/300.0][::-1]

    # Build dataloaders from the provided dataset
    n_train = int(len(dataset) * config.train_fraction)
    n_val = len(dataset) - n_train
    train_dataset, val_dataset = random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=0)

    final_losses = []

    print("\nRunning w0 range test...")
    print("-" * 70)
    print(f"{'w0_initial':>12}  {'Final Val Loss':>16}")
    print("-" * 70)

    for w0 in w0_values:
        # Fresh model for each candidate
        model = RadiationEmulator(
            input_dim=len(dataset.FEATURE_NAMES),
            hidden_dim=config.hidden_dim,
            n_layers=config.n_layers,
            w0=config.siren_w0,
            w0_initial=w0,
        ).to(config.device)

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.learning_rate,
        )

        smoothed_loss = None

        for epoch in range(n_epochs_per_w0):
            # Train one epoch
            model.train()
            for inputs, targets, weights_b, _ in train_loader:
                inputs = inputs.to(config.device)
                targets = targets.to(config.device)
                weights_b = weights_b.to(config.device)
                optimizer.zero_grad()
                loss, _ = compute_loss(model, inputs, targets, weights_b, config)
                loss.backward()
                optimizer.step()

            # Validate
            val_metrics = validate(model, val_loader, config)
            raw_loss = val_metrics['loss']

            if smoothed_loss is None:
                smoothed_loss = raw_loss
            else:
                smoothed_loss = smoothing * raw_loss + (1.0 - smoothing) * smoothed_loss

        final_losses.append(smoothed_loss)
        print(f"{w0:>12.1f}  {smoothed_loss:>16.4e}")

    print("-" * 70)
    best_idx = int(np.argmin(final_losses))
    print(f"\nBest w0_initial: {w0_values[best_idx]:.1f}  (loss={final_losses[best_idx]:.4e})")
    print(f"Re-run training with --w0-initial {w0_values[best_idx]:.1f}")

    return w0_values, final_losses


def plot_w0_finder(w0_values: list, losses: list):
    """Plot the w0 range test results."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(w0_values, losses, marker='o', linewidth=2)
    best_idx = int(np.argmin(losses))
    ax.axvline(w0_values[best_idx], color='red', linestyle='--',
               label=f'Best w0_initial: {w0_values[best_idx]:.1f}')
    ax.set_xscale('log')
    ax.set_xlabel('w0_initial (log scale)')
    ax.set_ylabel('Smoothed Validation Loss')
    ax.set_title('w0 Range Test (SIREN)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('w0_finder.png', dpi=150)
    print(f"  Plot saved to w0_finder.png")
    plt.show()

    return w0_values[best_idx]


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
    default_config = TrainingConfig()
    parser = argparse.ArgumentParser(description="Train or run inference with SIREN radiation emulator")
    parser.add_argument("--inference-only", action="store_true", help="Skip training, demo inference only")
    parser.add_argument("--data-file", type=str, default=default_config.data_file, help="Training data file")
    parser.add_argument("--model-file", type=str, default=default_config.model_file, help="Model output file")
    parser.add_argument("--log", type=bool, default=default_config.log_transform, help="Whether to log transform data")
    parser.add_argument("--epochs", type=int, default=default_config.n_epochs, help="Number of training epochs")
    parser.add_argument("--hidden-dim", type=int, default=default_config.hidden_dim, help="Hidden layer dimension")
    parser.add_argument("--n-layers", type=int, default=default_config.n_layers, help="Number of hidden layers")
    parser.add_argument("--learning-rate", type=float, default=default_config.learning_rate, help="Initial learning rate")
    parser.add_argument("--w0-initial", type=float, default=default_config.siren_w0_initial,
                        help="First-layer SIREN frequency (try 30, 60, 120)")
    parser.add_argument("--w0", type=float, default=default_config.siren_w0,
                        help="Hidden-layer SIREN frequency (usually keep at 30)")
    parser.add_argument("--find-lr", action="store_true", help="Run LR range test and exit")
    parser.add_argument("--find-w0", action="store_true", help="Run omega_0 range test and exit")
    args = parser.parse_args()

    config = TrainingConfig(
        data_file=args.data_file,
        model_file=args.model_file,
        n_epochs=args.epochs,
        hidden_dim=args.hidden_dim,
        n_layers=args.n_layers,
        learning_rate=args.learning_rate,
        siren_w0_initial=args.w0_initial,
        siren_w0=args.w0,
        run_lr_finder=args.find_lr,
        run_w0_finder=args.find_w0,
        log_transform=args.log,
    )

    if args.inference_only:
        demo_inference(config)
    else:
        train_model(config)
        print("\n")
        demo_inference(config)