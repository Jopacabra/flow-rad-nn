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
import os


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
    transform: str = "arcsinh"

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
    model_file: str = "radiation_emulator.pt"
    normalization_file: str = "radiation_normalization.json"

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers: int = 4  # DataLoader worker processes

    # Utilities
    run_lr_finder: bool = False


# ==============================================================================
# Dataset
# ==============================================================================
class RadiationDataset(Dataset):
    """Dataset for radiation intensity training data."""

    # Input feature names in order
    FEATURE_NAMES = ['x', 'kx', 'ky', 'E', 'z0', 'zf', 'u_perp', 'T', 'g']

    def __init__(self, data_file: str, transform_output: str = "arcsinh"):
        """
        Load training data from HDF5 file.

        Parameters
        ----------
        data_file : str
            Path to HDF5 file with training data
        transform_output : str
            Whether to apply arcsinh, log, etc. transformation to outputs
        """
        self.transform = transform_output
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

        # Store information potentially needed for transforms
        self.y_sign = np.sign(self.y)
        self.f0 = np.std(self.y)  # Scale factor for dataset -- choose physically motivated scale

        # Compute output normalization
        print(f"Using transform: {self.transform}")
        if self.transform == "log":
            # Log transform: y_transformed = sign(y) * log(|y| + epsilon)
            self.y_transformed = torch.tensor(self.y_sign * np.log(np.abs(self.y) + self.epsilon))
        elif self.transform == "arcsinh":

            def arcsinh_transform(y, f0):
                return torch.arcsinh(y / f0)

            self.y_transformed = arcsinh_transform(torch.tensor(self.y), self.f0)
        else:
            self.y_transformed = self.y

        self.y_mean = self.y_transformed.mean()
        self.y_std = self.y_transformed.std() + 1e-8

        # Normalize
        self.X_normalized = torch.tensor((self.X - self.X_mean) / self.X_std)
        self.y_normalized = (self.y_transformed - self.y_mean) / self.y_std

        # Normalize weights to have mean 1
        self.weights = torch.tensor(self.weights / self.weights.mean())

        print(f"Loaded {len(self.y)} samples from {data_file}")
        print(f"Input shape: {self.X.shape}")
        print(f"Output range: [{self.y.min():.4e}, {self.y.max():.4e}]")

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return (
            self.X_normalized[idx],
            self.y_normalized[idx],
            self.weights[idx],
        )

    def __getitems__(self, idxs):
        X = self.X_normalized[idxs]  # single numpy fancy-index
        y = self.y_normalized[idxs]
        w = self.weights[idxs]
        return [
            (
                X[i],
                y[i],
                w[i],
            )
            for i in range(len(idxs))
        ]

    def get_normalization_params(self) -> Dict:
        """Return normalization parameters for saving."""
        return {
            'X_mean': self.X_mean.tolist(),
            'X_std': self.X_std.tolist(),
            'y_mean': float(self.y_mean),
            'y_std': float(self.y_std),
            'transform': str(self.transform),
            'f0': float(self.f0),
            'epsilon': self.epsilon,
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
        k_y_uv = k_perp * torch.sin(phi)

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
        base_params[:, -3] = k_perp * torch.cos(phi)
        base_params[:, -2] = k_perp * torch.sin(phi)
        f_low = model(base_params).squeeze()

        # High-k point (same direction, same other params)
        high_params = base_params.clone()
        high_params[:, -3] = (alpha * k_perp) * torch.cos(phi)
        high_params[:, -2] = (alpha * k_perp) * torch.sin(phi)
        f_high = model(high_params).squeeze()

        # Target ratio: f(alpha*k) / f(k) = alpha^{-power}
        expected_ratio = alpha ** (-power)
        # Use log-ratio loss for numerical stability; avoids division-by-zero
        log_ratio = torch.log(torch.abs(f_high) + 1e-30) - torch.log(torch.abs(f_low) + 1e-30)
        target_log_ratio = torch.full_like(log_ratio, torch.log(torch.tensor(expected_ratio)))

        uv_power_law =  nn.functional.mse_loss(log_ratio, target_log_ratio)
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
            train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers,
            persistent_workers=config.num_workers > 0, pin_memory=True
        )
        val_loader = DataLoader(
            val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers,
            persistent_workers=config.num_workers > 0, pin_memory=True
        )
    elif config.device == "cpu":
        train_loader = DataLoader(
            train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers,
            persistent_workers=config.num_workers > 0
        )
        val_loader = DataLoader(
            val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers,
            persistent_workers=config.num_workers > 0
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

    if hasattr(torch, 'compile'):
        model = torch.compile(model)

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
    # ReduceLROnPlateau reduces learning rate when validation loss plateaus, helps prevent oscillation over local minima
    #
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10,  # verbose deprecated -- prints LR later on
        min_lr=6.24e-5,  # Minimum learning rate, so we don't go to negligibly small values and stagnate learning
    )

    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0

    print("\nStarting training...")
    print("-" * 70)

    try:
        for epoch in range(config.n_epochs):
            # Train
            train_metrics = train_epoch(model, train_loader, optimizer, config)

            # Validate
            val_metrics = validate(model, val_loader, config)

            # Update scheduler
            scheduler.step(val_metrics['mse'])  # Scheduler tracks mse, not the overall loss.

            # Print progress
            # if (epoch + 1) % 10 == 0 or epoch == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(
                f"Epoch {epoch + 1:3d}/{config.n_epochs} | "
                f"Train Loss: {train_metrics['loss']:.4e} | "
                f"Val Loss: {val_metrics['loss']:.4e} | "
                f"Train MSE: {train_metrics['mse']:.4e} | "
                f"Val MSE: {val_metrics['mse']:.4e} | "
                f"MSE Ratio: {(train_metrics['mse']/val_metrics['mse']):.4e} | "
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
    except KeyboardInterrupt:
        print("Keyboard interrupt. Training stopped.")

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

        print(f"Loaded model from {model_file}")
        print(f"  Validation loss: {checkpoint['val_loss']:.4e}")
        print(f"  Trained for {checkpoint['epoch'] + 1} epochs")

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

        # Convert to tensor
        inputs_tensor = torch.from_numpy(inputs).to(self.device)

        # Normalize
        inputs_norm = (inputs_tensor - self.X_mean.to(self.device)) / self.X_std.to(self.device)

        # Predict
        with torch.no_grad():
            predictions_norm = self.model(inputs_norm).squeeze()

        # Denormalize
        predictions_transformed = predictions_norm.cpu().numpy() * self.y_std + self.y_mean

        # Inverse log transform
        if self.transform == "log":
            # y_transformed = sign(y) * log(|y| + epsilon)
            # For simplicity, assume y > 0 most of the time (physical intensity)
            # Inverse: y = exp(y_transformed) - epsilon
            # But we stored sign*log(|y|+eps), so need to handle sign
            # Approximation: exp(pred) - epsilon (works when pred > 0)
            predictions = np.exp(np.abs(predictions_transformed)) - self.epsilon
            predictions = np.sign(predictions_transformed) * predictions
        elif self.transform == "arcsinh":
            # Inverse of transform above
            predictions = self.f0 * np.sinh(predictions_transformed)
        else:
            # No transformation, so predictions are already in physical units
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

    def sample_emission(self,
            E: np.ndarray,
            z0: np.ndarray,
            zf: np.ndarray,
            u_perp: np.ndarray,
            T: np.ndarray,
            g: np.ndarray,
            N_samples: int = 1,
    ) -> (float, np.ndarray):
        """
        Computes a complete grid in x, kx, ky and returns a 3D inverse CDF sample value for x, kx, ky vector.
        """

        # Grid of x, kx, ky values
        max_kx_ky = 5  # Maybe should be dependent on energy, needs testing.
        x_values = np.logspace(-4, 0, 100)
        kx_values = np.linspace(-max_kx_ky, max_kx_ky, 100)
        ky_values = np.linspace(0, max_kx_ky, 100)
        x_grid, kx_grid, ky_grid = np.meshgrid(x_values, kx_values, ky_values, indexing='ij')
        n_pts = x_grid.size

        # Predict intensity spectrum points
        I_nn_flat = self.predict(
            x=x_grid.ravel(),
            kx=kx_grid.ravel(),
            ky=ky_grid.ravel(),
            E=np.full(n_pts, E),
            z0=np.full(n_pts, z0),
            zf=np.full(n_pts, zf),
            u_perp=np.full(n_pts, u_perp),
            T=np.full(n_pts, T),
            g=np.full(n_pts, g),
        )

        # --- Reshape flat predictions back onto the 3D grid ---
        I_nn = I_nn_flat.reshape(x_grid.shape)  # shape: (n_x, n_kx, n_ky)

        # -------------------------------------------------------
        # 1. INTEGRATION
        #    Integrate over ky first, then kx, then x (in log-space).
        #    np.trapz(y, x) integrates y along the last axis by default.
        # -------------------------------------------------------
        # Integrate over ky (axis 2)
        I_kx_x = 2*np.trapezoid(I_nn, ky_values, axis=2)  # shape: (n_x, n_kx), multiply by two for symmetric -ky half of grid
        # Integrate over kx (axis 1)
        I_x = np.trapezoid(I_kx_x, kx_values, axis=1)  # shape: (n_x,)
        # Integrate over x in log-space (accounts for log-spaced grid) -- includes Jacobian, factor of x
        total_integral = np.trapezoid(I_x*x_values, np.log(x_values))  # scalar

        print(f"Total integral: {total_integral:.6e}")

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
        uniform_samples = np.random.uniform(size=N_samples)  # Should use the global RNG...
        sign_samples = np.random.choice([-1, 1], size=N_samples)
        flat_indices = np.searchsorted(cdf, uniform_samples)  # shape: (N_samples,)

        # Convert flat indices back to 3D grid indices
        ix, ikx, iky = np.unravel_index(flat_indices, I_nn.shape)

        # Look up the corresponding coordinate values
        sampled_x = x_values[ix]
        sampled_kx = kx_values[ikx]
        sampled_ky = sign_samples*ky_values[iky]  # Apply a random sign to the ky values to simulate symmetric -ky values

        # Actual longitudinal component of the momentum vector
        # We know $k^+ = x p^+$, so we apply the lightcone non-diagonal metric and the on-shell condition, $k^2 = 0$
        sampled_kz = (1/np.sqrt(2)) * (sampled_x*E - ((sampled_kx**2 + sampled_ky**2)/(2*sampled_x*E)))

        # sampled_kz = 0  # Just give zero kz for now
        # mag = np.sqrt(sampled_kx**2 + sampled_ky**2 + sampled_kz**2)
        # emission_momentum = total_integral * np.column_stack([sampled_kx/mag, sampled_ky/mag, sampled_kz/mag])
        # emission_momentum =  np.column_stack([sampled_x*E * sampled_kx / mag, sampled_x*E * sampled_ky / mag, sampled_x*E * sampled_kz / mag])
        emission_momentum = np.column_stack([sampled_kx, sampled_ky, sampled_kz])

        # # Apply random uniform jitter to point, to simulate continuous sampling
        # sampled_x += np.random.uniform(-dx / 2, dx / 2, size=N_samples)
        # sampled_kx += np.random.uniform(-dkx / 2, dkx / 2, size=N_samples)
        # sampled_ky += np.random.uniform(-dky / 2, dky / 2, size=N_samples)

        print(f"Sampled {N_samples} points.")
        print(f"  x  range: [{sampled_x.min():.4e},  {sampled_x.max():.4e}]")
        print(f"  kx range: [{sampled_kx.min():.3f}, {sampled_kx.max():.3f}]")
        print(f"  ky range: [{sampled_ky.min():.3f}, {sampled_ky.max():.3f}]")

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
    parser.add_argument("--learning-rate", type=float, default=default_config.learning_rate, help="Initial learning rate")
    parser.add_argument("--find-lr", action="store_true", help="Run LR range test and exit")
    parser.add_argument("--n-workers", type=int, default=default_config.num_workers, help="Number of workers for data loading")
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
    )

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

    if args.inference_only:
        # Demo inference only
        demo_inference(config)
    else:
        # Train the model
        train_model(config)

        # Then demo inference
        print("\n")
        demo_inference(config)
