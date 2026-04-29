"""
generate_training_data.py

Generates training data for a PINN emulator of the medium-induced radiation
intensity distribution. Uses adaptive importance sampling to automatically
emphasize regions of high uncertainty or high function variation.

The sampling strategy works in rounds:
1. Initial round: Latin Hypercube Sampling (LHS) for uniform coverage
2. Subsequent rounds: Importance sampling based on:
   - High MC integration uncertainty (hard to integrate → need more samples)
   - High gradient magnitude (estimated from neighbors → rapidly varying regions)
   - Undersampled regions (to maintain coverage)

Output: HDF5 file with columns [x, kx, ky, E, z0, zf, u_perp, T, g, I, I_err, weight]

Note: The Casimir factor (CF) is NOT included in the output. Multiply by CF at runtime
      depending on the hard particle flavor (4/3 for quarks, 3 for gluons).
"""

from scipy.stats import qmc
from scipy.spatial import cKDTree
from dataclasses import dataclass
from typing import Optional, Tuple, List
import time
import sys
import numpy as np
import vegas
import h5py

# Import medium property functions from plasma_interaction
sys.path.append("/home/jo/PycharmProjects/ape")
import plasma_interaction


# ==============================================================================
# Physical constants
# ==============================================================================
NC = 3

# Soft parton species for rho0 summation
SOFT_PIDS = [1, -1, 2, -2, 3, -3, 21]


# ==============================================================================
# Configuration
# ==============================================================================
@dataclass
class SamplingConfig:
    """Configuration for the training data generation."""

    # Parameter ranges: (min, max) for each input dimension
    # Kinematic parameters
    x_range: Tuple[float, float] = (0.01, 0.99)
    kx_range: Tuple[float, float] = (-5.0, 5.0)
    ky_range: Tuple[float, float] = (0.0, 5.0)  # Only ky >= 0 due to symmetry
    E_range: Tuple[float, float] = (5.0, 100.0)
    z0_range: Tuple[float, float] = (0.0, 2.0)
    zf_range: Tuple[float, float] = (2.0, 10.0)
    u_perp_range: Tuple[float, float] = (0.0, 0.7)

    # Medium parameters (now sampled)
    T_range: Tuple[float, float] = (0.150, 0.500)  # Temperature in GeV
    g_range: Tuple[float, float] = (1.5, 2.5)  # Coupling constant g = sqrt(4*pi*alpha_s)

    # Integration settings
    q_lim_factor: float = 0.3  # q integration limits = ±factor * E
    nitn_warmup: int = 5
    nitn: int = 8
    neval: int = 3000

    # Sampling settings
    n_initial: int = 1000  # Initial LHS samples
    n_per_round: int = 500  # Samples per adaptive round
    n_rounds: int = 10  # Number of adaptive rounds

    # Importance sampling weights
    weight_uncertainty: float = 0.4  # Weight for high MC uncertainty regions
    weight_gradient: float = 0.3  # Weight for high gradient regions
    weight_coverage: float = 0.3  # Weight for undersampled regions

    # Output
    output_file: str = "radiation_training_data.h5"
    checkpoint_every: int = 2  # Save checkpoint every N rounds


# ==============================================================================
# Medium property helpers
# ==============================================================================
def compute_rho0(T: float) -> float:
    """
    Compute total density rho0 by summing over all soft parton species.
    Uses plasma_interaction.rho() for each species.
    """
    rho0 = 0.0
    for soft_pid in SOFT_PIDS:
        rho0 += plasma_interaction.rho(T, soft_pid=soft_pid)
    return rho0


def compute_mu(T: float, g: float) -> float:
    """
    Compute Debye mass using plasma_interaction.mu_DeBye().
    """
    return plasma_interaction.mu_DeBye(T, g=g)


# ==============================================================================
# Integrand (adapted from radiation.py)
# ==============================================================================
def make_batch_integrand(x, kx, ky, E, T, g, u_perp):
    """
    Constructs a vegas batch integrand for the medium-induced gluon radiation
    intensity at fixed (x, kx, ky, E, T, g, u_perp), integrating over (qx, qy, z).

    Assumes u_y = 0 (flow along +x only).

    Note: The Casimir factor (CF) is NOT included. Multiply by CF at runtime
          depending on the hard particle flavor (4/3 for quarks, 3 for gluons).
    """
    # Compute alpha_s from g
    alpha_s = g**2 / (4 * np.pi)

    # Medium properties derived from T and g
    rho0 = compute_rho0(T)
    mu = compute_mu(T, g)
    _mu2 = mu ** 2

    # Flow (u_y = 0 by assumption)
    ux = u_perp
    uy = 0.0

    # Constants WITHOUT CF factor (user will multiply by CF later)
    _constants = alpha_s / (16 * (np.pi ** 2))

    kk = kx * kx + ky * ky
    ku = kx * ux + ky * uy
    uu = ux * ux + uy * uy

    # Protect against kk = 0
    if kk < 1e-10:
        kk = 1e-10

    # Precompute g^4 for scattering potential
    g4 = g ** 4

    @vegas.batchintegrand
    def integrand(pts):
        qx = pts[:, 0]
        qy = pts[:, 1]
        z = pts[:, 2]

        kmqx = kx - qx
        kmqy = ky - qy

        qq = qx * qx + qy * qy
        kq = kx * qx + ky * qy
        qu = qx * ux + qy * uy
        kmqu = kmqx * ux + kmqy * uy
        kmqq = kmqx * qx + kmqy * qy
        kmqkmq = kmqx ** 2 + kmqy ** 2

        # Protect against division by zero
        kmqkmq = np.maximum(kmqkmq, 1e-10)

        q2_mu2 = qq + _mu2
        R_sq = qq + _mu2 - qu ** 2
        R_sq = np.maximum(R_sq, 1e-10)  # Numerical protection
        R = np.sqrt(R_sq)

        # Scattering potential (uses g^4)
        v2 = g4 / (q2_mu2 ** 2)
        vm2dv2 = -2.0 / q2_mu2

        # Term 1
        t1 = (
            (
                (4.0 * kq / (kk * kmqkmq))
                - (2.0 / (x * E)) * (1.0 / (kk * kmqkmq)) * (
                    2.0 * kmqu * kk
                    + 2.0 * ku * kmqq
                    + kq * qu * (2.0 * kk - kmqkmq) * vm2dv2
                )
            )
            * (1 - np.cos((kmqkmq / (2.0 * x * E)) * z))
        )

        # Term 2
        t2 = (
            (1.0 / (x * E)) * (ku / kk)
            * (4.0 + qq * vm2dv2)
            * (1 - np.cos((kk / (2.0 * x * E)) * z))
        )

        # Term 3
        t3 = (
            (-1)
            * (1.0 / (4 * (R ** 3) * x * E))
            * (_mu2 * ((qu ** 4) + 6 * (qu ** 2) * (R ** 2) - 3 * R ** 4)
               - 2 * (R ** 2) * ((qu ** 4) - (R ** 4)))
            * vm2dv2
            * np.sin((kk / (2.0 * x * E)) * z)
        )

        # Term 4
        t4 = (
            (1 / (x * E))
            * ((kk * (1 - uu) + (ku ** 2)) / (kk ** 2))
            * ((((R ** 2) + (qu ** 2)) ** 2) / (R ** 3))
            * np.sin((kk / (2.0 * x * E)) * z)
        )

        return _constants * rho0 * v2 * (t1 + t2 + t3 + t4)

    return integrand


def integrate_point(x, kx, ky, E, T, g, u_perp, z0, zf, config: SamplingConfig):
    """
    Integrate the radiation intensity for a single parameter point.
    Returns (mean, sdev).
    """
    q_lim = config.q_lim_factor * E
    integration_region = [(-q_lim, q_lim), (-q_lim, q_lim), (z0, zf)]

    integ = vegas.Integrator(integration_region)
    integrand = make_batch_integrand(x, kx, ky, E, T, g, u_perp)

    # Warm-up
    integ(integrand, nitn=config.nitn_warmup, neval=config.neval)

    # Production
    result = integ(integrand, nitn=config.nitn, neval=config.neval)

    return result.mean, result.sdev


# ==============================================================================
# Sampling utilities
# ==============================================================================
def normalize_to_unit_cube(points: np.ndarray, ranges: List[Tuple[float, float]]) -> np.ndarray:
    """Normalize points from physical ranges to [0, 1]^d."""
    ranges = np.array(ranges)
    return (points - ranges[:, 0]) / (ranges[:, 1] - ranges[:, 0])


def denormalize_from_unit_cube(points: np.ndarray, ranges: List[Tuple[float, float]]) -> np.ndarray:
    """Denormalize points from [0, 1]^d to physical ranges."""
    ranges = np.array(ranges)
    return points * (ranges[:, 1] - ranges[:, 0]) + ranges[:, 0]


def get_ranges(config: SamplingConfig) -> List[Tuple[float, float]]:
    """Extract parameter ranges as a list.

    Order: [x, kx, ky, E, z0, zf, u_perp, T, g]
    """
    return [
        config.x_range,
        config.kx_range,
        config.ky_range,
        config.E_range,
        config.z0_range,
        config.zf_range,
        config.u_perp_range,
        config.T_range,
        config.g_range,
    ]


def generate_lhs_samples(n_samples: int, config: SamplingConfig) -> np.ndarray:
    """Generate Latin Hypercube samples in the parameter space."""
    ranges = get_ranges(config)
    n_dims = len(ranges)

    sampler = qmc.LatinHypercube(d=n_dims, seed=42)
    samples_unit = sampler.random(n=n_samples)
    samples = denormalize_from_unit_cube(samples_unit, ranges)

    return samples


class ImportanceSampler:
    """
    Adaptive importance sampler that emphasizes regions based on:
    1. High MC integration uncertainty
    2. High estimated gradient (function variation)
    3. Low sample density (coverage)
    """

    def __init__(self, config: SamplingConfig):
        self.config = config
        self.ranges = get_ranges(config)
        self.n_dims = len(self.ranges)

        # Storage for all computed points
        self.points = []  # List of (x, kx, ky, E, z0, zf, u_perp, T, g)
        self.values = []  # List of I values
        self.errors = []  # List of I_err values

        # KD-tree for nearest neighbor queries (rebuilt after each round)
        self.tree: Optional[cKDTree] = None
        self.points_normalized: Optional[np.ndarray] = None

    def add_samples(self, points: np.ndarray, values: np.ndarray, errors: np.ndarray):
        """Add computed samples to the database."""
        for i in range(len(points)):
            self.points.append(points[i])
            self.values.append(values[i])
            self.errors.append(errors[i])

        # Rebuild KD-tree
        self._rebuild_tree()

    def _rebuild_tree(self):
        """Rebuild the KD-tree for nearest neighbor queries."""
        if len(self.points) < 2:
            return

        points_arr = np.array(self.points)
        self.points_normalized = normalize_to_unit_cube(points_arr, self.ranges)
        self.tree = cKDTree(self.points_normalized)

    def compute_importance_weights(self) -> np.ndarray:
        """
        Compute importance weights for each existing sample.
        Higher weight = more likely to sample nearby.
        """
        n = len(self.points)
        if n < 10:
            return np.ones(n) / n

        values = np.array(self.values)
        errors = np.array(self.errors)

        # 1. Uncertainty-based weight: high MC error → high weight
        # Normalize by median to avoid outliers dominating
        err_median = np.median(np.abs(errors)) + 1e-10
        w_uncertainty = np.abs(errors) / err_median
        w_uncertainty = np.clip(w_uncertainty, 0.1, 10.0)

        # 2. Gradient-based weight: estimate local variation
        w_gradient = self._estimate_gradient_weights(values)

        # 3. Coverage-based weight: inverse local density
        w_coverage = self._estimate_coverage_weights()

        # Combine weights
        cfg = self.config
        weights = (
            cfg.weight_uncertainty * w_uncertainty +
            cfg.weight_gradient * w_gradient +
            cfg.weight_coverage * w_coverage
        )

        # Normalize to probability distribution
        weights = np.maximum(weights, 1e-10)
        weights /= weights.sum()

        return weights

    def _estimate_gradient_weights(self, values: np.ndarray, k_neighbors: int = 5) -> np.ndarray:
        """Estimate gradient magnitude using nearest neighbors."""
        n = len(values)
        if n < k_neighbors + 1 or self.tree is None:
            return np.ones(n)

        gradients = np.zeros(n)

        for i in range(n):
            # Find k nearest neighbors
            dists, indices = self.tree.query(self.points_normalized[i], k=k_neighbors + 1)

            # Exclude self (distance = 0)
            neighbor_mask = dists > 1e-10
            if neighbor_mask.sum() == 0:
                continue

            neighbor_dists = dists[neighbor_mask]
            neighbor_indices = indices[neighbor_mask]
            neighbor_values = values[neighbor_indices]

            # Estimate gradient as max |ΔI / Δr|
            value_diffs = np.abs(neighbor_values - values[i])
            gradients[i] = np.max(value_diffs / (neighbor_dists + 1e-10))

        # Normalize
        grad_median = np.median(gradients) + 1e-10
        w_gradient = gradients / grad_median
        w_gradient = np.clip(w_gradient, 0.1, 10.0)

        return w_gradient

    def _estimate_coverage_weights(self, k_neighbors: int = 5) -> np.ndarray:
        """Estimate inverse local density (sparse regions get higher weight)."""
        n = len(self.points)
        if n < k_neighbors + 1 or self.tree is None:
            return np.ones(n)

        # Average distance to k nearest neighbors
        densities = np.zeros(n)

        for i in range(n):
            dists, _ = self.tree.query(self.points_normalized[i], k=k_neighbors + 1)
            # Exclude self
            dists = dists[dists > 1e-10]
            if len(dists) > 0:
                densities[i] = 1.0 / (np.mean(dists) + 1e-10)

        # Inverse density → sparse regions have high weight
        density_median = np.median(densities) + 1e-10
        w_coverage = density_median / (densities + 1e-10)
        w_coverage = np.clip(w_coverage, 0.1, 10.0)

        return w_coverage

    def generate_importance_samples(self, n_samples: int) -> np.ndarray:
        """
        Generate new samples using importance sampling.
        Samples are drawn near existing high-weight points with added noise.
        """
        if len(self.points) < 10:
            # Fall back to LHS if not enough data
            return generate_lhs_samples(n_samples, self.config)

        weights = self.compute_importance_weights()

        # Sample indices according to importance weights
        indices = np.random.choice(len(self.points), size=n_samples, p=weights)

        # Generate new points by adding noise around selected points
        base_points = np.array([self.points[i] for i in indices])

        # Noise scale: fraction of parameter range, decreasing with round
        noise_scale = 0.1  # 10% of range
        ranges = np.array(self.ranges)
        range_widths = ranges[:, 1] - ranges[:, 0]

        noise = np.random.normal(0, noise_scale, size=base_points.shape) * range_widths
        new_points = base_points + noise

        # Clip to valid ranges
        new_points = np.clip(new_points, ranges[:, 0], ranges[:, 1])

        # Ensure z0 < zf constraint
        z0_idx, zf_idx = 4, 5
        mask = new_points[:, z0_idx] >= new_points[:, zf_idx]
        if mask.any():
            # Swap z0 and zf where violated
            new_points[mask, z0_idx], new_points[mask, zf_idx] = \
                new_points[mask, zf_idx].copy(), new_points[mask, z0_idx].copy()

        return new_points


# ==============================================================================
# Data generation pipeline
# ==============================================================================
class TrainingDataGenerator:
    """Main class for generating training data with adaptive importance sampling."""

    def __init__(self, config: SamplingConfig):
        self.config = config
        self.sampler = ImportanceSampler(config)

        # Results storage
        self.all_points = []
        self.all_values = []
        self.all_errors = []
        self.all_weights = []

    def compute_batch(self, points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute intensity values for a batch of parameter points."""
        n = len(points)
        values = np.zeros(n)
        errors = np.zeros(n)

        for i, pt in enumerate(points):
            x, kx, ky, E, z0, zf, u_perp, T, g = pt

            try:
                mean, sdev = integrate_point(
                    x, kx, ky, E, T, g, u_perp, z0, zf, self.config
                )
                values[i] = mean
                errors[i] = sdev
            except Exception as e:
                print(f"  Warning: Integration failed at point {i}: {e}")
                values[i] = np.nan
                errors[i] = np.nan

            if (i + 1) % 50 == 0:
                print(f"    Computed {i + 1}/{n} points...")

        return values, errors

    def run_initial_round(self):
        """Run the initial LHS sampling round."""
        print("=" * 70)
        print("Round 0: Initial Latin Hypercube Sampling")
        print("=" * 70)

        points = generate_lhs_samples(self.config.n_initial, self.config)

        # Ensure z0 < zf
        z0_idx, zf_idx = 4, 5
        for i in range(len(points)):
            if points[i, z0_idx] >= points[i, zf_idx]:
                points[i, z0_idx], points[i, zf_idx] = points[i, zf_idx], points[i, z0_idx]

        print(f"  Computing {len(points)} initial samples...")
        t0 = time.time()
        values, errors = self.compute_batch(points)
        dt = time.time() - t0
        print(f"  Done in {dt:.1f}s ({dt/len(points):.2f}s per point)")

        # Filter out failed integrations
        valid = ~np.isnan(values)
        points = points[valid]
        values = values[valid]
        errors = errors[valid]

        # Initial weights are uniform
        weights = np.ones(len(points))

        # Store results
        self._store_results(points, values, errors, weights)
        self.sampler.add_samples(points, values, errors)

        print(f"  Valid samples: {len(points)}")
        print(f"  Value range: [{values.min():.4e}, {values.max():.4e}]")
        print(f"  Mean error: {errors.mean():.4e}")

    def run_adaptive_round(self, round_num: int):
        """Run an adaptive importance sampling round."""
        print("=" * 70)
        print(f"Round {round_num}: Adaptive Importance Sampling")
        print("=" * 70)

        # Generate new samples using importance sampling
        points = self.sampler.generate_importance_samples(self.config.n_per_round)

        print(f"  Computing {len(points)} adaptive samples...")
        t0 = time.time()
        values, errors = self.compute_batch(points)
        dt = time.time() - t0
        print(f"  Done in {dt:.1f}s ({dt/len(points):.2f}s per point)")

        # Filter out failed integrations
        valid = ~np.isnan(values)
        points = points[valid]
        values = values[valid]
        errors = errors[valid]

        # Compute importance weights for these new samples
        # (inverse of sampling probability for unbiased training)
        if len(self.sampler.points) > 10:
            sampling_weights = self.sampler.compute_importance_weights()
            # Find which existing points these were sampled near
            # For simplicity, use uniform weights adjusted by local density
            points_norm = normalize_to_unit_cube(points, self.sampler.ranges)
            _, indices = self.sampler.tree.query(points_norm, k=1)

            # Weight = 1 / sampling_probability (capped for stability)
            sample_probs = sampling_weights[indices]
            weights = 1.0 / (sample_probs * len(self.sampler.points) + 1e-10)
            weights = np.clip(weights, 0.1, 10.0)
        else:
            weights = np.ones(len(points))

        # Store results
        self._store_results(points, values, errors, weights)
        self.sampler.add_samples(points, values, errors)

        print(f"  Valid samples: {len(points)}")
        print(f"  Value range: [{values.min():.4e}, {values.max():.4e}]")
        print(f"  Mean error: {errors.mean():.4e}")

        # Print importance sampling statistics
        if len(self.sampler.points) > 10:
            imp_weights = self.sampler.compute_importance_weights()
            print(f"  Importance weight stats: min={imp_weights.min():.4f}, "
                  f"max={imp_weights.max():.4f}, entropy={self._entropy(imp_weights):.2f}")

    def _entropy(self, probs: np.ndarray) -> float:
        """Compute entropy of probability distribution (higher = more uniform)."""
        probs = probs[probs > 1e-10]
        return -np.sum(probs * np.log(probs))

    def _store_results(self, points, values, errors, weights):
        """Store results in memory."""
        for i in range(len(points)):
            self.all_points.append(points[i])
            self.all_values.append(values[i])
            self.all_errors.append(errors[i])
            self.all_weights.append(weights[i])

    def save_checkpoint(self, filename: Optional[str] = None):
        """Save current results to HDF5 file."""
        if filename is None:
            filename = self.config.output_file

        points = np.array(self.all_points)
        values = np.array(self.all_values)
        errors = np.array(self.all_errors)
        weights = np.array(self.all_weights)

        # Also save mirrored ky data (exploit symmetry)
        points_mirror = points.copy()
        points_mirror[:, 2] = -points_mirror[:, 2]  # ky → -ky

        # Combine original and mirrored
        points_full = np.vstack([points, points_mirror])
        values_full = np.concatenate([values, values])
        errors_full = np.concatenate([errors, errors])
        weights_full = np.concatenate([weights, weights])

        with h5py.File(filename, 'w') as f:
            # Input features (order: x, kx, ky, E, z0, zf, u_perp, T, g)
            f.create_dataset('x', data=points_full[:, 0])
            f.create_dataset('kx', data=points_full[:, 1])
            f.create_dataset('ky', data=points_full[:, 2])
            f.create_dataset('E', data=points_full[:, 3])
            f.create_dataset('z0', data=points_full[:, 4])
            f.create_dataset('zf', data=points_full[:, 5])
            f.create_dataset('u_perp', data=points_full[:, 6])
            f.create_dataset('T', data=points_full[:, 7])
            f.create_dataset('g', data=points_full[:, 8])

            # Outputs
            f.create_dataset('I', data=values_full)
            f.create_dataset('I_err', data=errors_full)
            f.create_dataset('weight', data=weights_full)

            # Metadata
            f.attrs['n_samples'] = len(points_full)
            f.attrs['n_original'] = len(points)
            f.attrs['soft_pids'] = SOFT_PIDS
            f.attrs['description'] = (
                'Training data for medium-induced radiation PINN. '
                'Includes original samples and ky-mirrored samples (symmetry). '
                'CF factor NOT included - multiply by CF at runtime '
                '(4/3 for quarks, 3 for gluons). '
                'Uses plasma_interaction.mu_DeBye() for Debye mass and '
                'plasma_interaction.rho() summed over soft_pids for density.'
            )

        print(f"  Saved {len(points_full)} samples to {filename}")

    def run(self):
        """Run the full data generation pipeline."""
        print("\n" + "=" * 70)
        print("TRAINING DATA GENERATION FOR RADIATION PINN")
        print("=" * 70)
        print(f"Configuration:")
        print(f"  Initial samples: {self.config.n_initial}")
        print(f"  Adaptive rounds: {self.config.n_rounds}")
        print(f"  Samples per round: {self.config.n_per_round}")
        print(f"  Total expected: ~{self.config.n_initial + self.config.n_rounds * self.config.n_per_round}")
        print(f"  Output file: {self.config.output_file}")
        print(f"  Parameter dimensions: 9 (x, kx, ky, E, z0, zf, u_perp, T, g)")
        print(f"  Note: CF factor NOT included in output")
        print()

        # Initial round
        self.run_initial_round()

        # Adaptive rounds
        for r in range(1, self.config.n_rounds + 1):
            self.run_adaptive_round(r)

            # Checkpoint
            if r % self.config.checkpoint_every == 0:
                checkpoint_file = self.config.output_file.replace('.h5', f'_checkpoint_r{r}.h5')
                self.save_checkpoint(checkpoint_file)

        # Final save
        self.save_checkpoint()

        print("\n" + "=" * 70)
        print("GENERATION COMPLETE")
        print("=" * 70)
        print(f"Total samples: {len(self.all_points)} (x2 with ky symmetry)")
        print(f"Output file: {self.config.output_file}")


# ==============================================================================
# Main entry point
# ==============================================================================
if __name__ == "__main__":
    # Configure the generation
    config = SamplingConfig(
        # Kinematic parameter ranges
        x_range=(0.01, 0.99),
        kx_range=(-5.0, 5.0),
        ky_range=(0.0, 5.0),  # Only positive due to symmetry
        E_range=(1.0, 100.0),
        z0_range=(0.0, 10.0),
        zf_range=(0.0, 10.0),
        u_perp_range=(0.0, 0.9),

        # Medium parameter ranges
        T_range=(0.150, 0.650),  # Temperature in GeV
        g_range=(1.5, 2.5),  # Coupling g = sqrt(4*pi*alpha_s)

        # Integration
        q_lim_factor=0.3,
        nitn_warmup=10,
        nitn=10,
        neval=10_000,

        # Sampling
        n_initial=500,  # Start smaller for testing; increase for production
        n_per_round=200,
        n_rounds=250,

        # Importance weights
        weight_uncertainty=0.4,
        weight_gradient=0.3,
        weight_coverage=0.3,

        # Output
        output_file="radiation_training_data.h5",
        checkpoint_every=10,
    )

    # Run generation
    generator = TrainingDataGenerator(config)
    generator.run()