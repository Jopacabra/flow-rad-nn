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
from pynndescent import NNDescent
from dataclasses import dataclass
from typing import Optional, Tuple, List
import time
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
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
    z0_range: Tuple[float, float] = (0.0, 10.0)
    zf_range: Tuple[float, float] = (0.0, 10.0)
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
    n_adaptive_per_round: int = 500  # Samples per adaptive round
    n_rounds: int = 10  # Number of adaptive rounds
    n_sobol_per_round: int = 2048  # Number of sobol points per round

    # Importance sampling weights
    weight_uncertainty: float = 0.1  # Weight for high MC uncertainty regions
    weight_gradient: float = 0.4  # Weight for high gradient regions
    weight_coverage: float = 0.5  # Weight for undersampled regions

    # Parallelization
    n_workers: int = 4  # Number of cores for parallelized bits

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


def _integrate_one(task):
    """Top-level worker function for parallel integration. Must be module-level for pickling."""
    idx, x, kx, ky, E, z0, zf, u_perp, T, g, config = task
    try:
        mean, sdev = integrate_point(x, kx, ky, E, T, g, u_perp, z0, zf, config)
        return idx, mean, sdev
    except Exception as e:
        print(f"  Warning: Integration failed at index {idx}: {e}")
        return idx, np.nan, np.nan


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


def generate_sobol_samples(n_samples: int, config: SamplingConfig,
                            sobol_sampler: qmc.Sobol) -> np.ndarray:
    """
    Draw the next n_samples points from a persistent Sobol sequence.
    Caller is responsible for maintaining the sampler state across rounds
    so that points are never repeated.
    """
    ranges = get_ranges(config)
    samples_unit = sobol_sampler.random(n=n_samples)
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
        self._ann_index = Optional[NNDescent]
        self.points_normalized: Optional[np.ndarray] = None
        self._points_arr: Optional[np.ndarray] = None  # numpy view of self.points

        # Cached importance weights — invalidated when new samples are added
        self._cached_weights: Optional[np.ndarray] = None

    def add_samples(self, points: np.ndarray, values: np.ndarray, errors: np.ndarray):
        """Add computed samples to the database."""
        for i in range(len(points)):
            self.points.append(points[i])
            self.values.append(values[i])
            self.errors.append(errors[i])

        # Rebuild KD-tree and invalidate weight cache
        self._rebuild_tree()
        self._cached_weights = None

    def _rebuild_tree(self):
        """Rebuild the KD-tree for nearest neighbor queries."""
        if len(self.points) < 2:
            return

        self._points_arr = np.array(self.points)
        self.points_normalized = normalize_to_unit_cube(self._points_arr, self.ranges)
        self._ann_index = NNDescent(
            self.points_normalized,
            n_neighbors=15,
            metric="euclidean",
            random_state=42,
            n_jobs=-1,
        )
        self._ann_index.prepare()  # pre-build search structures

    def compute_importance_weights(self) -> np.ndarray:
        """
        Compute importance weights for each existing sample.
        Higher weight = more likely to sample nearby.
        Returns cached result if weights are still valid.
        """
        if self._cached_weights is not None:
            return self._cached_weights

        n = len(self.points)
        if n < 10:
            return np.ones(n) / n

        errors = np.array(self.errors)

        # 1. Uncertainty-based weight: high MC error → high weight
        # Normalize by median to avoid outliers dominating
        err_median = np.median(np.abs(errors)) + 1e-10
        w_uncertainty = np.abs(errors) / err_median
        w_uncertainty = np.clip(w_uncertainty, 0.1, 10.0)

        # 2. Gradient-based weight: estimate local variation & Coverage-based weight: inverse local density
        w_gradient, w_coverage = self._estimate_weights()

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

        self._cached_weights = weights
        return weights

    def _estimate_weights(self, k_neighbors: int = 5):
        """
        Estimate gradient magnitude using nearest neighbors.
        &
        Estimate inverse local density (sparse regions get higher weight).
        """
        values = np.array(self.values)
        n_grad = len(values)
        n_cov = len(self.points)
        if n_grad < k_neighbors + 1 or self._ann_index is None:
            return np.ones(n_grad), np.ones(n_cov)

        # Batch query: all n points at once → shape (n, k_neighbors+1)
        t0 = time.time()
        indices, dists = self._ann_index.query(self.points_normalized, k=k_neighbors + 1)
        dt = time.time() - t0
        print(f"Tree query done in {dt:.1f}s")

        # dists[:, 0] is self (distance ≈ 0); slice it off
        t0 = time.time()
        neighbor_dists = dists[:, 1:]  # (n, k_neighbors)
        neighbor_indices = indices[:, 1:]  # (n, k_neighbors)
        dt = time.time() - t0
        print(f"slicing done in {dt:.1f}s")

        # Gather neighbour values: (n, k_neighbors)
        neighbor_values = values[neighbor_indices]

        # Gradient weights
        # |ΔI| / Δr for every (point, neighbour) pair, then take max over neighbours
        value_diffs = np.abs(neighbor_values - values[:, None])
        gradients = np.max(value_diffs / (neighbor_dists + 1e-10), axis=1)

        # Normalize
        grad_median = np.median(gradients) + 1e-10
        w_gradient = gradients / grad_median
        w_gradient = np.clip(w_gradient, 0.1, 10.0)

        # Coverage weights
        # Mean distance to k neighbours → local density proxy
        mean_dists = np.mean(neighbor_dists, axis=1)
        densities = 1.0 / (mean_dists + 1e-10)

        # Inverse density → sparse regions have high weight
        density_median = np.median(densities) + 1e-10
        w_coverage = density_median / (densities + 1e-10)
        w_coverage = np.clip(w_coverage, 0.1, 10.0)

        return w_gradient, w_coverage

    def generate_importance_samples(self, n_samples: int, round_num: int = 1) -> np.ndarray:
        """
        Generate new samples using importance sampling.
        Samples are drawn near existing high-weight points with added noise.
        """
        if len(self.points) < 10:
            return generate_lhs_samples(n_samples, self.config)

        weights = self.compute_importance_weights()

        # Sample indices according to importance weights
        indices = np.random.choice(len(self.points), size=n_samples, p=weights)

        # Direct numpy fancy-index instead of Python list comprehension
        base_points = self._points_arr[indices]

        # Noise scale decreases with round: broad early, refined later
        noise_scale = max(0.02, 0.15 * np.exp(-round_num / 50))
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

        # Persistent Sobol sampler — advances across all rounds so points are never repeated.
        # Seeded differently from the initial LHS sampler (seed=42) to avoid overlap.
        self._sobol_sampler = qmc.Sobol(d=len(get_ranges(config)))  # No seed, resuming works
        # Fast-forward past the initial round's worth of points so the sequence
        # never overlaps with generate_lhs_samples (which also uses Sobol, seed=42).
        # (Different seeds already guarantee no overlap, but this is explicit.)

        # Results storage
        self.all_points = []
        self.all_values = []
        self.all_errors = []
        self.all_weights = []

        # Track starting round for resumption
        self.starting_round = 0

    def load_checkpoint(self, filename: str) -> int:
        """
        Load previously computed data from an HDF5 checkpoint file.

        Parameters
        ----------
        filename : str
            Path to the checkpoint HDF5 file

        Returns
        -------
        int
            The round number extracted from filename (for resumption),
            or 0 if not determinable
        """
        print("=" * 70)
        print(f"Loading checkpoint: {filename}")
        print("=" * 70)

        with h5py.File(filename, 'r') as f:
            # Load all features - note: file contains mirrored ky data
            x = f['x'][:]
            kx = f['kx'][:]
            ky = f['ky'][:]
            E = f['E'][:]
            z0 = f['z0'][:]
            zf = f['zf'][:]
            u_perp = f['u_perp'][:]
            T = f['T'][:]
            g = f['g'][:]

            I = f['I'][:]
            I_err = f['I_err'][:]
            weights = f['weight'][:]

            n_total = len(I)
            n_original = f.attrs.get('n_original', n_total // 2)

        # Only keep the original points (not the mirrored ky ones)
        # The mirrored points are the second half of the array
        x = x[:n_original]
        kx = kx[:n_original]
        ky = ky[:n_original]
        E = E[:n_original]
        z0 = z0[:n_original]
        zf = zf[:n_original]
        u_perp = u_perp[:n_original]
        T = T[:n_original]
        g = g[:n_original]
        I = I[:n_original]
        I_err = I_err[:n_original]
        weights = weights[:n_original]

        # Stack into points array
        points = np.column_stack([x, kx, ky, E, z0, zf, u_perp, T, g])

        # Filter out any NaN values
        valid = np.isfinite(I) & np.isfinite(I_err)
        points = points[valid]
        I = I[valid]
        I_err = I_err[valid]
        weights = weights[valid]

        # Store in our lists
        for i in range(len(I)):
            self.all_points.append(points[i])
            self.all_values.append(I[i])
            self.all_errors.append(I_err[i])
            self.all_weights.append(weights[i])

        # Add to importance sampler
        self.sampler.add_samples(points, I, I_err)

        print(f"  Loaded {len(I)} original samples (excluding mirrored ky)")
        print(f"  Value range: [{I.min():.4e}, {I.max():.4e}]")
        print(f"  Mean error: {I_err.mean():.4e}")

        # Try to extract round number from filename
        # Expected format: radiation_training_data_checkpoint_r96.h5
        import re
        match = re.search(r'_r(\d+)\.h5$', filename)
        if match:
            round_num = int(match.group(1))
            print(f"  Detected round number: {round_num}")
            return round_num
        else:
            print(f"  Could not detect round number from filename")
            return 0

    def compute_batch(self, points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute intensity values for a batch of parameter points in parallel."""
        n = len(points)
        values = np.full(n, np.nan)
        errors = np.full(n, np.nan)

        tasks = [
            (i, *points[i], self.config)  # (idx, x, kx, ky, E, z0, zf, u_perp, T, g, config)
            for i in range(n)
        ]

        print(f"  Computing {n} points on {self.config.n_workers or 'all available'} workers...")
        t0 = time.time()
        completed = 0

        with ProcessPoolExecutor(max_workers=self.config.n_workers) as pool:
            futures = {pool.submit(_integrate_one, task): task for task in tasks}
            for future in as_completed(futures):
                idx, mean, sdev = future.result()
                values[idx] = mean
                errors[idx] = sdev
                completed += 1
                if completed % max(1, n // 10) == 0:
                    elapsed = time.time() - t0
                    eta = elapsed / completed * (n - completed)
                    print(f"    {completed}/{n} done "
                          f"({elapsed:.0f}s elapsed, ~{eta:.0f}s remaining)")

        dt = time.time() - t0
        print(f"  Batch complete in {dt:.1f}s ({dt / n:.2f}s per point)")
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
        print(f"  Filtering out failed integrations...")
        t0 = time.time()
        valid = ~np.isnan(values)
        points = points[valid]
        values = values[valid]
        errors = errors[valid]
        dt = time.time() - t0
        print(f"  Done in {dt:.1f}s")

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

        n_sobol = self.config.n_sobol_per_round
        n_adaptive = self.config.n_adaptive_per_round

        # --- Importance-sampled points ---
        if n_adaptive > 0:
            print(f"Sampling {n_adaptive} importance-sampled points...")
            t0 = time.time()
            adaptive_points = self.sampler.generate_importance_samples(n_adaptive)
            dt = time.time() - t0
            print(f"  Done in {dt:.1f}s")

        # --- Sobol exploration points ---
        if n_sobol > 0:
            print(f"Sampling {n_sobol} sobol points...")
            t0 = time.time()

            sobol_points = generate_sobol_samples(n_sobol, self.config, self._sobol_sampler)
            # Enforce z0 < zf on Sobol points
            z0_idx, zf_idx = 4, 5
            mask = sobol_points[:, z0_idx] >= sobol_points[:, zf_idx]
            if mask.any():
                sobol_points[mask, z0_idx], sobol_points[mask, zf_idx] = (
                    sobol_points[mask, zf_idx].copy(),
                    sobol_points[mask, z0_idx].copy(),
                )
            dt = time.time() - t0
            print(f"  Done in {dt:.1f}s")

        # --- Combine points ---
        print(f"  Split: {n_adaptive} importance-sampled + {n_sobol} Sobol exploration")
        if n_sobol > 0 and n_adaptive > 0:
            points = np.vstack([adaptive_points, sobol_points])
        elif n_sobol > 0:
            points = sobol_points
        else:
            points = adaptive_points

        print(f"  Computing {len(points)} samples...")
        t0 = time.time()
        values, errors = self.compute_batch(points)
        dt = time.time() - t0
        print(f"  Done in {dt:.1f}s ({dt / len(points):.2f}s per point)")

        # Filter out failed integrations
        print(f"  Filtering out failed integrations...")
        t0 = time.time()
        valid = ~np.isnan(values)
        points = points[valid]
        values = values[valid]
        errors = errors[valid]
        dt = time.time() - t0
        print(f"  Done in {dt:.1f}s")

        # Compute importance weights for these new samples.
        # Sobol points are treated as uniform draws → weight = 1.
        # Importance-sampled points get inverse-probability weights.
        if n_adaptive > 0:
            print(f"  Importance sampling weights for next round...")
            t0 = time.time()
            if len(self.sampler.points) > 10:
                sampling_weights = self.sampler.compute_importance_weights()
                points_norm = normalize_to_unit_cube(points, self.sampler.ranges)
                indices, _ = self.sampler._ann_index.query(points_norm, k=1)
                indices = indices.ravel()  # (n, 1) -> (n,)

                sample_probs = sampling_weights[indices]
                weights = 1.0 / (sample_probs * len(self.sampler.points) + 1e-10)
                weights = np.clip(weights, 0.1, 10.0)

                # Override weights for the Sobol portion: they were drawn uniformly
                if n_sobol > 0:
                    # The Sobol points sit at the end of the (post-filter) array.
                    # Recompute how many survived the validity filter.
                    n_adaptive_valid = int(valid[:n_adaptive].sum())
                    weights[n_adaptive_valid:] = 1.0
            else:
                weights = np.ones(len(points))
            dt = time.time() - t0
            print(f"  Done in {dt:.1f}s")
        else:
            weights = np.ones(len(points))

        # Store results
        self._store_results(points, values, errors, weights)
        self.sampler.add_samples(points, values, errors)

        print(f"  Valid samples: {len(points)}")
        print(f"  Value range: [{values.min():.4e}, {values.max():.4e}]")
        print(f"  Mean error: {errors.mean():.4e}")

        # Print importance sampling statistics
        if n_adaptive > 0:
            print(f"Printing importance sampling stats...")
            t0 = time.time()
            if len(self.sampler.points) > 10:
                imp_weights = sampling_weights
                print(f"  Importance weight stats: min={imp_weights.min():.4f}, "
                      f"max={imp_weights.max():.4f}, entropy={self._entropy(imp_weights):.2f}")
            dt = time.time() - t0
            print(f"  Done in {dt:.1f}s")

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

    def run(self, resume_from: Optional[str] = None):
        """
        Run the full data generation pipeline.

        Parameters
        ----------
        resume_from : str, optional
            Path to checkpoint file to resume from. If provided, skips
            initial round and continues from the detected round number.
        """
        print("\n" + "=" * 70)
        print("TRAINING DATA GENERATION FOR RADIATION PINN")
        print("=" * 70)
        print(f"Configuration:")
        print(f"  Initial samples: {self.config.n_initial}")
        print(f"  Adaptive rounds: {self.config.n_rounds}")
        print(f"  Samples per round: {self.config.n_adaptive_per_round}")
        print(f"  Total expected: ~{self.config.n_initial + self.config.n_rounds * self.config.n_adaptive_per_round}")
        print(f"  Output file: {self.config.output_file}")
        print(f"  Parameter dimensions: 9 (x, kx, ky, E, z0, zf, u_perp, T, g)")
        print(f"  Note: CF factor NOT included in output")
        print()

        # Determine starting point
        if resume_from is not None:
            self.starting_round = self.load_checkpoint(resume_from)
            print(f"\nResuming from round {self.starting_round + 1}")
        else:
            # Initial round
            self.run_initial_round()
            self.starting_round = 0

        # Adaptive rounds
        for r in range(self.starting_round + 1, self.config.n_rounds + 1):
            self.run_adaptive_round(r)

            # Checkpoint
            if r % self.config.checkpoint_every == 0:
                print("Checkpointing...")
                t0 = time.time()
                checkpoint_file = self.config.output_file.replace('.h5', f'_checkpoint_r{r}.h5')
                self.save_checkpoint(checkpoint_file)
                dt = time.time() - t0
                print(f"  Done in {dt:.1f}s")

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
    import argparse

    parser = argparse.ArgumentParser(description="Generate training data for radiation PINN")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint file to resume from")
    args = parser.parse_args()

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
        T_range=(0.150, 0.650),  # Temperature range in GeV
        g_range=(1.5, 2.5),  # Coupling g = sqrt(4*pi*alpha_s)

        # Integration
        q_lim_factor=0.3,
        nitn_warmup=10,
        nitn=10,
        neval=10_000,

        # Sampling
        n_initial=131072,  # Use powers of 2 for optimal sobol sampling
        n_adaptive_per_round=0,
        n_rounds=2500,  # Increase for more data
        n_sobol_per_round=8192,  # Number drawn from Sobol sequence each round

        # Importance weights
        weight_uncertainty=0.1,  # Weight for high MC uncertainty regions
        weight_gradient=0.4,  # Weight for high gradient regions
        weight_coverage=0.5,  # Weight for undersampled regions

        # Parallelization
        n_workers=4,  # Number of cores for parallelized bits

        # Output
        output_file="radiation_training_data.h5",
        checkpoint_every=3,
    )

    # Run generation
    generator = TrainingDataGenerator(config)  # control number of workers here.
    generator.run(resume_from=args.resume)
