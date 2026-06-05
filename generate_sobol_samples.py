"""
generate_sobol_batch.py

Generates a single batch of Sobol-sampled training data for the radiation PINN
emulator. Designed to be run as a Slurm job array, where each array task writes
an independent HDF5 file. Merge outputs afterwards with merge.py.

Integration and physical model are identical to generate_training_data.py.

Usage (standalone):
    python generate_sobol_batch.py --batch-id 0 --n-points 4096 --n-workers 8

Usage (Slurm job array):
    sbatch submit_sobol.sh
"""

import argparse
import sys
import time
import numpy as np
import h5py
from pathlib import Path
from scipy.stats import qmc
from concurrent.futures import ProcessPoolExecutor, as_completed

import vegas

# Allow import of plasma_interaction from the parent ape directory
ape_dir = str(Path(__file__).resolve().parent.parent)
sys.path.append(ape_dir)
import plasma_interaction


# ==============================================================================
# Physical constants
# ==============================================================================
NC = 3
SOFT_PIDS = [1, -1, 2, -2, 3, -3, 21]


# ==============================================================================
# Parameter space
# ==============================================================================
# Order: [x, kx, ky, E, z0, zf, u_perp, T, g]
PARAM_RANGES = [
    (0.01, 0.99),   # x
    (-5.0, 5.0),    # kx  (GeV)
    (0.0,  5.0),    # ky  (GeV) — positive half only; mirrored at save time
    (1.0,  100.0),  # E   (GeV)
    (0.0,  50.0),   # z0  (fm)
    (0.0,  50.0),   # zf  (fm)
    (0.0,  0.9),    # u_perp
    (0.150, 0.650), # T   (GeV)
    (1.5,  2.5),    # g
]
N_DIMS = len(PARAM_RANGES)

# Integration settings (match generate_training_data.py defaults)
Q_LIM_FACTOR  = 0.3
NITN_WARMUP   = 10
NITN          = 10
NEVAL         = 10_000


# ==============================================================================
# Medium property helpers
# ==============================================================================
def compute_rho0(T: float) -> float:
    rho0 = 0.0
    for pid in SOFT_PIDS:
        rho0 += plasma_interaction.rho(T, soft_pid=pid)
    return rho0


def compute_mu(T: float, g: float) -> float:
    return plasma_interaction.mu_DeBye(T, g=g)


# ==============================================================================
# Vegas integrand
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


def integrate_point(x, kx, ky, E, T, g, u_perp, z0, zf):
    """Integrate the radiation intensity for a single parameter point."""
    q_lim = Q_LIM_FACTOR * E
    region = [(-q_lim, q_lim), (-q_lim, q_lim), (z0, zf)]

    integ    = vegas.Integrator(region)
    integrand = make_batch_integrand(x, kx, ky, E, T, g, u_perp)

    integ(integrand, nitn=NITN_WARMUP, neval=NEVAL)
    result = integ(integrand, nitn=NITN, neval=NEVAL)

    return result.mean, result.sdev


# ==============================================================================
# Worker (must be module-level for pickling)
# ==============================================================================
def _worker(task):
    """Integrate one point. Returns (idx, mean, sdev)."""
    idx, x, kx, ky, E, z0, zf, u_perp, T, g = task
    try:
        mean, sdev = integrate_point(x, kx, ky, E, T, g, u_perp, z0, zf)
        return idx, mean, sdev
    except Exception as exc:
        print(f"  Warning: integration failed at index {idx}: {exc}", flush=True)
        return idx, np.nan, np.nan


# ==============================================================================
# Sobol sampling
# ==============================================================================
def sobol_batch(n_points: int, batch_id: int) -> np.ndarray:
    """
    Draw n_points from the Sobol sequence using a scramble seed derived from
    batch_id, so that different batches cover different, non-overlapping regions
    of the parameter space.
    """
    ranges = np.array(PARAM_RANGES)
    sampler = qmc.Sobol(d=N_DIMS, scramble=True, seed=batch_id)
    # Sobol requires powers of 2; round up silently
    n_draw = int(2 ** np.ceil(np.log2(max(n_points, 2))))
    unit_pts = sampler.random(n_draw)[:n_points]
    return unit_pts * (ranges[:, 1] - ranges[:, 0]) + ranges[:, 0]


# ==============================================================================
# Main batch computation
# ==============================================================================
def run_batch(n_points: int, batch_id: int, n_workers: int, output_file: str):
    print("=" * 70, flush=True)
    print(f"Sobol batch | batch_id={batch_id} | n_points={n_points} | "
          f"workers={n_workers}", flush=True)
    print("=" * 70, flush=True)

    # --- Sample ---
    print("Sampling Sobol points...", flush=True)
    points = sobol_batch(n_points, batch_id)

    # Enforce z0 < zf (indices 4 and 5)
    swap = points[:, 4] >= points[:, 5]
    points[swap, 4], points[swap, 5] = points[swap, 5].copy(), points[swap, 4].copy()

    print(f"  {len(points)} points sampled.", flush=True)

    # --- Integrate (serially) ---
    values = np.full(n_points, np.nan)
    errors = np.full(n_points, np.nan)

    tasks = [(i, *points[i]) for i in range(n_points)]

    print(f"Integrating on {n_workers} worker(s)...", flush=True)
    t0 = time.time()
    completed = 0
    log_every = max(1, n_points // 20)

    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(_worker, task): task for task in tasks}
        for future in as_completed(futures):
            idx, mean, sdev = future.result()
            values[idx] = mean
            errors[idx] = sdev
            completed += 1
            if completed % log_every == 0:
                elapsed = time.time() - t0
                eta = elapsed / completed * (n_points - completed)
                print(f"  {completed}/{n_points} | {elapsed:.0f}s elapsed | "
                      f"~{eta:.0f}s remaining", flush=True)

    dt = time.time() - t0
    print(f"Integration complete in {dt:.1f}s ({dt / n_points:.2f}s/point)", flush=True)

    # --- Filter NaNs ---
    valid = np.isfinite(values) & np.isfinite(errors)
    n_valid = valid.sum()
    print(f"Valid points: {n_valid}/{n_points} "
          f"({100 * n_valid / n_points:.1f}%)", flush=True)

    pts_v  = points[valid]
    vals_v = values[valid]
    errs_v = errors[valid]

    # # --- Mirror ky symmetry ---
    # pts_mirror      = pts_v.copy()
    # pts_mirror[:, 2] = -pts_mirror[:, 2]   # ky → −ky
    #
    # pts_full  = np.vstack([pts_v,  pts_mirror])
    # vals_full = np.concatenate([vals_v, vals_v])
    # errs_full = np.concatenate([errs_v, errs_v])

    pts_full  = pts_v
    vals_full = vals_v
    errs_full = errs_v
    weights   = np.ones(len(vals_full))

    # --- Save ---
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_file, "w") as f:
        f.create_dataset("x",      data=pts_full[:, 0])
        f.create_dataset("kx",     data=pts_full[:, 1])
        f.create_dataset("ky",     data=pts_full[:, 2])
        f.create_dataset("E",      data=pts_full[:, 3])
        f.create_dataset("z0",     data=pts_full[:, 4])
        f.create_dataset("zf",     data=pts_full[:, 5])
        f.create_dataset("u_perp", data=pts_full[:, 6])
        f.create_dataset("T",      data=pts_full[:, 7])
        f.create_dataset("g",      data=pts_full[:, 8])
        f.create_dataset("I",      data=vals_full)
        f.create_dataset("I_err",  data=errs_full)
        f.create_dataset("weight", data=weights)

        f.attrs["batch_id"]   = batch_id
        f.attrs["n_original"] = n_valid
        f.attrs["n_samples"]  = len(vals_full)
        f.attrs["soft_pids"]  = SOFT_PIDS
        f.attrs["description"] = (
            "Sobol-sampled training data for radiation PINN. "
            "Includes original ky >= 0 samples only."
            "CF factor NOT included (multiply by 4/3 quarks, 3 gluons at runtime)."
        )

    print(f"Saved {len(vals_full)} samples ({n_valid} original) "
          f"to {output_file}", flush=True)


# ==============================================================================
# Entry point
# ==============================================================================
if __name__ == "__main__":
    import multiprocessing
    multiprocessing.set_start_method("spawn", force=True)

    parser = argparse.ArgumentParser(
        description="Generate a Sobol-sampled batch of radiation training data."
    )
    parser.add_argument("--batch-id",  type=int, default=0,
                        help="Batch index; use "
                             "$SLURM_ARRAY_TASK_ID in a job array)")
    parser.add_argument("--n-points",  type=int, default=4096,
                        help="Number of Sobol points to compute (powers of 2 recommended)")
    parser.add_argument("--n-workers", type=int, default=4,
                        help="Parallel workers for Vegas integration "
                             "(set equal to --cpus-per-task in Slurm)")
    parser.add_argument("--output-dir", type=str, default="data/batches",
                        help="Directory to write output HDF5 files into")
    args = parser.parse_args()

    output_file = str(
        Path(args.output_dir) / f"sobol_batch_{args.batch_id:04d}.h5"
    )

    run_batch(
        n_points   = args.n_points,
        batch_id   = args.batch_id,
        n_workers  = args.n_workers,
        output_file = output_file,
    )