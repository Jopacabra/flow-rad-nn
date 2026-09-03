"""
generate_sobol_batch.py

Generates a single batch of Sobol-sampled training data for the radiation PINN
emulator. Designed to be run as a Slurm job array, where each array task writes
an independent HDF5 file. Merge outputs afterwards with merge.py.

Usage (standalone):
    python generate_sobol_batch.py --batch-id 0 --n-points 4096 --n-workers 8

Usage (Slurm job array):
    sbatch submit_sobol.sh
"""

import argparse
import time
import numpy as np
import h5py
from pathlib import Path
from scipy.stats import qmc
from concurrent.futures import ProcessPoolExecutor, as_completed
from integration import integrate_analytic_z_t234_brutemc_t1 as integrate_point
from integration import SOFT_PIDS


# ==============================================================================
# Constants
# ==============================================================================
HBARC = 0.197327  # GeV·fm


# ==============================================================================
# Parameter space
# ==============================================================================
# Order: [x, kx, ky, E, z0, u_perp, T, g]
# zf is hardcoded as z0 + 0.1/HBARC
PARAM_RANGES = [
    (0.01, 0.99),   # x
    (0, 5.0),    # k_perp  (GeV)
    (0.0,  2*np.pi),    # k_phi  (rad)
    (1.0,  100.0),  # E   (GeV)
    (0.0,  50.0),   # z0  (invGeV)  # Up to 10 fmish
    (0.0,  0.99),   # u_perp
    (0.3, 1.2), # mu   (GeV)
]
N_DIMS = len(PARAM_RANGES)

# Fixed proper-time step in GeV (dtau = 0.1 fm)
DTAU_GEV = 0.1 / HBARC


# ==============================================================================
# Worker (must be module-level for pickling)
# ==============================================================================
def _worker(task):
    """Integrate one point. Returns (idx, mean, sdev)."""
    idx, x, k_perp, k_phi, E, z0, u_perp, mu = task
    zf = z0 + DTAU_GEV
    try:
        mean, sdev = integrate_point(x, k_perp, k_phi, E, mu, u_perp, z0, zf)
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
    # points columns: [x, kx, ky, E, z0, u_perp, T, g]
    # zf is NOT sampled; it is derived from z0 at integration time

    print(f"  {len(points)} points sampled.", flush=True)

    # --- Integrate ---
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

    pts_full  = pts_v
    vals_full = vals_v
    errs_full = errs_v
    weights   = np.ones(len(vals_full))

    # Derive zf column for storage (z0 is column index 4)
    zf_full = pts_full[:, 4] + DTAU_GEV

    # --- Save ---
    # Columns: [x, kx, ky, E, z0, u_perp, T, g]
    #           0   1   2  3   4    5     6  7
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_file, "w") as f:
        f.create_dataset("x",      data=pts_full[:, 0])
        f.create_dataset("k_perp",     data=pts_full[:, 1])
        f.create_dataset("k_phi",     data=pts_full[:, 2])
        f.create_dataset("E",      data=pts_full[:, 3])
        f.create_dataset("z0",     data=pts_full[:, 4])
        f.create_dataset("zf", data=zf_full)  # derived, stored for reference
        f.create_dataset("u_perp", data=pts_full[:, 5])
        f.create_dataset("mu",      data=pts_full[:, 6])
        f.create_dataset("I",      data=vals_full)
        f.create_dataset("I_err",  data=errs_full)
        f.create_dataset("weight", data=weights)

        f.attrs["batch_id"]   = batch_id
        f.attrs["n_original"] = n_valid
        f.attrs["n_samples"]  = len(vals_full)
        f.attrs["soft_pids"]  = SOFT_PIDS
        f.attrs["HBARC"]      = HBARC
        f.attrs["dtau_fm"]    = 0.1
        f.attrs["description"] = (
            "Sobol-sampled training data for radiation PINN. "
            "zf is hardcoded as z0 + 0.1/HBARC (dtau=0.1 fm) and stored for reference only. "
            "Includes original ky >= 0 samples only. "
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
        n_points    = args.n_points,
        batch_id    = args.batch_id,
        n_workers   = args.n_workers,
        output_file = output_file,
    )