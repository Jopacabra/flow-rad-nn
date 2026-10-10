"""
generate_sobol_batch.py

Generates a single batch of Sobol-sampled training data for the radiation PINN.
Designed to be run as a Slurm job array, where each array task writes
an independent HDF5 file. Merge outputs afterwards with merge.py.

Sampling strategy
------------------
mu, E, z0, u_perp are independent. x and k_perp are kinematically bounded:
given the Sobol unit-cube point, mu and E are mapped first, then x is mapped
log-uniformly within its (mu, E)-dependent window [mu/E, 1-mu/E], then k_perp
is mapped log-uniformly within its (x, E, mu)-dependent window
[mu, kperp_max(x,E,mu)] using the exact same kinematic_domain() used by
training/deployment (radiation_nn.py). This guarantees every generated
point is kinematically valid by construction, instead of relying on a
fixed rectangular box in k_perp (or x) that we have to trim down to the
valid region later.

This is a Rosenblatt-type conditional transform of the Sobol sequence: it
preserves Sobol's low-discrepancy structure in the "upstream" dimensions
(mu, E) while guaranteeing validity in the "downstream" ones (x, k_perp).
It's a mild degradation of equidistribution relative to pure independent
Sobol, but vastly better than rejection sampling (which wastes a growing
fraction of draws as the box/domain mismatch grows) and than a fixed-box
approach.

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
from integration import integrate_analytic_z_brutemc_t1 as integrate_point
from radiation_nn import kinematic_domain  # single source of truth for the valid (x, k_perp) domain


#############
# Constants #
#############
HBARC = 0.197327  # GeV·fm

# Fixed proper-time step in GeV (dtau = 0.1 fm)
DTAU_GEV = 0.1 / HBARC


###################
# Parameter space #
###################
# Independent (unconditioned) ranges
MU_RANGE     = (0.3, 1.4)     # mu    (GeV)
Z0_RANGE     = (0.0, 50.0)    # z0    (1/GeV)
UPERP_RANGE  = (0.0, 0.999)    # u_perp (unitless)

# E is log-uniform, but its *lower* bound is lifted above mu by a safety
# margin so that x_min = mu/E never gets uncomfortably close to x_max = 1-mu/E.
# (Guards against a vanishingly small/degenerate x window at low E, high mu.)
E_ABS_RANGE     = (1.0, 150.0)       # absolute floor/ceiling on E (GeV)
E_OVER_MU_MARGIN = 2.0 * np.sqrt(2)  # Our bounds => for there to be any valid region, E >= 2 * sqrt(2) * mu
E_BOUNDARY_EPS   = 1e-6              # tiny relative pad so we sample arbitrarily close to, but never exactly on,
                                     # the zero-measure E=2*mu edge (avoids float degeneracy in kperp window)

# x and k_perp are not given fixed ranges here -- they are derived
# per-point from (mu, E) and (x, mu, E) respectively, log-uniformly within
# the exact kinematic window. See `_conditional_sample` below.

# Small inset (in log-space, as a fraction of the window's log-width) to
# keep samples off the exact kinematic boundary -- avoids feeding the
# integrator points arbitrarily close to a vanishing/singular edge.
LOG_EDGE_INSET = 1e-3

N_DIMS = 6  # [mu, E, x, k_perp, z0, u_perp] unit-cube dims, in generation order


##############################################
# Worker (must be module-level for pickling) #
##############################################
def _worker(task):
    """Integrate one point. Returns (idx, mean, sdev)."""
    idx, x, k_perp, E, z0, u_perp, mu = task
    zf = z0 + DTAU_GEV
    try:
        A0A1mean, A0A1sdev = integrate_point(x, k_perp, 0, E, mu, u_perp, z0, zf)
        A0mean, A0sdev = integrate_point(x, k_perp, np.pi/2, E, mu, u_perp, z0, zf)
        A1mean = A0A1mean - A0mean
        A1sdev = np.sqrt(A0A1sdev**2 + A0sdev**2)
        return idx, A0mean, A0sdev, A1mean, A1sdev, A0A1sdev
    except Exception as exc:
        print(f"  Warning: integration failed at index {idx}: {exc}", flush=True)
        return idx, np.nan, np.nan, np.nan, np.nan, np.nan


##########################################################
# Sobol sampling with conditional (Rosenblatt) transform #
##########################################################
def _log_uniform(u, lo, hi):
    """Map u in [0,1] to log-uniform value in [lo, hi], elementwise."""
    log_lo = np.log(lo)
    log_hi = np.log(hi)
    return np.exp(log_lo + u * (log_hi - log_lo))


def _inset_log_window(lo, hi, frac=LOG_EDGE_INSET):
    """Shrink a (lo, hi) window symmetrically in log-space to avoid exact
    boundary samples."""
    log_lo = np.log(lo)
    log_hi = np.log(hi)
    width = log_hi - log_lo
    return np.exp(log_lo + frac * width), np.exp(log_hi - frac * width)


def sobol_batch(n_points: int, batch_id: int) -> np.ndarray:
    """
    Draw n_points from the Sobol sequence using a scramble seed derived from
    batch_id, then apply the kinematic conditional transform.

    Returns array with columns [x, k_perp, E, z0, u_perp, mu] (matches the
    order expected by `_worker` / downstream HDF5 layout). Only rows with a
    genuinely non-empty kinematic window are returned -- the returned array
    may have fewer than n_points rows.
    """
    sampler = qmc.Sobol(d=N_DIMS, scramble=True, seed=batch_id)
    n_draw = int(2 ** np.ceil(np.log2(max(n_points, 2))))
    u = sampler.random(n_draw)[:n_points]  # (n_points, 6)

    # --- mu: independent, linear-uniform ---
    mu = MU_RANGE[0] + u[:, 0] * (MU_RANGE[1] - MU_RANGE[0])

    # --- E: log-uniform. Lower bound is the EXACT kinematic boundary
    #     E = 2*sqrt(2)*mu (below which NO x gives a non-empty k_perp
    #     window at all -- see kinematic_domain), nudged up by a tiny
    #     relative epsilon to avoid landing exactly on the zero-measure
    #     edge. ---
    E_lo = np.maximum(E_ABS_RANGE[0], E_OVER_MU_MARGIN * mu * (1.0 + E_BOUNDARY_EPS))
    E_hi = np.maximum(np.full_like(E_lo, E_ABS_RANGE[1]), E_lo * 1.001)
    E = _log_uniform(u[:, 1], E_lo, E_hi)

    # --- x: log-uniform within the EXACT kinematic window. Pull x_min/x_max
    #     straight from kinematic_domain rather than re-deriving them here,
    #     so this can never drift out of sync with the formula used by
    #     training (_scan_file) and deployment masking again. ---
    x_min, x_max, _, _, _ = kinematic_domain(np.full_like(E, 0.5), E, mu)
    # (x argument above is a dummy -- x_min/x_max don't depend on x itself)
    x_lo, x_hi = _inset_log_window(x_min, x_max)
    x = _log_uniform(u[:, 2], x_lo, x_hi)

    # --- k_perp: log-uniform within the exact kinematic window ---
    _, _, kperp_min, kperp_max_sq, valid = kinematic_domain(x, E, mu)
    kperp_max = np.sqrt(np.clip(kperp_max_sq, a_min=(kperp_min ** 2) * 1.0001, a_max=None))
    kp_lo, kp_hi = _inset_log_window(kperp_min, kperp_max)
    k_perp = _log_uniform(u[:, 3], kp_lo, kp_hi)

    # --- z0, u_perp: independent, linear-uniform (unchanged) ---
    z0 = Z0_RANGE[0] + u[:, 4] * (Z0_RANGE[1] - Z0_RANGE[0])
    u_perp = UPERP_RANGE[0] + u[:, 5] * (UPERP_RANGE[1] - UPERP_RANGE[0])

    points = np.column_stack([x, k_perp, E, z0, u_perp, mu])

    # Defense-in-depth: with E_OVER_MU_MARGIN now set to the TRUE boundary
    # (2*sqrt(2)*mu) this should be ~0 points. A nonzero count here would
    # indicate the margin/eps need revisiting again, or a mismatch between
    # this file's logic and kinematic_domain().
    n_bad = int((~valid).sum())
    if n_bad > 0:
        frac = n_bad / n_points
        print(f"  Dropping {n_bad}/{n_points} ({100*frac:.2f}%) points that landed "
              f"in a zero-measure/degenerate kinematic window. If this fraction "
              f"is more than a rounding-level sliver, check E_OVER_MU_MARGIN / "
              f"E_BOUNDARY_EPS against kinematic_domain().", flush=True)
        points = points[valid]

    return points


##########################
# Main batch computation #
##########################
def run_batch(n_points: int, batch_id: int, n_workers: int, output_file: str):
    print("=" * 70, flush=True)
    print(f"Sobol batch | batch_id={batch_id} | n_points={n_points} | "
          f"workers={n_workers}", flush=True)
    print("=" * 70, flush=True)

    # Sample
    print("Sampling Sobol points (kinematically conditioned)...", flush=True)
    points = sobol_batch(n_points, batch_id)
    n_sampled = len(points)  # may be < n_points if any degenerate rows were dropped
    print(f"  {n_sampled} points sampled.", flush=True)

    # Integrate
    A0values = np.full(n_sampled, np.nan)
    A0errors = np.full(n_sampled, np.nan)
    A1values = np.full(n_sampled, np.nan)
    A1errors = np.full(n_sampled, np.nan)
    A0A1errors = np.full(n_sampled, np.nan)

    tasks = [(i, *points[i]) for i in range(n_sampled)]

    print(f"Integrating on {n_workers} worker(s)...", flush=True)
    t0 = time.time()
    completed = 0
    log_every = max(1, n_sampled // 20)

    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(_worker, task): task for task in tasks}
        for future in as_completed(futures):
            idx, A0mean, A0sdev, A1mean, A1sdev, A0A1sdev = future.result()
            A0values[idx] = A0mean
            A0errors[idx] = A0sdev
            A1values[idx] = A1mean
            A1errors[idx] = A1sdev
            A0A1errors[idx] = A0A1sdev
            completed += 1
            if completed % log_every == 0:
                elapsed = time.time() - t0
                eta = elapsed / completed * (n_sampled - completed)
                print(f"  {completed}/{n_sampled} | {elapsed:.0f}s elapsed | "
                      f"~{eta:.0f}s remaining", flush=True)

    dt = time.time() - t0
    print(f"Integration complete in {dt:.1f}s ({dt / n_sampled:.2f}s/point)", flush=True)

    # Filter NaNs (unchanged logic, just against n_sampled now)
    valid = (np.isfinite(A0values) & np.isfinite(A0errors) & np.isfinite(A1values) & np.isfinite(A1errors)
             & np.isfinite(A0A1errors))
    n_valid = valid.sum()
    print(f"Valid points: {n_valid}/{n_sampled} "
          f"({100 * n_valid / n_sampled:.1f}%)", flush=True)

    pts_full      = points[valid]
    A0vals_full   = A0values[valid]
    A0errs_full   = A0errors[valid]
    A1vals_full   = A1values[valid]
    A1errs_full   = A1errors[valid]
    A0A1errs_full = A0A1errors[valid]
    weights       = np.ones(len(pts_full))

    # Columns: [x, k_perp, E, z0, u_perp, mu]
    #           0    1     2   3    4     5
    zf_full = pts_full[:, 3] + DTAU_GEV   # fixed: z0 is column 3, not 4

    # Save
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_file, "w") as f:
        f.create_dataset("x",      data=pts_full[:, 0])
        f.create_dataset("k_perp", data=pts_full[:, 1])
        f.create_dataset("E",      data=pts_full[:, 2])
        f.create_dataset("z0",     data=pts_full[:, 3])
        f.create_dataset("zf",     data=zf_full)
        f.create_dataset("u_perp", data=pts_full[:, 4])
        f.create_dataset("mu",     data=pts_full[:, 5])
        f.create_dataset("A0",      data=A0vals_full)
        f.create_dataset("A0_err",  data=A0errs_full)
        f.create_dataset("A1",      data=A1vals_full)
        f.create_dataset("A1_err",  data=A1errs_full)
        f.create_dataset("A0A1_err", data=A0A1errs_full)
        f.create_dataset("weight", data=weights)

        f.attrs["batch_id"]   = batch_id
        f.attrs["n_requested"] = n_points
        f.attrs["n_sampled"]   = n_sampled
        f.attrs["n_original"] = n_valid
        f.attrs["n_samples"]  = len(pts_full)
        f.attrs["HBARC"]      = HBARC
        f.attrs["dtau_fm"]    = 0.1
        f.attrs["description"] = (
            "Sobol-sampled training data for radiation PINN, with x and k_perp "
            "drawn log-uniformly WITHIN THE EXACT KINEMATIC WINDOW conditioned "
            "on each point's (E, mu) -- not a fixed rectangular box. E is sampled "
            "down to the exact x_min=x_max boundary (E=2*mu). "
            "zf is hardcoded as z0 + 0.1/HBARC (dtau=0.1 fm) and stored for reference only. "
            "CF factor NOT included (multiply by 4/3 quarks, 3 gluons at runtime)."
            "A1 is derived via difference -- A1 error is quadratic sum of error on A0 and error on A0+A1."
        )

    print(f"Saved {len(pts_full)} samples ({n_valid} original) "
          f"to {output_file}", flush=True)


###############
# Entry point #
###############
if __name__ == "__main__":
    import multiprocessing
    multiprocessing.set_start_method("spawn", force=True)

    parser = argparse.ArgumentParser(
        description="Generate a Sobol-sampled batch of radiation training data."
    )
    parser.add_argument("--batch-id",  type=int, default=0,
                        help="Batch index; use "
                             "$SLURM_ARRAY_TASK_ID in a job array)")
    parser.add_argument("--n-points",  type=int, default=256,
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