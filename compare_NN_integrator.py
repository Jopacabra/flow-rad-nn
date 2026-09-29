"""
compare_NN_integrator.py

Compares the NN emulator output against the Vegas integrator on a dense
(k_perp, k_phi) polar grid at fixed x and fixed medium/kinematic parameters.

Produces a 3-panel polar figure per x-slice:
  Left:   Reference integrator I(k_perp, k_phi)
  Centre: NN emulator I(k_perp, k_phi)
  Right:  Relative residual (NN - Ref) / |Ref|

Usage:
    # Defaults (see DEFAULTS dict below)
    python compare_NN_integrator.py

    # Custom parameters
    python compare_NN_integrator.py --x 0.2 --E 50.0 --z0 0.0 \
        --u-perp 0.3 --mu 0.3 --n-kperp 25 --n-kphi 48 \
        --kperp-max 4.0 --output kperpkphi_comparison.png
"""

import argparse
import sys
from pathlib import Path
import time
import numpy as np
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor, as_completed

# Allow running from the flow-rad-nn directory
ape_dir = str(Path(__file__).resolve().parent.parent)
sys.path.append(ape_dir)

from integration import integrate_analytic_z_brutemc_t1 as integrate_point
from integration import precompute_t234_qintegrals, compute_t234_harmonics_grid
from radiation_nn import RadiationEmulatorInference, kinematic_domain

# ==============================================================================
# Defaults
# ==============================================================================
HBARC = 0.197327  # GeV·fm
DEFAULTS = dict(
    x         = 0.001,
    E         = 10.0,
    z0        = 0.0,  # in inverse GeV!!!
    u_perp    = 0.3,
    mu        = 0.6,
    n_kperp   = 30,
    n_kphi    = 48,
    kperp_max = None,  # Use maximum kinematically allowed kperp
    x_values  = [0.2, 0.5],   # fixed x values for the x-slice plots
)
DTAU = 0.1/HBARC
DEFAULTS["zf"] = DEFAULTS["z0"] + DTAU  # dtau = 0.1 fm

# ==============================================================================
# Reference computation
# ==============================================================================
def _integrate_one(args):
    """Top-level wrapper required for ProcessPoolExecutor pickling."""
    ikperp, iphi, x, k_perp, k_phi, E, mu, u_perp, z0, zf = args
    mean, sdev = integrate_point(x, k_perp, k_phi, E, mu, u_perp, z0, zf)
    return ikperp, iphi, mean, sdev


def compute_reference_grid(
    x, E, z0, zf, u_perp, mu,
    kperp_values: np.ndarray,
    kphi_values: np.ndarray,
    n_workers: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the Vegas reference on the full (k_perp, k_phi) grid in parallel.

    Returns
    -------
    I_ref : ndarray, shape (n_kperp, n_kphi)
    I_err : ndarray, shape (n_kperp, n_kphi)
    """
    n_kperp = len(kperp_values)
    n_kphi = len(kphi_values)
    n_total = n_kperp * n_kphi

    I_ref = np.full((n_kperp, n_kphi), np.nan)
    I_err = np.full((n_kperp, n_kphi), np.nan)

    # Build task list
    tasks = [
        (ikperp, iphi, x, kperp_values[ikperp], kphi_values[iphi], E, mu, u_perp, z0, zf)
        for ikperp in range(n_kperp)
        for iphi in range(n_kphi)
    ]

    print(f"  Computing {n_total} reference points on {n_workers} workers...")
    t0 = time.time()
    completed = 0

    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(_integrate_one, task): task for task in tasks}
        for future in as_completed(futures):
            ikperp, iphi, mean, sdev = future.result()
            I_ref[ikperp, iphi] = mean
            I_err[ikperp, iphi] = sdev
            completed += 1
            if completed % max(1, n_total // 10) == 0:
                elapsed = time.time() - t0
                eta = elapsed / completed * (n_total - completed)
                print(f"    {completed}/{n_total} done "
                      f"({elapsed:.0f}s elapsed, ~{eta:.0f}s remaining)")

    dt = time.time() - t0
    print(f"  Reference grid complete in {dt:.1f}s "
          f"({dt / n_total:.2f}s per point)")
    return I_ref, I_err


# ==============================================================================
# NN prediction grid
# ==============================================================================
def compute_nn_grid(
    emulator: RadiationEmulatorInference,
    x, E, z0, u_perp, mu,
    kperp_values: np.ndarray,
    kphi_values: np.ndarray,
) -> np.ndarray:
    """
    Evaluate the NN emulator on the full (k_perp, k_phi) grid in one batched call.

    Returns
    -------
    I_nn : ndarray, shape (n_kperp, n_kphi)
    """
    kperp_grid, kphi_grid = np.meshgrid(kperp_values, kphi_values, indexing='ij')
    kperp_grid = kperp_grid.astype(np.float32)
    kphi_grid = kphi_grid.astype(np.float32)
    n_pts = kperp_grid.size

    I_nn_flat = emulator.predict(
        x      = np.full(n_pts, x),
        k_perp = kperp_grid.ravel(),
        phi    = kphi_grid.ravel(),
        E      = np.full(n_pts, E),
        z0     = np.full(n_pts, z0),
        u_perp = np.full(n_pts, u_perp),
        mu     = np.full(n_pts, mu),
    )
    return I_nn_flat.reshape(len(kperp_values), len(kphi_values))


# ==============================================================================
# Combined multi-row polar plot
# ==============================================================================
def make_combined_plot(
    rows: list,          # list of (params, kperp_values, kphi_values, I_ref, I_err, I_nn, kperp_min, kperp_max)
    output_file: str,
):
    """
    Plot all comparison rows in a single figure using polar axes.

    Each entry in *rows* produces one row of three polar panels:
      [col 0] Reference integrator
      [col 1] NN emulator
      [col 2] Relative residual
    The colour scale for the reference/NN panels is shared within each row
    (based on that row's reference 98th percentile).  The residual scale is
    fixed at ±2 for every row so rows are directly comparable.
    """
    n_rows = len(rows)
    fig, axes = plt.subplots(
        n_rows, 3,
        figsize=(16, 5.2 * n_rows),
        squeeze=False,
        subplot_kw=dict(projection='polar'),
        constrained_layout=True,
    )

    def _add_kperp_circle(ax, r, color, ls):
        if r is None or not np.isfinite(r):
            return
        theta_full = np.linspace(-np.pi, np.pi, 200)
        ax.plot(theta_full, np.full_like(theta_full, r),
                color=color, linewidth=0.8, linestyle=ls, zorder=5)

    for row_idx, (params, kperp_values, kphi_values, I_ref, I_err, I_nn, kperp_min, kperp_max) in enumerate(rows):
        ax_ref, ax_nn, ax_res = axes[row_idx]

        pcolor_kwargs = dict(shading='auto')

        # Per-row color scale
        ref_max = np.nanpercentile(np.abs(I_ref), 98)
        vmin, vmax = -ref_max, ref_max

        with np.errstate(invalid='ignore', divide='ignore'):
            rel_residual = (I_nn - I_ref) / (np.abs(I_ref) + 1e-30)
            rel_residual[~np.isfinite(rel_residual)] = np.nan
            rel_err = I_err / (np.abs(I_ref) + 1e-30)

        # --- Reference panel ---
        im0 = ax_ref.pcolormesh(
            kphi_values, kperp_values, I_ref, cmap='RdBu_r',
            vmin=vmin, vmax=vmax, **pcolor_kwargs
        )
        ax_ref.contour(
            kphi_values, kperp_values, rel_err,
            levels=[0.5], colors='yellow', linewidths=1.0, linestyles='--',
        )
        fig.colorbar(im0, ax=ax_ref, shrink=0.75)

        # --- NN panel ---
        im1 = ax_nn.pcolormesh(
            kphi_values, kperp_values, I_nn, cmap='RdBu_r',
            vmin=vmin, vmax=vmax, **pcolor_kwargs
        )
        fig.colorbar(im1, ax=ax_nn, shrink=0.75)

        # --- Residual panel ---
        im2 = ax_res.pcolormesh(
            kphi_values, kperp_values, rel_residual, cmap='coolwarm',
            vmin=-2, vmax=2, **pcolor_kwargs
        )
        fig.colorbar(im2, ax=ax_res, shrink=0.75, label='Relative residual')

        # Titles (column headers on top row only)
        if row_idx == 0:
            ax_ref.set_title('Reference (Vegas integrator)\n'
                             r'dashed = $\sigma_\mathrm{MC}/|I| > 0.5$', fontsize=9)
            ax_nn.set_title('NN emulator', fontsize=9)
            ax_res.set_title(
                r'Relative residual $(I_\mathrm{NN} - I_\mathrm{ref})/|I_\mathrm{ref}|$',
                fontsize=9,
            )

        # Row label: put x value as a title on the leftmost (reference) panel
        ax_ref.text(
            -0.25, 0.5, f"$x={params['x']:.3f}$",
            transform=ax_ref.transAxes,
            fontsize=10, rotation=90, va='center', ha='center',
        )

        # Cosmetic polar-axis formatting
        for ax in (ax_ref, ax_nn, ax_res):
            ax.set_theta_zero_location('E')
            ax.set_theta_direction(1)
            ax.set_rlabel_position(135)
            ax.tick_params(labelsize=7)
            _add_kperp_circle(ax, kperp_min, 'black', '-')
            _add_kperp_circle(ax, kperp_max, 'green', '--')
            ax.set_ylim(0, kperp_max)
            # ax.set_axis_off()

        # Console statistics
        valid = np.isfinite(I_ref) & np.isfinite(I_nn)
        if valid.sum() > 0:
            mae = np.mean(np.abs(I_nn[valid] - I_ref[valid]))
            mre = np.nanmedian(np.abs(rel_residual[valid]))
            print(f"\n  x={params['x']:.3f}  MAE={mae:.4e}  "
                  f"Median|rel|={mre:.3f} ({mre * 100:.1f}%)  "
                  f"|rel|>0.5: {(np.abs(rel_residual[valid]) > 0.5).sum()}/{valid.sum()}")

        # Total parameter string label as supertitle
    last_params = rows[-1][0]
    param_str = (
        f"$E={last_params['E']:.1f}$ GeV, "
        f"$z_0={last_params['z0']:.1f}$ GeV$^{{-1}}$, "
        f"$z_f={last_params['zf']:.1f}$ GeV$^{{-1}}$, "
        f"$u_\\perp={last_params['u_perp']:.2f}$, "
        f"$\\mu={last_params['mu']:.3f}$ GeV"
    )
    fig.suptitle(param_str, fontsize=9)
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n  Combined plot saved to: {output_file}")
    plt.show()


# ==============================================================================
# Main
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(
        description='Compare NN emulator vs Vegas integrator on a (k_perp, k_phi) polar grid.'
    )
    parser.add_argument('--E', type=float, default=DEFAULTS['E'], help='Parton energy (GeV)')
    parser.add_argument('--z0', type=float, default=DEFAULTS['z0'], help='Initial longitudinal position (invGeV)')
    parser.add_argument('--u-perp', type=float, default=DEFAULTS['u_perp'], help='Transverse flow magnitude')
    parser.add_argument('--mu', type=float, default=DEFAULTS['mu'], help='Debye mass (GeV)')
    parser.add_argument('--n-kperp', type=int, default=DEFAULTS['n_kperp'],
                        help='Number of k_perp (radial) grid points')
    parser.add_argument('--n-kphi', type=int, default=DEFAULTS['n_kphi'],
                        help='Number of k_phi (angular) grid points, spanning [-pi, pi)')
    parser.add_argument('--kperp-max', type=float, default=None,
                        help='Override kperp grid upper bound (GeV); default uses the kinematically allowed maximum')
    parser.add_argument('--kperp-min', type=float, default=None,
                        help='Override kperp grid lower bound (GeV); default uses the kinematically allowed minimum')
    parser.add_argument('--workers', type=int, default=4, help='Parallel workers for reference computation')
    parser.add_argument('--x-values', type=float, nargs='+',
                        default=DEFAULTS['x_values'],
                        help='List of fixed x values for additional comparison rows '
                             '(e.g. --x-values 0.01 0.3 0.7)')
    parser.add_argument('--model-file',
                        type=str, default='data/radiation_emulator.pt')
    parser.add_argument('--output', type=str, default='kperp_kphi_comparison.png')
    parser.add_argument("--full", action="store_true", help="Compute complete spectra, including t2, t3, & t4")
    args = parser.parse_args()

    params = dict(E=args.E, z0=args.z0, zf=args.z0 + DTAU,
                  u_perp=args.u_perp, mu=args.mu,
                  )

    print("=" * 70)
    print("k_perp-k_phi DENSITY COMPARISON: NN vs Vegas")
    print("=" * 70)
    for k, v in params.items():
        print(f"  {k:8s} = {v}")
    print(f"  k_perp grid: {args.n_kperp} points")
    print(f"  k_phi  grid: {args.n_kphi} points")
    print()

    # Load NN radiation emulator
    emulator = RadiationEmulatorInference(
        model_file=args.model_file,
        device='cpu',
    )

    # --- x-slice rows ---
    # Only used to establish the global x-domain bounds for the eps-padded endpoints
    x_min, x_max, _, _, _ = kinematic_domain(0, args.E, args.mu)
    eps = 0.1
    args.x_values.insert(0, x_min + eps)  # add minimum x
    args.x_values.append(x_max - eps)  # add maximum x
    plot_rows = []
    if args.x_values:
        print("\n" + "=" * 70)
        print(f"x-SLICE COMPARISONS: {args.x_values}")
        print("=" * 70)
        for x_val in args.x_values:
            print(f"\n--- x = {x_val} ---")
            # Recompute BOTH bounds at this x_val -- kperp_min depends on x too
            _, _, kperp_min, kperp_max_sq, _ = kinematic_domain(x_val, args.E, args.mu)
            kperp_max = np.sqrt(kperp_max_sq)
            print(f"kperp_min = {kperp_min:.3f} GeV, kperp_max = {kperp_max:.3f} GeV")
            x_params = dict(params, x=x_val)

            # Build polar grids: radial in [kperp_min, kperp_max], angular over full circle
            r_max = args.kperp_max if args.kperp_max is not None else kperp_max
            r_min = args.kperp_min if args.kperp_min is not None else kperp_min
            kperp_values = np.linspace(r_min, r_max, args.n_kperp)
            kphi_values = np.linspace(-np.pi, np.pi, args.n_kphi, endpoint=False)

            print("  Computing reference grid...")
            I_ref_x, I_err_x = compute_reference_grid(
                x=x_val, E=args.E, z0=args.z0, zf=args.z0 + DTAU,
                u_perp=args.u_perp, mu=args.mu,
                kperp_values=kperp_values,
                kphi_values=kphi_values,
                n_workers=args.workers,
            )

            print("  Computing NN grid...")
            t0 = time.time()
            I_nn_x = compute_nn_grid(
                emulator=emulator,
                x=x_val, E=args.E, z0=args.z0,
                u_perp=args.u_perp, mu=args.mu,
                kperp_values=kperp_values,
                kphi_values=kphi_values,
            )
            print(f"  NN Prediction max: {np.amax(I_nn_x)}")
            print(f"  NN Prediction min: {np.amin(I_nn_x)}")
            print(f"  NN Prediction mean: {np.mean(I_nn_x)}")
            print(f"  NN Prediction nanmean: {np.nanmean(I_nn_x)}")
            print(f"  NN prediction: {(time.time() - t0) * 1000:.1f} ms "
                  f"for {args.n_kperp * args.n_kphi} points")

            # If full grids are desired, add grids computed for t2, t3, and t4 to NN and ref.
            if args.full:
                # Native polar evaluation grid
                Kperp_grid, Phi_grid = np.meshgrid(kperp_values, kphi_values, indexing='ij')

                # Precompute qintegrals
                q_max = np.sqrt(3 * args.E * args.mu)
                qints = precompute_t234_qintegrals(args.E, args.mu, args.u_perp, q_max)

                # Batched numerical integrator call for the remaining points, summing to collapse x axis
                A0_234, A1_234, A2_234 = compute_t234_harmonics_grid(
                    [x_val], Kperp_grid, args.E, args.mu, args.u_perp, args.z0, args.z0 + DTAU, qints=qints
                )
                A0_234, A1_234, A2_234 = np.sum(A0_234, axis=0), np.sum(A1_234, axis=0), np.sum(A2_234, axis=0)

                # Reconstruct full angular dependence via broadcasting -- shape (n_kperp, n_kphi)
                cos_phi = np.cos(Phi_grid)
                cos_2phi = np.cos(2 * Phi_grid)
                I_t234 = A0_234 + A1_234 * cos_phi + A2_234 * cos_2phi

                # Sum with the t1 grid
                I_ref_x = I_ref_x + I_t234
                I_nn_x = I_nn_x + I_t234

            plot_rows.append((x_params, kperp_values, kphi_values, I_ref_x, I_err_x, I_nn_x, kperp_min, kperp_max))

    # --- Combined plot ---
    print("\nGenerating combined comparison plot...")
    make_combined_plot(
        rows=plot_rows,
        output_file=args.output
    )


if __name__ == '__main__':
    main()