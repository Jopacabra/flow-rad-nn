"""
test_kxky_density.py

Compares the NN emulator output against the Vegas integrator on a dense
(kx, ky) grid at fixed x and fixed medium/kinematic parameters.

Produces a 3-panel figure:
  Left:   Reference integrator I(kx, ky)
  Centre: NN emulator I(kx, ky)
  Right:  Relative residual (NN - Ref) / |Ref|

Usage:
    # Defaults (see DEFAULTS dict below)
    python test_kxky_density.py

    # Custom parameters
    python test_kxky_density.py --x 0.2 --E 50.0 --z0 0.0 \
        --u-perp 0.3 --T 0.3 --g 2.0 --n-kx 25 --n-ky 25 \
        --kx-max 4.0 --ky-max 4.0 --output kxky_comparison.png
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
from radiation_nn import RadiationEmulatorInference, kinematic_domain

# ==============================================================================
# Defaults
# ==============================================================================
HBARC = 0.197327  # GeV·fm
DEFAULTS = dict(
    x       = 0.001,
    E       = 10.0,
    z0      = 0.0,  # in inverse GeV!!!
    u_perp  = 0.3,
    mu       = 0.6,
    n_kx    = 30,
    n_ky    = 30,
    kperp_max  = None,  # Use maximum kinematically allowed kperp
    x_values = [0.2, 0.5],   # fixed x values for the x-slice plots
)
DTAU = 0.1/HBARC
DEFAULTS["zf"] = DEFAULTS["z0"] + DTAU  # dtau = 0.1 fm

# ==============================================================================
# Reference computation
# ==============================================================================
def _integrate_one(args):
    """Top-level wrapper required for ProcessPoolExecutor pickling."""
    ix, iky, x, kx, ky, E, mu, u_perp, z0, zf = args
    k_perp = np.hypot(kx, ky)
    k_phi = np.arctan2(ky, kx)
    mean, sdev = integrate_point(x, k_perp, k_phi, E, mu, u_perp, z0, zf)
    return ix, iky, mean, sdev


def compute_reference_grid(
    x, E, z0, zf, u_perp, mu,
    kx_values: np.ndarray,
    ky_values: np.ndarray,
    n_workers: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the Vegas reference on the full (kx, ky) grid in parallel.

    Returns
    -------
    I_ref : ndarray, shape (n_kx, n_ky)
    I_err : ndarray, shape (n_kx, n_ky)
    """
    n_kx = len(kx_values)
    n_ky = len(ky_values)
    n_total = n_kx * n_ky

    I_ref = np.full((n_kx, n_ky), np.nan)
    I_err = np.full((n_kx, n_ky), np.nan)

    # Build task list
    tasks = [
        (ikx, iky, x, kx_values[ikx], ky_values[iky], E, mu, u_perp, z0, zf)
        for ikx in range(n_kx)
        for iky in range(n_ky)
    ]

    print(f"  Computing {n_total} reference points on {n_workers} workers...")
    t0 = time.time()
    completed = 0

    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(_integrate_one, task): task for task in tasks}
        for future in as_completed(futures):
            ikx, iky, mean, sdev = future.result()
            I_ref[ikx, iky] = mean
            I_err[ikx, iky] = sdev
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
    kx_values: np.ndarray,
    ky_values: np.ndarray,
) -> np.ndarray:
    """
    Evaluate the NN emulator on the full (kx, ky) grid in one batched call.

    Returns
    -------
    I_nn : ndarray, shape (n_kx, n_ky)
    """
    # Evaluation grid in kx, ky
    kx_grid, ky_grid = np.meshgrid(kx_values, ky_values, indexing='ij')

    # Relabeled evaluation grid in k_perp, phi
    k_perp_grid = np.sqrt(kx_grid ** 2 + ky_grid ** 2).astype(np.float32)
    phi_grid = np.arctan2(ky_grid, kx_grid).astype(np.float32)
    n_pts = k_perp_grid.size

    I_nn_flat = emulator.predict(
        x      = np.full(n_pts, x),
        k_perp = k_perp_grid.ravel(),
        phi    = phi_grid.ravel(),
        E      = np.full(n_pts, E),
        z0     = np.full(n_pts, z0),
        u_perp = np.full(n_pts, u_perp),
        mu      = np.full(n_pts, mu),
    )
    return I_nn_flat.reshape(len(kx_values), len(ky_values))


# ==============================================================================
# Combined multi-row plot
# ==============================================================================
def make_combined_plot(
    rows: list,          # list of (params, I_ref, I_err, I_nn)
    output_file: str,
):
    """
    Plot all comparison rows in a single figure.

    Each entry in *rows* produces one row of three panels:
      [col 0] Reference integrator
      [col 1] NN emulator
      [col 2] Relative residual
    The colour scale for the reference/NN panels is shared within each row
    (based on that row's reference 98th percentile).  The residual scale is
    fixed at ±2 for every row so rows are directly comparable.

    Parameters
    ----------
    rows : list of (params dict, I_ref ndarray, I_err ndarray, I_nn ndarray)
    """
    n_rows = len(rows)
    fig, axes = plt.subplots(
        n_rows, 3,
        figsize=(16, 5 * n_rows),
        squeeze=False,
    )

    for row_idx, (params, kx_values, ky_values, I_ref, I_err, I_nn, kperp_min, kperp_max) in enumerate(rows):
        def _add_kperp_circles(ax, kperp_min, kperp_max, extent):
            """
            Overlay thin, black dashed circles of radius kperp_min and kperp_max
            (centered at the origin, kx=0, ky=0) onto ax, but only if the circle
            is actually visible within the axes' current extent.
            """
            x0, x1, y0, y1 = extent
            # Farthest distance from the origin reached anywhere in the plot box
            max_radius = np.hypot(max(abs(x0), abs(x1)), max(abs(y0), abs(y1)))
            # Closest distance from the origin to the plot box (0 if origin is inside)
            min_radius = 0.0
            if x0 > 0 or x1 < 0:
                min_radius = min(abs(x0), abs(x1))
            if y0 > 0 or y1 < 0:
                min_radius = np.hypot(min_radius, min(abs(y0), abs(y1)))

            ls = ['-', '--']  # minimum gets solid line, maximum gets dashed line
            colors = ["black", "green"]
            for i, r in enumerate([kperp_min, kperp_max]):
                if r is not None and np.isfinite(r) and min_radius <= r <= max_radius:
                    circle = plt.Circle(
                        (0, 0), r,
                        edgecolor=colors[i], facecolor='none',
                        linewidth=0.8, linestyle=ls[i], zorder=5,
                    )
                    ax.add_patch(circle)

        # Compute extent
        extent = [ky_values[0], ky_values[-1], kx_values[0], kx_values[-1]]
        print(extent)
        imshow_kwargs = dict(
            origin='lower',
            aspect='equal',
            extent=extent,
            interpolation='nearest',
        )

        # label axes
        ax_ref, ax_nn, ax_res = axes[row_idx]

        # Per-row color scale
        ref_max = np.nanpercentile(np.abs(I_ref), 98)
        vmin, vmax = -ref_max, ref_max

        with np.errstate(invalid='ignore', divide='ignore'):
            rel_residual = (I_nn - I_ref) / (np.abs(I_ref) + 1e-30)
            rel_residual[~np.isfinite(rel_residual)] = np.nan

        # --- Reference panel ---
        im0 = ax_ref.imshow(
            I_ref, cmap='RdBu_r', vmin=vmin, vmax=vmax, **imshow_kwargs
        )
        with np.errstate(invalid='ignore', divide='ignore'):
            rel_err = I_err / (np.abs(I_ref) + 1e-30)

        ax_ref.contour(
            kx_values, ky_values, rel_err,
            levels=[0.5], colors='yellow', linewidths=1.0, linestyles='--',
        )
        ax_ref.set_title('Reference (Vegas integrator)\n'
                         r'dashed = $\sigma_\mathrm{MC}/|I| > 0.5$')
        ax_ref.set_xlabel(r'$k_y$ (GeV)')
        ax_ref.set_ylabel(r'$k_x$ (GeV)')
        fig.colorbar(im0, ax=ax_ref, label=r'$I$ (no $C_F$)')

        # --- NN panel ---
        im1 = ax_nn.imshow(
            I_nn, cmap='RdBu_r', vmin=vmin, vmax=vmax, **imshow_kwargs
        )
        ax_nn.set_title('NN emulator')
        ax_nn.set_xlabel(r'$k_y$ (GeV)')
        ax_nn.set_ylabel(r'$k_x$ (GeV)')
        fig.colorbar(im1, ax=ax_nn, label=r'$I$ (no $C_F$)')

        # --- Residual panel ---
        im2 = ax_res.imshow(
            rel_residual, cmap='coolwarm', vmin=-2, vmax=2, **imshow_kwargs
        )
        ax_res.set_title(
            r'Relative residual $(I_\mathrm{NN} - I_\mathrm{ref})/|I_\mathrm{ref}|$'
        )
        ax_res.set_xlabel(r'$k_y$ (GeV)')
        ax_res.set_ylabel(r'$k_x$ (GeV)')
        fig.colorbar(im2, ax=ax_res, label='Relative residual')

        # Plot a thin, black dashed circle at kperp_min and kperp_max, if they are on the plots
        for ax in (ax_ref, ax_nn, ax_res):
            _add_kperp_circles(ax, kperp_min, kperp_max, extent)

        # Row label on the left spine
        ax_ref.set_ylabel(
            f"$x={params['x']:.3f}$, " + '\n\n' + r'$k_x$ (GeV)',
            fontsize=9,
        )

        # Console statistics
        valid = np.isfinite(I_ref) & np.isfinite(I_nn)
        if valid.sum() > 0:
            mae = np.mean(np.abs(I_nn[valid] - I_ref[valid]))
            mre = np.nanmedian(np.abs(rel_residual[valid]))
            print(f"\n  x={params['x']:.3f}  MAE={mae:.4e}  "
                  f"Median|rel|={mre:.3f} ({mre*100:.1f}%)  "
                  f"|rel|>0.5: {(np.abs(rel_residual[valid]) > 0.5).sum()}/{valid.sum()}")

    # Total parameter string label as supertitle
    param_str = (
        # f"$x={params['x']:.2f}$, "
        f"$E={params['E']:.1f}$ GeV, "
        f"$z_0={params['z0']:.1f}$ GeV^-1, "
        f"$z_f={params['zf']:.1f}$ GeV^-1, "
        f"$u_\\perp={params['u_perp']:.2f}$, "
        f"$mu={params['mu']:.3f}$ GeV, "
    )
    fig.suptitle(param_str, fontsize=9)
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n  Combined plot saved to: {output_file}")
    plt.show()


# ==============================================================================
# Main
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(
        description='Compare NN emulator vs Vegas integrator on a (kx, ky) grid.'
    )
    parser.add_argument('--E',       type=float, default=DEFAULTS['E'],      help='Parton energy (GeV)')
    parser.add_argument('--z0',      type=float, default=DEFAULTS['z0'],     help='Initial longitudinal position (invGeV)')
    parser.add_argument('--u-perp',  type=float, default=DEFAULTS['u_perp'], help='Transverse flow magnitude')
    parser.add_argument('--mu',       type=float, default=DEFAULTS['mu'],    help='Debye mass (GeV)')
    parser.add_argument('--n-kx',    type=int,   default=DEFAULTS['n_kx'],   help='Number of kx grid points')
    parser.add_argument('--n-ky',    type=int,   default=DEFAULTS['n_ky'],   help='Number of ky grid points (ky >= 0)')
    parser.add_argument('--kperp-max',  type=float, default=None, help='kperp grid range [-kx_max, kx_max] (GeV)')
    parser.add_argument('--workers', type=int,   default=4,                  help='Parallel workers for reference computation')
    parser.add_argument('--x-values', type=float, nargs='+',
                        default=DEFAULTS['x_values'],
                        help='List of fixed x values for additional comparison rows '
                             '(e.g. --x-values 0.01 0.3 0.7)')
    parser.add_argument('--model-file',
                        type=str, default='data/radiation_emulator.pt')
    parser.add_argument('--output',  type=str,   default='kxky_comparison.png')
    args = parser.parse_args()

    params = dict(E=args.E, z0=args.z0, zf=args.z0 + DTAU,
        u_perp=args.u_perp, mu=args.mu,
    )

    print("=" * 70)
    print("kx-ky DENSITY COMPARISON: NN vs Vegas")
    print("=" * 70)
    for k, v in params.items():
        print(f"  {k:8s} = {v}")
    print(f"  kx grid: {args.n_kx} points")
    print(f"  ky grid: {args.n_ky} points")
    print()

    # Load NN radiation emulator
    emulator = RadiationEmulatorInference(
        model_file=args.model_file,
        device='cpu',
    )

    # --- x-slice rows ---
    x_min, x_max, kperp_min, _, _ = kinematic_domain(0, args.E, args.mu)
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
            _, _, _, kperp_max_sq, _ = kinematic_domain(x_val, args.E, args.mu)
            kperp_max = np.sqrt(kperp_max_sq)
            print(f"kperp_min = {kperp_min:.1f} GeV, kperp_max = {kperp_max:.1f} GeV")
            x_params = dict(params, x=x_val)

            # Build grids
            if args.kperp_max is None:
                kx_values = np.linspace(-kperp_max, kperp_max, args.n_kx)
                ky_values = np.linspace(0.0, kperp_max, args.n_ky//2)
            else:
                kx_values = np.linspace(-args.kperp_max, args.kperp_max, args.n_kx)
                ky_values = np.linspace(0.0, args.kperp_max, args.n_ky//2)
            print(kx_values)
            print(ky_values)

            print("  Computing reference grid...")
            I_ref_x, I_err_x = compute_reference_grid(
                x=x_val, E=args.E, z0=args.z0, zf=args.z0 + DTAU,
                u_perp=args.u_perp, mu=args.mu,
                kx_values=kx_values,
                ky_values=ky_values,
                n_workers=args.workers,
            )

            print("  Computing NN grid...")
            t0 = time.time()
            I_nn_x = compute_nn_grid(
                emulator=emulator,
                x=x_val, E=args.E, z0=args.z0,
                u_perp=args.u_perp, mu=args.mu,
                kx_values=kx_values,
                ky_values=ky_values,
            )
            print(f"  NN Prediction max: {np.amax(I_nn_x)}")
            print(f"  NN Prediction min: {np.amin(I_nn_x)}")
            print(f"  NN Prediction mean: {np.mean(I_nn_x)}")
            print(f"  NN Prediction nanmean: {np.nanmean(I_nn_x)}")
            print(f"  NN prediction: {(time.time() - t0) * 1000:.1f} ms "
                  f"for {args.n_kx * args.n_ky} points")

            # Mirror grids
            flip_ax = 1
            I_ref_x = np.concatenate((np.flip(I_ref_x, axis=flip_ax), I_ref_x), axis=flip_ax)
            I_err_x = np.concatenate((np.flip(I_err_x, axis=flip_ax), I_err_x), axis=flip_ax)
            I_nn_x = np.concatenate((np.flip(I_nn_x, axis=flip_ax), I_nn_x), axis=flip_ax)

            # Mirror ky coordinates
            flip_ax = 0
            plot_ky = np.concatenate((-1*np.flip(ky_values, axis=flip_ax), ky_values))

            plot_rows.append((x_params, kx_values, plot_ky, I_ref_x, I_err_x, I_nn_x, kperp_min, kperp_max))

    # --- Combined plot ---
    print("\nGenerating combined comparison plot...")
    make_combined_plot(
        rows=plot_rows,
        output_file=args.output
    )


if __name__ == '__main__':
    main()