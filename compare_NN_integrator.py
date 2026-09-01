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

from integration import integrate_point
from train_radiation_nn import RadiationEmulatorInference

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
    kx_max  = 4.0,
    ky_max  = 4.0,
    x_values = [0.01, 0.1],   # fixed x values for the x-slice plots
)
DTAU = 0.1/HBARC
DEFAULTS["zf"] = DEFAULTS["z0"] + DTAU  # dtau = 0.1 fm

# ==============================================================================
# Reference computation
# ==============================================================================
def _integrate_one(args):
    """Top-level wrapper required for ProcessPoolExecutor pickling."""
    ix, iky, x, kx, ky, E, T, g, u_perp, z0, zf = args
    mean, sdev = integrate_point(x, kx, ky, E, T, g, u_perp, z0, zf)
    return ix, iky, mean, sdev


def compute_reference_grid(
    x, E, z0, zf, u_perp, T, g,
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
    print("!" * 70)
    print("WARNING -- MISMATCHED MU DEFINITION")
    print("!" * 70)
    n_kx = len(kx_values)
    n_ky = len(ky_values)
    n_total = n_kx * n_ky

    I_ref = np.full((n_kx, n_ky), np.nan)
    I_err = np.full((n_kx, n_ky), np.nan)

    # Build task list
    tasks = [
        (ikx, iky, x, kx_values[ikx], ky_values[iky], E, T, g, u_perp, z0, zf)
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
# Plotting
# ==============================================================================
def make_comparison_plot(
    kx_values, ky_values,
    I_ref, I_err,
    I_nn,
    params: dict,
    output_file: str,
):
    """
    Three-panel density plot:
      [0] Reference integrator
      [1] NN emulator
      [2] Relative residual (NN - Ref) / |Ref|
    """
    # Symmetric colour scale based on the reference, ignoring NaNs
    ref_max = np.nanpercentile(np.abs(I_ref), 98)
    vmin, vmax = -ref_max, ref_max

    # Relative residual — guard against near-zero reference values
    with np.errstate(invalid='ignore', divide='ignore'):
        rel_residual = (I_nn - I_ref) / (np.abs(I_ref) + 1e-30)
        rel_residual[~np.isfinite(rel_residual)] = np.nan

    res_abs = np.nanpercentile(np.abs(rel_residual), 98)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    extent = [ky_values[0], ky_values[-1], kx_values[0], kx_values[-1]]
    imshow_kwargs = dict(
        origin='lower',
        aspect='auto',
        extent=extent,
        interpolation='nearest',
    )

    # --- Panel 0: Reference ---
    im0 = axes[0].imshow(
        I_ref, cmap='RdBu_r', vmin=vmin, vmax=vmax, **imshow_kwargs
    )
    axes[0].set_title('Reference (Vegas integrator)')
    axes[0].set_xlabel(r'$k_y$ (GeV)')
    axes[0].set_ylabel(r'$k_x$ (GeV)')
    fig.colorbar(im0, ax=axes[0], label=r'$I$ (no $C_F$)')

    # Overlay integration error as contour where I_err / |I_ref| > 0.5
    # so you can see where the reference itself is unreliable
    with np.errstate(invalid='ignore', divide='ignore'):
        rel_err = I_err / (np.abs(I_ref) + 1e-30)
    axes[0].contour(
        ky_values, kx_values, rel_err,
        levels=[0.5], colors='yellow', linewidths=1.0, linestyles='--',
    )
    axes[0].set_title('Reference (Vegas integrator)'
                      r'dashed = $\sigma_\mathrm{MC}/|I| > 0.5$')

    # --- Panel 1: NN ---
    im1 = axes[1].imshow(
        I_nn, cmap='RdBu_r', vmin=vmin, vmax=vmax, **imshow_kwargs
    )
    axes[1].set_title('NN emulator')
    axes[1].set_xlabel(r'$k_y$ (GeV)')
    axes[1].set_ylabel(r'$k_x$ (GeV)')
    fig.colorbar(im1, ax=axes[1], label=r'$I$ (no $C_F$)')

    # --- Panel 2: Relative residual ---
    # im2 = axes[2].imshow(
    #     rel_residual, cmap='coolwarm', vmin=-res_abs, vmax=res_abs, **imshow_kwargs
    # )
    im2 = axes[2].imshow(
        rel_residual, cmap='coolwarm', vmin=-2, vmax=2, **imshow_kwargs
    )
    axes[2].set_title(r'Relative residual $(I_\mathrm{NN} - I_\mathrm{ref})/|I_\mathrm{ref}|$')
    axes[2].set_xlabel(r'$k_y$ (GeV)')
    axes[2].set_ylabel(r'$k_x$ (GeV)')
    fig.colorbar(im2, ax=axes[2], label='Relative residual')

    # Shared title with parameter values
    param_str = (
        f"$x={params['x']:.2f}$, "
        f"$E={params['E']:.1f}$ GeV, "
        f"$z_0={params['z0']:.1f}$ GeV^-1, "
        f"$z_f={params['zf']:.1f}$ GeV^-1, "
        f"$u_\\perp={params['u_perp']:.2f}$, "
        f"$mu={params['mu']:.3f}$ GeV, "
    )
    fig.suptitle(param_str, fontsize=11, y=1.01)

    # Summary statistics in console
    valid = np.isfinite(I_ref) & np.isfinite(I_nn)
    if valid.sum() > 0:
        mae = np.mean(np.abs(I_nn[valid] - I_ref[valid]))
        mre = np.nanmedian(np.abs(rel_residual[valid]))
        print(f"\n  MAE:            {mae:.4e}")
        print(f"  Median |rel|:   {mre:.3f}  ({mre*100:.1f}%)")
        print(f"  Points with |rel| > 0.5:  "
              f"{(np.abs(rel_residual[valid]) > 0.5).sum()} / {valid.sum()}")

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n  Plot saved to: {output_file}")
    plt.show()


# ==============================================================================
# Combined multi-row plot
# ==============================================================================
def make_combined_plot(
    kx_values, ky_values,
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

    extent = [ky_values[0], ky_values[-1], kx_values[0], kx_values[-1]]
    imshow_kwargs = dict(
        origin='lower',
        aspect='auto',
        extent=extent,
        interpolation='nearest',
    )

    for row_idx, (params, I_ref, I_err, I_nn) in enumerate(rows):
        ax_ref, ax_nn, ax_res = axes[row_idx]

        # Per-row colour scale
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
            ky_values, kx_values, rel_err,
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

        # Row label on the left spine
        ax_ref.set_ylabel(
            f"$x={params['x']:.2f}$, " + '\n\n' + r'$k_x$ (GeV)',
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
    parser.add_argument('--x',       type=float, default=DEFAULTS['x'],      help='Momentum fraction x')
    parser.add_argument('--E',       type=float, default=DEFAULTS['E'],      help='Parton energy (GeV)')
    parser.add_argument('--z0',      type=float, default=DEFAULTS['z0'],     help='Initial longitudinal position (invGeV)')
    parser.add_argument('--u-perp',  type=float, default=DEFAULTS['u_perp'], help='Transverse flow magnitude')
    parser.add_argument('--mu',       type=float, default=DEFAULTS['mu'],    help='Debye mass (GeV)')
    parser.add_argument('--n-kx',    type=int,   default=DEFAULTS['n_kx'],   help='Number of kx grid points')
    parser.add_argument('--n-ky',    type=int,   default=DEFAULTS['n_ky'],   help='Number of ky grid points (ky >= 0)')
    parser.add_argument('--kx-max',  type=float, default=DEFAULTS['kx_max'], help='kx grid range [-kx_max, kx_max] (GeV)')
    parser.add_argument('--ky-max',  type=float, default=DEFAULTS['ky_max'], help='ky grid range [0, ky_max] (GeV)')
    parser.add_argument('--workers', type=int,   default=4,                  help='Parallel workers for reference computation')
    parser.add_argument('--x-values', type=float, nargs='+',
                        default=DEFAULTS['x_values'],
                        help='List of fixed x values for additional comparison rows '
                             '(e.g. --x-values 0.01 0.3 0.7)')
    parser.add_argument('--model-file',
                        type=str, default='data/radiation_emulator.pt')
    parser.add_argument('--normalization-file',
                        type=str, default='data/radiation_normalization.json')
    parser.add_argument('--output',  type=str,   default='kxky_comparison.png')
    args = parser.parse_args()

    params = dict(
        x=args.x, E=args.E, z0=args.z0, zf=args.z0 + DTAU,
        u_perp=args.u_perp, mu=args.mu,
    )

    print("=" * 70)
    print("kx-ky DENSITY COMPARISON: NN vs Vegas")
    print("=" * 70)
    for k, v in params.items():
        print(f"  {k:8s} = {v}")
    print(f"  kx grid: {args.n_kx} points in [{-args.kx_max:.1f}, {args.kx_max:.1f}] GeV")
    print(f"  ky grid: {args.n_ky} points in [0, {args.ky_max:.1f}] GeV")
    print()

    # Build grids
    # ky >= 0 only (training data is for ky >= 0; mirroring is done at save time)
    kx_values = np.linspace(-args.kx_max, args.kx_max, args.n_kx)
    ky_values = np.linspace(0.0, args.ky_max, args.n_ky)

    # --- Reference ---
    print("Step 1: Computing reference (Vegas integrator)")
    I_ref, I_err = compute_reference_grid(
        x=args.x, E=args.E, z0=args.z0, zf=args.z0 + DTAU,
        u_perp=args.u_perp, T=(args.mu/np.sqrt(1 + 2/6))/2, g=2,
        kx_values=kx_values,
        ky_values=ky_values,
        n_workers=args.workers,
    )

    # --- NN ---
    print("\nStep 2: Loading NN emulator and predicting")
    emulator = RadiationEmulatorInference(
        model_file=args.model_file,
        normalization_file=args.normalization_file,
        device='cpu',
    )
    t0 = time.time()
    I_nn = compute_nn_grid(
        emulator=emulator,
        x=args.x, E=args.E, z0=args.z0,
        u_perp=args.u_perp, mu=args.mu,
        kx_values=kx_values,
        ky_values=ky_values,
    )
    print(f"  NN Prediction max: {np.amax(I_nn)}")
    print(f"  NN Prediction min: {np.amin(I_nn)}")
    print(f"  NN Prediction mean: {np.mean(I_nn)}")
    print(f"  NN Prediction nanmean: {np.nanmean(I_nn)}")
    print(f"  NN prediction: {(time.time() - t0) * 1000:.1f} ms "
          f"for {args.n_kx * args.n_ky} points")

    # Accumulate rows: start with the primary (--x) row
    plot_rows = [(params, I_ref, I_err, I_nn)]

    # --- x-slice rows ---
    if args.x_values:
        print("\n" + "=" * 70)
        print(f"x-SLICE COMPARISONS: {args.x_values}")
        print("=" * 70)
        for x_val in args.x_values:
            print(f"\n--- x = {x_val} ---")
            x_params = dict(params, x=x_val)

            print("  Computing reference grid...")
            I_ref_x, I_err_x = compute_reference_grid(
                x=x_val, E=args.E, z0=args.z0, zf=args.z0 + DTAU,
                u_perp=args.u_perp, T=(args.mu**2 - 2/6)/2, g=2,
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
            print(f"  NN prediction: {(time.time() - t0) * 1000:.1f} ms")

            plot_rows.append((x_params, I_ref_x, I_err_x, I_nn_x))

    # --- Combined plot ---
    print("\nGenerating combined comparison plot...")
    make_combined_plot(
        kx_values=kx_values,
        ky_values=ky_values,
        rows=plot_rows,
        output_file=args.output,
    )


if __name__ == '__main__':
    main()