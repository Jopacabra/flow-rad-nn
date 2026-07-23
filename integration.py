import sys
from pathlib import Path
import numpy as np
import vegas

# Import medium property functions from plasma_interaction
ape_dir = str(Path(__file__).resolve().parent.parent)
sys.path.append(ape_dir)
from plasma_interaction import rho, mu_DeBye


# ==============================================================================
# Integration settings
# ==============================================================================
NITN_WARMUP = 5
NITN = 8
NEVAL = 3000

# ==============================================================================
# Physical constants
# ==============================================================================
NC = 3  # Number of colors
SOFT_PIDS = [1, -1, 2, -2, 3, -3, 21]  # Soft parton species for rho0 summation

# ==============================================================================
# Medium property helpers
# ==============================================================================
def compute_rho0(T: float) -> float:
    rho0 = 0.0
    for pid in SOFT_PIDS:
        rho0 += rho(T, soft_pid=pid)
    return rho0


def compute_mu(T: float, g: float) -> float:
    return mu_DeBye(T, g=g)

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

def integrate_point(x, kx, ky, E, T, g, u_perp, z0, zf):
    """Integrate the radiation intensity for a single parameter point."""
    q_lim = np.sqrt(6 * E * T)
    region = [(-q_lim, q_lim), (-q_lim, q_lim), (z0, zf)]

    integ    = vegas.Integrator(region)
    integrand = make_batch_integrand(x, kx, ky, E, T, g, u_perp)

    integ(integrand, nitn=NITN_WARMUP, neval=NEVAL)
    result = integ(integrand, nitn=NITN, neval=NEVAL)

    return result.mean, result.sdev


def _integrate_one(task):
    """Top-level worker function for parallel integration. Must be module-level for pickling."""
    idx, x, kx, ky, E, z0, zf, u_perp, T, g = task
    try:
        mean, sdev = integrate_point(x, kx, ky, E, T, g, u_perp, z0, zf)
        return idx, mean, sdev
    except Exception as e:
        print(f"  Warning: Integration failed at index {idx}: {e}")
        return idx, np.nan, np.nan