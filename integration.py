import sys
import time
from pathlib import Path

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

import vegas

# Import medium property functions from plasma_interaction
ape_dir = str(Path(__file__).resolve().parent.parent)
sys.path.append(ape_dir)
from plasma_interaction import rho, mu_DeBye


#############
# Constants #
#############
_EPS_OMEGA = 1e-6  # threshold for small-argument Taylor fallback (avoids inf*0 / Ci(0) issues)

HBARC = 0.1973269804  # GeV * fm
NC = 3  # Number of colors
SOFT_PIDS = [1, -1, 2, -2, 3, -3, 21]  # Soft parton species for rho0 summation

# NITN_WARMUP = 10  # Iterations for the MC integrators during "warmup"
# NITN = 50  # Number of iterations for actual integration
# NEVAL = 10000  # Number of evaluations for integration
# MCADAPT = True  # Whether to do grid adaptation -- negligible overhead, helps accuracy

NITN_WARMUP = 0  # Iterations for the MC integrators during "warmup"
NITN = 3  # Number of iterations for actual integration
NEVAL = 10000  # Number of evaluations for integration
MCADAPT = True  # Whether to do grid adaptation -- negligible overhead, helps accuracy

DELTAZ = 0.1 / HBARC  # Fixed pathlength where applicable


###########################
# Medium property helpers #
###########################
def compute_rho0(T: float) -> float:
    rho0 = 0.0
    for pid in SOFT_PIDS:
        rho0 += rho(T, soft_pid=pid)
    return rho0


def compute_mu(T: float, g: float) -> float:
    return mu_DeBye(T, g=g)


#####################
# Special Functions #
#####################
def ellippi(n, m):
    """
    Complete elliptic integral of the third kind.
    """
    return sp.special.elliprf(0., 1. - m, 1.) + \
           (n / 3.) * sp.special.elliprj(0., 1. - m, 1., 1. - n)


#########################################
# Persistent, Brute Force MC Integrands #
#########################################
"""
Rescaling every integration variable onto a FIXED [0,1]^d region means the
vegas.Integrator itself never has to change between points, so we build
each one exactly once (at import time) and reuse it for the whole batch.
"""
class _IntegrandParams:
    """Shared parameter-setting logic for all three integrand variants."""

    __slots__ = (
        "x", "E", "mu", "mu2", "u_perp", "z0", "zf", "deltaz",
        "kx", "ky", "ux", "kk", "ku", "uu", "q_lim",
    )

    def set_params(self, x, k_perp, k_phi, E, mu, u_perp, z0, zf):
        self.x, self.E, self.mu = x, E, mu
        self.mu2 = mu * mu
        self.u_perp = u_perp
        self.z0, self.zf = z0, zf
        self.deltaz = zf - z0

        kx = k_perp * np.cos(k_phi)
        ky = k_perp * np.sin(k_phi)
        ux = u_perp

        kk = kx * kx + ky * ky
        if kk < 1e-10:
            kk = 1e-10

        self.kx, self.ky, self.ux = kx, ky, ux
        self.kk = kk
        self.ku = kx * ux
        self.uu = ux * ux
        self.q_lim = np.sqrt(3.0 * E * mu)
        return self


class _BruteMCIntegrand(_IntegrandParams):
    """
    Full (q, q_phi, z) brute-force integrand, rescaled onto the unit cube:
        pts[:,0] = u_q   -> q   = u_q   * q_lim     in (0, q_lim)
        pts[:,1] = u_phi -> phi = u_phi * 2*pi      in (0, 2*pi)
        pts[:,2] = u_z   -> z   = z0 + u_z * deltaz in (z0, zf)
    The extra Jacobian (q_lim * 2*pi * deltaz) from this rescaling is folded
    into the return value, on top of the original polar jacobian `q`.
    """

    def __call__(self, pts):
        x, E, mu2, ux = self.x, self.E, self.mu2, self.ux
        kx, ky, kk, ku, uu = self.kx, self.ky, self.kk, self.ku, self.uu
        q_lim, deltaz, z0 = self.q_lim, self.deltaz, self.z0

        q = pts[:, 0] * q_lim
        q_phi = pts[:, 1] * (2.0 * np.pi)
        z = z0 + pts[:, 2] * deltaz

        qx = q * np.cos(q_phi)
        qy = q * np.sin(q_phi)

        j = q  # polar jacobian dqx dqy = q dq dphi

        kmqx = kx - qx
        kmqy = ky - qy

        qq = qx * qx + qy * qy
        kq = kx * qx + ky * qy
        qu = qx * ux
        kmqu = kmqx * ux
        kmqq = kmqx * qx + kmqy * qy
        kmqkmq = kmqx ** 2 + kmqy ** 2

        q2_mu2 = qq + mu2
        R_sq = qq + mu2 - qu ** 2

        kmqkmq = np.maximum(kmqkmq, 1e-10)
        R_sq = np.maximum(R_sq, 1e-10)
        R = np.sqrt(R_sq)

        v2 = 1.0 / (q2_mu2 ** 2)
        vm2dv2 = -2.0 / q2_mu2

        omega_kmq = kmqkmq / (2.0 * x * E)
        omega_k = kk / (2.0 * x * E)

        t1 = (
            (
                (4.0 * kq / (kk * kmqkmq))
                - (2.0 / (x * E)) * (1.0 / (kk * kmqkmq)) * (
                    2.0 * kmqu * kk
                    + 2.0 * ku * kmqq
                    + kq * qu * (2.0 * kk - kmqkmq) * vm2dv2
                )
            )
            * (1 - np.cos(omega_kmq * z))
        )

        t2 = (
            (1.0 / (x * E)) * (ku / kk)
            * (4.0 + qq * vm2dv2)
            * (1 - np.cos(omega_k * z))
        )

        t3 = (
            (-1)
            * (1.0 / (4 * (R ** 3) * x * E))
            * ((mu2 * ((qu ** 4) + 6 * (qu ** 2) * (R ** 2) - 3 * R ** 4)
               - 2 * (R ** 2) * ((qu ** 4) - (R ** 4))) / kk)
            * vm2dv2
            * np.sin(omega_k * z)
        )

        t4 = (
            (1 / (x * E))
            * ((kk * (1 - uu) + (ku ** 2)) / (kk ** 2))
            * ((((R ** 2) + (qu ** 2)) ** 2) / (R ** 3))
            * np.sin(omega_k * z)
        )

        jac = q_lim * 2.0 * np.pi * deltaz
        return j * v2 * (t1 + t2 + t3 + t4) * jac


class _AnalyticZFullIntegrand(_IntegrandParams):
    """(q, q_phi) integrand with z integrated analytically. Same rescaling
    idea as above, but no z-axis (jac = q_lim * 2*pi)."""

    def __call__(self, pts):
        x, E, mu2, ux = self.x, self.E, self.mu2, self.ux
        kx, ky, kk, ku, uu = self.kx, self.ky, self.kk, self.ku, self.uu
        q_lim, deltaz, z0 = self.q_lim, self.deltaz, self.z0

        q = pts[:, 0] * q_lim
        q_phi = pts[:, 1] * (2.0 * np.pi)

        qx = q * np.cos(q_phi)
        qy = q * np.sin(q_phi)
        j = q

        kmqx = kx - qx
        kmqy = ky - qy

        qq = qx * qx + qy * qy
        kq = kx * qx + ky * qy
        qu = qx * ux
        kmqu = kmqx * ux
        kmqq = kmqx * qx + kmqy * qy
        kmqkmq = kmqx ** 2 + kmqy ** 2

        q2_mu2 = qq + mu2
        R_sq = qq + mu2 - qu ** 2

        kmqkmq = np.maximum(kmqkmq, 1e-10)
        R_sq = np.maximum(R_sq, 1e-10)
        R = np.sqrt(R_sq)

        v2 = 1.0 / (q2_mu2 ** 2)
        vm2dv2 = -2.0 / q2_mu2

        omega_kmq = kmqkmq / (2.0 * x * E)
        omega_k = kk / (2.0 * x * E)

        t1 = (
            (
                (4.0 * kq / (kk * kmqkmq))
                - (2.0 / (x * E)) * (1.0 / (kk * kmqkmq)) * (
                    2.0 * kmqu * kk
                    + 2.0 * ku * kmqq
                    + kq * qu * (2.0 * kk - kmqkmq) * vm2dv2
                )
            )
            * (1 - np.sinc(omega_kmq * deltaz / (2 * np.pi)) * np.cos(omega_kmq * (z0 + deltaz / 2))) * deltaz
        )

        t2 = (
            (1.0 / (x * E)) * (ku / kk)
            * (4.0 + qq * vm2dv2)
            * (1 - np.sinc(omega_k * deltaz / (2 * np.pi)) * np.cos(omega_k * (z0 + deltaz / 2))) * deltaz
        )

        t3 = (
            (-1)
            * (1.0 / (4 * (R ** 3) * x * E))
            * ((mu2 * ((qu ** 4) + 6 * (qu ** 2) * (R ** 2) - 3 * R ** 4)
               - 2 * (R ** 2) * ((qu ** 4) - (R ** 4))) / kk)
            * vm2dv2
            * np.sinc(omega_k * deltaz / (2 * np.pi)) * np.sin(omega_k * (z0 + deltaz / 2)) * deltaz
        )

        t4 = (
            (1 / (x * E))
            * ((kk * (1 - uu) + (ku ** 2)) / (kk ** 2))
            * ((((R ** 2) + (qu ** 2)) ** 2) / (R ** 3))
            * np.sinc(omega_k * deltaz / (2 * np.pi)) * np.sin(omega_k * (z0 + deltaz / 2)) * deltaz
        )

        jac = q_lim * 2.0 * np.pi
        return j * v2 * (t1 + t2 + t3 + t4) * jac


class _T1OnlyIntegrand(_IntegrandParams):
    """t1-only version of the analytic-z integrand (same rescaling)."""

    def __call__(self, pts):
        x, E, mu2, ux = self.x, self.E, self.mu2, self.ux
        kx, ky, kk, ku = self.kx, self.ky, self.kk, self.ku
        q_lim, deltaz, z0 = self.q_lim, self.deltaz, self.z0

        q = pts[:, 0] * q_lim
        q_phi = pts[:, 1] * (2.0 * np.pi)

        qx = q * np.cos(q_phi)
        qy = q * np.sin(q_phi)
        j = q

        kmqx = kx - qx
        kmqy = ky - qy

        kq = kx * qx + ky * qy
        qq = qx * qx + qy * qy
        qu = qx * ux
        kmqu = kmqx * ux
        kmqq = kmqx * qx + kmqy * qy
        kmqkmq = kmqx ** 2 + kmqy ** 2
        kmqkmq = np.maximum(kmqkmq, 1e-10)

        q2_mu2 = qq + mu2
        v2 = 1.0 / (q2_mu2 ** 2)
        vm2dv2 = -2.0 / q2_mu2

        omega_kmq = kmqkmq / (2.0 * x * E)

        t1 = (
            (
                (4.0 * kq / (kk * kmqkmq))
                - (2.0 / (x * E)) * (1.0 / (kk * kmqkmq)) * (
                    2.0 * kmqu * kk
                    + 2.0 * ku * kmqq
                    + kq * qu * (2.0 * kk - kmqkmq) * vm2dv2
                )
            )
            * (1 - np.sinc(omega_kmq * deltaz / (2 * np.pi)) * np.cos(omega_kmq * (z0 + deltaz / 2))) * deltaz
        )

        jac = q_lim * 2.0 * np.pi
        return j * v2 * t1 * jac


# Build every Integrator + integrand ONCE, at import time.
_INTEG_3D = vegas.Integrator([(0., 1.), (0., 1.), (0., 1.)])
_INTEG_2D_FULL = vegas.Integrator([(0., 1.), (0., 1.)])
_INTEG_2D_T1 = vegas.Integrator([(0., 1.), (0., 1.)])

_brutemc_obj = _BruteMCIntegrand()
_analytic_z_obj = _AnalyticZFullIntegrand()
_t1_only_obj = _T1OnlyIntegrand()

_brutemc_integrand = vegas.batchintegrand(_brutemc_obj)
_analytic_z_integrand = vegas.batchintegrand(_analytic_z_obj)
_t1_only_integrand = vegas.batchintegrand(_t1_only_obj)


###################################
# Simplified Per-Term Integrators #
###################################
def t2_analytic(x, k_perp, k_phi, E, mu, u_perp, z0, zf, q_max):
    """
    Implementation of analytic solution for term 2.
    """

    # Derived properties
    _mu2 = mu ** 2
    _mu4 = mu ** 4
    _qmax2 = q_max ** 2
    _qmax4 = q_max ** 4
    _qmax6 = q_max ** 6
    _qmax2_mu2 = (_qmax2 + _mu2)
    deltaz = zf - z0

    # Get cartesian kx
    kx = k_perp * np.cos(k_phi)

    # Flow (u_y = 0 by assumption)
    ux = u_perp

    # Dot products constant for every integration point
    kk = k_perp * k_perp
    ku = kx * ux  # ky * uy = 0 by construction

    # Oscillation frequency
    omega_k = (kk / (2.0 * x * E))

    # Protect against kk = 0
    if kk < 1e-10:
        kk = 1e-10

    analytic = (1 / (x * E)) * (ku / kk)
    q_integral = np.pi * ( 3 * _qmax4 + 4 * _qmax2 * _mu2) / ( _mu2 * (_qmax2_mu2 ** 2) )
    #((2*_qmax2 / (_mu2 * _qmax2_mu2)) + ((_qmax6 + 3 * _qmax4 * _mu2) / (12 * _mu4 * (_qmax2_mu2**3))))
    z_integral = (1 - np.sinc(omega_k * deltaz / (2 * np.pi))*np.cos(omega_k*(z0 + (deltaz/2)))) * deltaz

    return analytic * q_integral * z_integral


def t3_integrand_radial(q, mu, u):
    """
    Radial integrand for term 3 in terms of elliptic integrals
    """
    Q, M, U = q**2, mu**2, u**2
    B     = Q + M
    Delta = Q*(1 - U) + M
    m     = Q*U / B
    P_E   = 4*(4*Q**2*(1-U) + Q*M*(1+4*U) - 3*M**2)

    E = sp.special.ellipe(m)
    K = sp.special.ellipk(m)

    J = np.sqrt(B) * (P_E * E / Delta - 8*(Q - M)*K)
    val = J / B**3
    return val * q


def t3_elliptic(x, k_perp, k_phi, E, mu, u_perp, z0, zf, q_max):
    """
    Implementation of numerical elliptic integrals for term 3.
    """
    kk = k_perp * k_perp
    deltaz = zf - z0
    omega_k = kk / (2.0 * x * E)

    n = 32
    x_lg, w = np.polynomial.legendre.leggauss(n)

    q = 0.5 * q_max * (x_lg + 1.0)      # shape (n,) -- true 1D, no stray dim
    jac = 0.5 * q_max

    vals = t3_integrand_radial(q, mu, u_perp)   # shape (n,)
    q_integral = jac * np.dot(vals, w)          # true scalar

    z_integral = (np.sinc(omega_k * deltaz / (2 * np.pi))
                  * np.sin(omega_k * (z0 + deltaz / 2)) * deltaz)

    analytic = 1.0 / (2.0 * kk * x * E)

    return analytic * q_integral * z_integral   # plain scalar, no .item() needed


def t4_elliptic(x, k_perp, k_phi, E, mu, u_perp, z0, zf, q_max):
    """
    Implementation of numerical elliptic integrals for term 4. This is a valid analytic solution expression.
    """
    uu = u_perp * u_perp
    kk = k_perp * k_perp
    ku = k_perp * u_perp * np.cos(k_phi)
    kuku = ku * ku
    deltaz = zf - z0
    _k4 = kk * kk
    ku = k_perp * u_perp * np.cos(k_phi)
    kuku = ku * ku
    deltaz = zf - z0

    # Oscillation frequency
    omega_k = (kk / (2.0 * x * E))

    b2 = 1.0 - uu
    term1 = 2.0 * np.pi / (mu * np.sqrt(b2))

    m = u_perp ** 2 * q_max ** 2 / (q_max ** 2 + mu ** 2)
    n = u_perp ** 2
    term2 = 4.0 * ellippi(n, m) / np.sqrt(q_max ** 2 + mu ** 2)

    analytic = (1/(x * E)) * ((kk * (1 - uu) + kuku) / _k4)
    q_integral = term1 - term2
    z_integral = np.sinc(omega_k * deltaz / (2 * np.pi) ) * np.sin(omega_k * (z0 + (deltaz/2))) * deltaz

    return analytic * q_integral * z_integral


##############################################
# Reusable Integrand Integrators, Per Method #
##############################################
def integrate_brutemc_t1234(x, k_perp, k_phi, E, mu, u_perp, z0, zf):
    """Integrate a single parameter point using brute force MC method w/ VEGAS+."""
    _brutemc_obj.set_params(x, k_perp, k_phi, E, mu, u_perp, z0, zf)

    if NITN_WARMUP: _INTEG_3D(_brutemc_integrand, nitn=NITN_WARMUP, neval=NEVAL, adapt=MCADAPT)
    result = _INTEG_3D(_brutemc_integrand, nitn=NITN, neval=NEVAL, adapt=MCADAPT)

    return result.mean, result.sdev


def integrate_analytic_z_brutemc_t1234(x, k_perp, k_phi, E, mu, u_perp, z0, zf):
    """
    Integrate a single parameter point using brute force MC method w/ VEGAS+.
    Apply analytic z integration.
    """
    _analytic_z_obj.set_params(x, k_perp, k_phi, E, mu, u_perp, z0, zf)

    if NITN_WARMUP: _INTEG_2D_FULL(_analytic_z_integrand, nitn=NITN_WARMUP, neval=NEVAL, adapt=MCADAPT)
    result = _INTEG_2D_FULL(_analytic_z_integrand, nitn=NITN, neval=NEVAL, adapt=MCADAPT)

    return result.mean, result.sdev


def integrate_analytic_z_t234_brutemc_t1(x, k_perp, k_phi, E, mu, u_perp, z0, zf):
    """
    Integrate a single parameter point using brute force MC method w/ VEGAS+ for t1,
    with analytic/elliptic solutions for t2, t3, t4.
    """
    _t1_only_obj.set_params(x, k_perp, k_phi, E, mu, u_perp, z0, zf)
    q_lim = _t1_only_obj.q_lim

    if NITN_WARMUP: _INTEG_2D_T1(_t1_only_integrand, nitn=NITN_WARMUP, neval=NEVAL, adapt=MCADAPT)
    t1_result = _INTEG_2D_T1(_t1_only_integrand, nitn=NITN, neval=NEVAL, adapt=MCADAPT)

    t2_result = t2_analytic(x, k_perp, k_phi, E, mu, u_perp, z0, zf, q_lim)
    t3_result = t3_elliptic(x, k_perp, k_phi, E, mu, u_perp, z0, zf, q_lim)
    t4_result = t4_elliptic(x, k_perp, k_phi, E, mu, u_perp, z0, zf, q_lim)

    return t1_result.mean + t2_result + t3_result + t4_result, t1_result.sdev


############################
# Method Benchmarking Code #
############################
def _random_batch(n_points, seed=0):
    """
    Generates a random batch of points for every integrator.
    """
    rng = np.random.default_rng(seed)
    deltaz = 0.1 / HBARC
    batch = []
    for _ in range(n_points):
        x = rng.uniform(0.01, 0.99)
        k_perp = rng.uniform(0, 5.0)
        k_phi = rng.uniform(0, 2*np.pi)
        E = rng.uniform(5.0, 50.0)
        mu = rng.uniform(0.3, 1.2)
        u_perp = rng.uniform(0.0, 0.6)
        z0 = rng.uniform(0.0, 10 / HBARC)  # 0 to 10 fm
        zf = z0 + deltaz
        batch.append((x, k_perp, k_phi, E, mu, u_perp, z0, zf))
    return batch


def run_benchmark(n_points=25, n_radial_panels=12, n_phi=16, seed=0, n_origin_subdiv_max=400):
    batch = _random_batch(n_points, seed=seed)

    methods = {
        "vegas t1 + a/e t234 + analytic z (MC)": lambda p: integrate_analytic_z_t234_brutemc_t1(*p)[0],
        "vegas + analytic z (MC)": lambda p: integrate_analytic_z_brutemc_t1234(*p)[0],
        "vegas (MC)": lambda p: integrate_brutemc_t1234(*p)[0],
    }

    results = {}
    for name, fn in methods.items():
        t0 = time.perf_counter()
        values = [float(fn(p)) for p in batch]  # Coerce all integrators to give a float. No shape errors possible!
        t1 = time.perf_counter()
        total = t1 - t0
        results[name] = {
            "total_s": total,
            "per_point_ms": 1000.0 * total / n_points,
            "values": values,
        }
    return results


def _error_stats(vals, ref_vals):
    vals = np.asarray(vals, dtype=float)
    ref_vals = np.asarray(ref_vals, dtype=float)
    diff = vals - ref_vals
    with np.errstate(divide="ignore", invalid="ignore"):
        reldiff = np.abs(diff) / np.abs(ref_vals)
        signed_reldiff = diff / ref_vals
    median_ref = np.median(np.abs(ref_vals))
    norm_diff = np.abs(diff) / (median_ref + 1e-300)

    finite = np.isfinite(reldiff)
    reldiff_f = reldiff[finite]
    signed_f = signed_reldiff[finite]

    return {
        "bias_mean": np.mean(signed_f),                 # signed -> systematic over/under
        "abs_mean": np.mean(reldiff_f),
        "abs_median": np.median(reldiff_f),
        "abs_p90": np.percentile(reldiff_f, 90),
        "abs_p99": np.percentile(reldiff_f, 99),
        "abs_max": np.max(reldiff_f),
        "argmax": np.argmax(reldiff),
        "norm_mean": np.mean(norm_diff),                 # robust to near-zero ref values
        "frac_1pct": np.mean(reldiff_f < 0.01),
        "frac_5pct": np.mean(reldiff_f < 0.05),
        "frac_20pct": np.mean(reldiff_f < 0.20),
    }


def print_benchmark_report(results, reference="vegas t1 + a/e t234 + analytic z (MC)"):
    ref_vals = np.array(results[reference]["values"])
    ref_time = results[reference]["per_point_ms"]

    print(f"Reference: {reference}  ({ref_time:.4f} ms/point, n={len(ref_vals)} points)")
    print("=" * 100)
    col = "{:32s}{:>12s}{:>10s}{:>12s}{:>12s}{:>10s}{:>10s}{:>10s}"
    print(col.format("method", "ms/point", "speedup", "bias", "mean|rd|",
                      "med|rd|", "p90|rd|", "max|rd|"))
    print("-" * 100)

    for name, r in results.items():
        vals = r["values"]
        s = _error_stats(vals, ref_vals)
        speedup = ref_time / r["per_point_ms"]
        row = "{:32s}{:12.4f}{:10.2f}x{:12.2%}{:12.2%}{:10.2%}{:10.2%}{:10.2%}"
        print(row.format(name, r["per_point_ms"], speedup, s["bias_mean"],
                          s["abs_mean"], s["abs_median"], s["abs_p90"], s["abs_max"]))
        print(f"    within 1%: {s['frac_1pct']:.1%}   within 5%: {s['frac_5pct']:.1%}   "
              f"within 20%: {s['frac_20pct']:.1%}   "
              f"<|diff|/median(|ref|)>: {s['norm_mean']:.3e}   "
              f"worst point idx: {s['argmax']}")

    print("=" * 100)
    print("bias        = mean signed relative error (systematic over/under-estimate)")
    print("mean/med/p90/max |rd| = distribution of |value - ref|/|ref| across points")
    print("norm mean   = |diff| normalized by median(|ref|), robust to near-zero ref values")





if __name__ == "__main__":
    print("=" * 70)
    print("Integration Method Comparison")
    print("=" * 70)
    results = run_benchmark(n_points=200, n_radial_panels=12, n_phi=16)
    print_benchmark_report(results)