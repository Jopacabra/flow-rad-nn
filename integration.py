import sys
import time
from pathlib import Path

import numpy as np
import scipy as sp

import vegas


#############
# Constants #
#############
_EPS_OMEGA = 1e-6  # threshold for small-argument Taylor fallback (avoids inf*0 / Ci(0) issues)

HBARC = 0.1973269804  # GeV * fm

# High accuracy, slow
NITN_WARMUP = 10  # Iterations for the MC integrators during "warmup"
NITN = 50  # Number of iterations for actual integration
NEVAL = 10000  # Number of evaluations for integration
MCADAPT = True  # Whether to do grid adaptation -- negligible overhead, helps accuracy

# Paltry accuracy, fast for testing
# NITN_WARMUP = 0  # Iterations for the MC integrators during "warmup"
# NITN = 3  # Number of iterations for actual integration
# NEVAL = 10000  # Number of evaluations for integration
# MCADAPT = False  # Whether to do grid adaptation -- negligible overhead, helps accuracy

MCADAPT_GRIDS = False  # Whether to do grid adaptation evaluating on a grid -- Outliers ruin the grid!


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

@vegas.batchintegrand
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

@vegas.batchintegrand
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

@vegas.batchintegrand
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


#####################################
# Simplified Per-Term q-Integrators #
#####################################
def t2_analytic_qint(_qmax4, _qmax2, _mu2, _qmax2_mu2):
    """
    Return the analytic solution for term 2's q-integration
    """
    return np.pi * ( 3 * _qmax4 + 4 * _qmax2 * _mu2) / ( _mu2 * (_qmax2_mu2 ** 2) )

def t4_elliptic_qint(uu, mu, u_perp, q_max):
    """
    Return the solution for term 4's q-integration
    """
    b2 = 1.0 - uu
    term1 = 2.0 * np.pi / (mu * np.sqrt(b2))

    m = u_perp ** 2 * q_max ** 2 / (q_max ** 2 + mu ** 2)
    n = u_perp ** 2
    term2 = 4.0 * ellippi(n, m) / np.sqrt(q_max ** 2 + mu ** 2)

    return term1 - term2


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
    q_integral = t2_analytic_qint(_qmax4, _qmax2, _mu2, _qmax2_mu2)
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
    _k4 = kk * kk
    ku = k_perp * u_perp * np.cos(k_phi)
    kuku = ku * ku
    deltaz = zf - z0

    # Oscillation frequency
    omega_k = (kk / (2.0 * x * E))

    analytic = (1/(x * E)) * ((kk * (1 - uu) + kuku) / _k4)
    q_integral = t4_elliptic_qint(uu, mu, u_perp, q_max)
    z_integral = np.sinc(omega_k * deltaz / (2 * np.pi) ) * np.sin(omega_k * (z0 + (deltaz/2))) * deltaz

    return analytic * q_integral * z_integral


##############################################
# Reusable Integrand Integrators, Per Method #
##############################################
def integrate_brutemc_t1234(x, k_perp, k_phi, E, mu, u_perp, z0, zf, adapt=MCADAPT):
    """Integrate a single parameter point using brute force MC method w/ VEGAS+."""
    _brutemc_obj.set_params(x, k_perp, k_phi, E, mu, u_perp, z0, zf)

    if NITN_WARMUP: _INTEG_3D(_brutemc_integrand, nitn=NITN_WARMUP, neval=NEVAL, adapt=adapt)
    result = _INTEG_3D(_brutemc_integrand, nitn=NITN, neval=NEVAL, adapt=adapt)

    return result.mean, result.sdev


def integrate_analytic_z_brutemc_t1234(x, k_perp, k_phi, E, mu, u_perp, z0, zf, adapt=MCADAPT):
    """
    Integrate a single parameter point using brute force MC method w/ VEGAS+.
    Apply analytic z integration.
    """
    _analytic_z_obj.set_params(x, k_perp, k_phi, E, mu, u_perp, z0, zf)

    if NITN_WARMUP: _INTEG_2D_FULL(_analytic_z_integrand, nitn=NITN_WARMUP, neval=NEVAL, adapt=adapt)
    result = _INTEG_2D_FULL(_analytic_z_integrand, nitn=NITN, neval=NEVAL, adapt=adapt)

    return result.mean, result.sdev


def integrate_analytic_z_t234_brutemc_t1(x, k_perp, k_phi, E, mu, u_perp, z0, zf, adapt=MCADAPT):
    """
    Integrate a single parameter point using brute force MC method w/ VEGAS+ for t1,
    with analytic/elliptic solutions for t2, t3, t4.
    """
    _t1_only_obj.set_params(x, k_perp, k_phi, E, mu, u_perp, z0, zf)
    q_lim = _t1_only_obj.q_lim

    if NITN_WARMUP: _INTEG_2D_T1(_t1_only_integrand, nitn=NITN_WARMUP, neval=NEVAL, adapt=adapt)
    t1_result = _INTEG_2D_T1(_t1_only_integrand, nitn=NITN, neval=NEVAL, adapt=adapt)

    t2_result = t2_analytic(x, k_perp, k_phi, E, mu, u_perp, z0, zf, q_lim)
    t3_result = t3_elliptic(x, k_perp, k_phi, E, mu, u_perp, z0, zf, q_lim)
    t4_result = t4_elliptic(x, k_perp, k_phi, E, mu, u_perp, z0, zf, q_lim)

    return t1_result.mean + t2_result + t3_result + t4_result, t1_result.sdev


def _direct_compute_harmonics(
        x,
        k_perp,
        E,
        mu,
        u_perp,
        z0,
        zf,
        q_max,
        uu,
        _mu2,
        _mu4,
        _qmax2,
        _qmax4,
        _qmax6,
        _qmax2_mu2,
        kk,
        _k4,
        lkllul,
        _halflkllul2,
        sin_z_integral,
        cos_z_integral
        ):
    """
    Extremely ugly signature direct function to avoid recomputing values per-point in grid usage,
    but still not duplicate code.
    """

    # Term 1
    _t1_only_obj.set_params(x, k_perp, 0, E, mu, u_perp, z0, zf)  # cos(0) - > 1
    if NITN_WARMUP: _INTEG_2D_T1(_t1_only_integrand, nitn=NITN_WARMUP, neval=NEVAL, adapt=MCADAPT_GRIDS)
    t1_A0A1_result = _INTEG_2D_T1(_t1_only_integrand, nitn=NITN, neval=NEVAL, adapt=MCADAPT_GRIDS)

    _t1_only_obj.set_params(x, k_perp, np.pi / 2, E, mu, u_perp, z0, zf)  # cos(pi/2) - > 0
    if NITN_WARMUP: _INTEG_2D_T1(_t1_only_integrand, nitn=NITN_WARMUP, neval=NEVAL, adapt=MCADAPT_GRIDS)
    t1_A0_result = _INTEG_2D_T1(_t1_only_integrand, nitn=NITN, neval=NEVAL, adapt=MCADAPT_GRIDS)

    A0_T1 = t1_A0_result.mean
    A1_T1 = t1_A0A1_result.mean - A0_T1  # Take the difference to get the A1 term alone

    # Term 2 -- Only A1
    t2_analytic = (1 / (x * E)) * (lkllul / kk)
    t2_q_integral = t2_analytic_qint(_qmax4, _qmax2, _mu2, _qmax2_mu2)
    A1_T2 = t2_analytic * t2_q_integral * cos_z_integral

    # Term 3 -- Only A0
    A0_T3 = t3_elliptic(x, k_perp, None, E, mu, u_perp, z0, zf, q_max)
    
    # Term 4 -- A0 and A2
    t4_qintegral = t4_elliptic_qint(uu, mu, u_perp, q_max)
    t4_analytic_A0 = (1 / (x * E)) * ((kk * (1 - uu) + _halflkllul2) / _k4)
    t4_analytic_A2 = (1 / (x * E)) * (_halflkllul2 / _k4)
    A0_T4 = t4_qintegral * t4_analytic_A0 * sin_z_integral
    A2_T4 = t4_qintegral * t4_analytic_A2 * sin_z_integral

    # Combine harmonics
    A0 = A0_T1 + A0_T3 + A0_T4
    A1 = A1_T1 + A1_T2
    A2 = A2_T4

    return [A0, A1, A2]


def integrate_harmonics(x, k_perp, E, mu, u_perp, z0, zf):
    """
    Wrapper to integrate a given parameter point to compute harmonics in k_phi.
    """

    # Collect grid-wide derived property arrays
    q_max = np.sqrt(3 * E * mu)
    _mu2 = mu ** 2
    _mu4 = mu ** 4
    _qmax2 = q_max ** 2
    _qmax4 = q_max ** 4
    _qmax6 = q_max ** 6
    _qmax2_mu2 = (_qmax2 + _mu2)
    deltaz = zf - z0

    # Scalar products
    kk = k_perp ** 2
    uu = u_perp **2
    _k4 = k_perp ** 4
    lkllul = k_perp * u_perp
    _halflkllul2 = 0.5 * (lkllul ** 2)

    # Oscillation frequency
    omega_k = (kk / (2.0 * x * E))

    # Longitudinal integrals
    sin_z_integral = np.sinc(omega_k * deltaz / (2 * np.pi)) * np.sin(omega_k * (z0 + (deltaz / 2))) * deltaz
    cos_z_integral = (1 - np.sinc(omega_k * deltaz / (2 * np.pi)) * np.cos(omega_k * (z0 + (deltaz / 2)))) * deltaz

    # Compute the harmonics
    return _direct_compute_harmonics(x, k_perp, E, mu, u_perp, z0, zf, q_max, uu, _mu2, _mu4, _qmax2, _qmax4, _qmax6, _qmax2_mu2, kk,
                                     _k4, lkllul, _halflkllul2, sin_z_integral, cos_z_integral)


#################################
# Full Spectrum Grid Integrator #
#################################
def get_grid_harmonics(N_x, N_k_perp, N_k_phi, E, mu, u_perp, z0, zf):
    """
    Integrate a full grid in x, k_perp, k_phi using brute force MC method w/ VEGAS+ for t1,
    with analytic/elliptic solutions for t2, t3, t4.

    Exploit known k_phi harmonic decomposition for t2, t3, & t4.
    Perform two-computation trick for t1 to infer the harmonic structure of t1.
    ---

    Term 1 has only A0 and A1 harmonics
    Term 2 has only A1 harmonic
    Term 3 has only A0 harmonic
    Term 4 has only A0 and A2 harmonics

    Parameters
    ----------
    N_x, N_k_perp, N_k_phi : int
        Grid sizes along each axis
    E, mu, u_perp, z0, zf : float
        Physical parameters
    integrator : callable
        One of the defined integrators, e.g.
        `integrate_brutemc_t1234`, `integrate_analytic_z_brutemc_t1234`, or
        `integrate_analytic_z_t234_brutemc_t1`.
        Must have the signature `fn(x, k_perp, k_phi, E, mu, u_perp, z0, zf) -> (mean, sdev)`.

    Returns
    -------
    x_grid_out, k_perp_grid_out, k_phi_grid_out, N_grid
        arrays of shape (N_x, N_k_perp, N_k_phi).
    """
    # Collect grid-wide derived property arrays
    q_max = np.sqrt(3 * E * mu)
    uu = u_perp ** 2
    _mu2 = mu ** 2
    _mu4 = mu ** 4
    _qmax2 = q_max ** 2
    _qmax4 = q_max ** 4
    _qmax6 = q_max ** 6
    _qmax2_mu2 = (_qmax2 + _mu2)
    deltaz = zf - z0

    # Compute radiation kinematic bounds -- see https://arxiv.org/abs/nucl-th/0112071
    x_min = mu / E  # Minimum x based on minimum plasmon frequency
    x_max = 1 - x_min  # Maximum based on consistency with minimum as an IR regulator
    k_perp_min = mu  # Minimum k_perp based on minimum plasmon frequency
    # Note: maximum of ((Min[x^2, (1-x)^2]  * E^2) - mu^2) --> (E^2 - mu^2) / 4
    # k_perp_max = np.sqrt(((E ** 2) - (mu ** 2)) / 4)  # Maxium based on requiring positive z mom. of emission & emitter
    k_perp_max = 5

    # Create array of points in x
    x_min_pow = np.log10(x_min)  # minimum power of 10 in x to compute
    x_max_pow = np.log10(x_max)  # maximum power of 10 in x to compute
    x_values = np.logspace(x_min_pow, x_max_pow, N_x)

    # Create array of points in k_perp
    k_perp_values = np.linspace(k_perp_min, k_perp_max, N_k_perp)

    # Create array of points in phi
    k_phi_values = np.linspace(0, 2 * np.pi, N_k_phi, endpoint=False)  # does not include 2pi -- Overlap w/ 0

    # Create the 3D meshgrid coordinates -- shape (N_k_perp, N_k_phi, N_x)
    x_grid, k_perp_grid, k_phi_grid = np.meshgrid(x_values, k_perp_values, k_phi_values, indexing='ij')

    # Iterate over grid cells and compute harmonics
    harmonics = np.empty((N_x, N_k_perp, 3))
    for ix, x in enumerate(x_values):
        for ik, k_perp in enumerate(k_perp_values):
            # Scalar products
            kk = k_perp ** 2
            _k4 = k_perp ** 4
            lkllul = k_perp * u_perp
            _halflkllul2 = 0.5 * (lkllul ** 2)

            # Oscillation frequency
            omega_k = (kk / (2.0 * x * E))

            # Longitudinal integrals
            sin_z_integral = np.sinc(omega_k * deltaz / (2 * np.pi)) * np.sin(omega_k * (z0 + (deltaz / 2))) * deltaz
            cos_z_integral = (1 - np.sinc(omega_k * deltaz / (2 * np.pi)) * np.cos(
                omega_k * (z0 + (deltaz / 2)))) * deltaz

            # Compute harmonics
            A0, A1, A2 = _direct_compute_harmonics(x, k_perp,  E, mu, u_perp, z0, zf, q_max, uu, _mu2, _mu4, _qmax2,
                                                   _qmax4, _qmax6, _qmax2_mu2, kk, _k4, lkllul, _halflkllul2,
                                                   sin_z_integral, cos_z_integral)

            # Write the harmonics to the array like [A0, A1, A2]
            harmonics[ix, ik, 0] = A0
            harmonics[ix, ik, 1] = A1
            harmonics[ix, ik, 2] = A2

    # Reconstruct full angular dependence via broadcasting -- shape (n_kperp, n_phi, n_x)
    A0_2d = harmonics[:, :, 0]
    A1_2d = harmonics[:, :, 1]
    A2_2d = harmonics[:, :, 2]
    cos_phi = np.cos(k_phi_values)[None, None, :]
    cos_2phi = np.cos(2 * k_phi_values)[None, None, :]
    N_grid = A0_2d[:, :, None] + A1_2d[:, :, None] * cos_phi + A2_2d[:, :, None] * cos_2phi

    return x_grid, k_perp_grid, k_phi_grid, N_grid


def get_grid_explicit(N_x, N_k_perp, N_k_phi, E, mu, u_perp, z0, zf,
                       integrator=integrate_analytic_z_t234_brutemc_t1):
    """
    Integrate a full grid in x, k_perp, k_phi by explicitly evaluating the integral
    at every (x, k_perp, k_phi) grid node

    Parameters
    ----------
    N_x, N_k_perp, N_k_phi : int
        Grid sizes along each axis
    E, mu, u_perp, z0, zf : float
        Physical parameters
    integrator : callable
        One of the defined integrators, e.g.
        `integrate_brutemc_t1234`, `integrate_analytic_z_brutemc_t1234`, or
        `integrate_analytic_z_t234_brutemc_t1`. Must have the signature
        `fn(x, k_perp, k_phi, E, mu, u_perp, z0, zf) -> (mean, sdev)`.

    Returns
    -------
    x_grid_out, k_perp_grid_out, k_phi_grid_out, N_grid
        arrays of shape (N_x, N_k_perp, N_k_phi).
    """
    # Radiation kinematic bounds
    x_min = mu / E
    x_max = 1 - x_min
    k_perp_min = mu
    # k_perp_max = np.sqrt(((E ** 2) - (mu ** 2)) / 4)
    k_perp_max = 5

    x_min_pow = np.log10(x_min)
    x_max_pow = np.log10(x_max)
    x_values = np.logspace(x_min_pow, x_max_pow, N_x)

    k_perp_values = np.linspace(k_perp_min, k_perp_max, N_k_perp)
    k_phi_values = np.linspace(0, 2 * np.pi, N_k_phi, endpoint=False)

    N_grid = np.empty((N_x, N_k_perp, N_k_phi))

    for ix, x in enumerate(x_values):
        for ik, k_perp in enumerate(k_perp_values):
            for iphi, k_phi in enumerate(k_phi_values):
                val, _ = integrator(x, k_perp, k_phi, E, mu, u_perp, z0, zf, adapt=MCADAPT_GRIDS)
                N_grid[ix, ik, iphi] = val

    x_grid_out, k_perp_grid_out, k_phi_grid_out = np.meshgrid(
        x_values, k_perp_values, k_phi_values, indexing='ij'
    )

    return x_grid_out, k_perp_grid_out, k_phi_grid_out, N_grid


############################
# Method Benchmarking Code #
############################
def _random_batch(n_points, seed=0):
    """
    Generates a random batch of points for calling the integrators.
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


def run_benchmark(n_points=25, seed=0):
    batch = _random_batch(n_points, seed=seed)
    """
    Runs each defined integrator for n_points shared parameter points, then compares the results.
    """

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
    """
    Computes error statistics for the results of each integrator compared to the chosen reference integrator.
    """
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
    """
    Print the benchmark comparison information.
    """
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


def run_grid_benchmark(N_x=10, N_k_perp=10, N_k_phi=16, seed=0,
                        E=20.0, mu=0.7, u_perp=0.3, deltaz=0.5):
    """
    Build the same (x, k_perp, k_phi) grid multiple ways and time each:
      - `get_grid_harmonics`: analytic k_phi decomposition/reconstruction.
      - `get_grid_explicit` with each of the three single-point integrators:
        every grid node evaluated directly, with no k_phi shortcut.

    Returns a dict keyed by method name, each holding timing info and the
    raw N_grid array (for accuracy comparison in the printer).
    """
    rng = np.random.default_rng(seed)
    z0 = rng.uniform(0.0, 10 / HBARC)
    zf = z0 + deltaz

    n_grid_points = N_x * N_k_perp * N_k_phi

    methods = {
        "harmonic decomposition": lambda: get_grid_harmonics(
            N_x, N_k_perp, N_k_phi, E, mu, u_perp, z0, zf),
        "explicit: t1(MC)+a/e t234, analytic z": lambda: get_grid_explicit(
            N_x, N_k_perp, N_k_phi, E, mu, u_perp, z0, zf,
            integrator=integrate_analytic_z_t234_brutemc_t1),
        # "explicit: full t1234 (MC), analytic z": lambda: get_grid_explicit(
        #     N_x, N_k_perp, N_k_phi, E, mu, u_perp, z0, zf,
        #     integrator=integrate_analytic_z_brutemc_t1234),
        # "explicit: full t1234 (MC), brute z": lambda: get_grid_explicit(
        #     N_x, N_k_perp, N_k_phi, E, mu, u_perp, z0, zf,
        #     integrator=integrate_brutemc_t1234),
    }

    results = {}
    for name, fn in methods.items():
        t0 = time.perf_counter()
        _, _, _, N_grid = fn()
        t1 = time.perf_counter()
        total = t1 - t0
        results[name] = {
            "total_s": total,
            "per_point_ms": 1000.0 * total / n_grid_points,
            "grid": N_grid,
        }

    results["_meta"] = {
        "n_grid_points": n_grid_points,
        "shape": (N_x, N_k_perp, N_k_phi),
    }
    return results


def print_grid_benchmark_report(results, reference="harmonic decomposition"):
    """
    Print a comparison table across grid-construction methods, reusing the
    same relative-error statistics as `print_benchmark_report`.
    """
    meta = results.get("_meta", {})
    n_grid_points = meta.get("n_grid_points")
    shape = meta.get("shape")

    ref_grid = results[reference]["grid"]
    ref_vals = ref_grid.ravel()
    ref_time = results[reference]["per_point_ms"]
    total_time = results[reference]["total_s"]

    print(f"Reference: {reference}  ({total_time:.4f} s total, {ref_time:.4f} ms/point, "
          f"grid shape={shape}, n={n_grid_points} points)")
    print("=" * 110)
    col = "{:40s}{:>12s}{:>10s}{:>12s}{:>12s}{:>10s}{:>10s}{:>10s}"
    print(col.format("method", "ms/point", "speedup", "bias", "mean|rd|",
                      "med|rd|", "p90|rd|", "max|rd|"))
    print("-" * 110)

    for name, r in results.items():
        if name == "_meta":
            continue
        vals = r["grid"].ravel()
        s = _error_stats(vals, ref_vals)
        speedup = ref_time / r["per_point_ms"]
        row = "{:40s}{:12.4f}{:10.2f}x{:12.2%}{:12.2%}{:10.2%}{:10.2%}{:10.2%}"
        print(row.format(name, r["per_point_ms"], speedup, s["bias_mean"],
                          s["abs_mean"], s["abs_median"], s["abs_p90"], s["abs_max"]))
        print(f"    within 1%: {s['frac_1pct']:.1%}   within 5%: {s['frac_5pct']:.1%}   "
              f"within 20%: {s['frac_20pct']:.1%}   "
              f"<|diff|/median(|ref|)>: {s['norm_mean']:.3e}   "
              f"worst point idx: {s['argmax']}")

    print("=" * 110)
    print("bias        = mean signed relative error (systematic over/under-estimate)")
    print("mean/med/p90/max |rd| = distribution of |value - ref|/|ref| across grid points")
    print("norm mean   = |diff| normalized by median(|ref|), robust to near-zero ref values")


if __name__ == "__main__":
    print("=" * 70)
    print("Integration Method Comparison")
    print("=" * 70)
    results = run_benchmark(n_points=100)
    print_benchmark_report(results)


    print("\n" + "=" * 70)
    print("Grid Construction Comparison (harmonics vs explicit)")
    print("=" * 70)
    grid_results = run_grid_benchmark()
    print_grid_benchmark_report(grid_results)