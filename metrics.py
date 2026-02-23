"""
Pluggable spacetime metrics for the geodesic tracer.

Each metric encapsulates all metric-specific physics: geodesic equations,
initial conditions, ray tracing, and critical angles.

Uniform 8D state vector: [t, r, theta, phi, p_t, p_r, p_theta, p_phi].
"""

from abc import ABC, abstractmethod

import numpy as np

try:
    from numba import njit
    NUMBA_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    NUMBA_AVAILABLE = False

    def njit(*args, **kwargs):
        if args and callable(args[0]) and len(args) == 1 and not kwargs:
            return args[0]

        def decorator(func):
            return func
        return decorator


USE_COMPILED_TRACER = NUMBA_AVAILABLE


@njit(cache=True)
def _clip_scalar(x, lo, hi):
    if x < lo:
        return lo
    if x > hi:
        return hi
    return x


@njit(cache=True)
def _schwarzschild_orbit_rhs_numba(u, w, M):
    return w, -u + 3.0 * M * u * u


@njit(cache=True)
def _schwarzschild_trace_orbit_numba(M, R_S, r_obs, alpha, phi_max, h_max):
    f0 = 1.0 - R_S / r_obs
    if f0 <= 0.0:
        return 0, np.nan, 0.0, 0.0

    b = r_obs * np.sin(alpha) / np.sqrt(f0)
    if b == 0.0:
        return 0, np.nan, 0.0, 0.0

    u = 1.0 / r_obs
    w0_sq = 1.0 / (b * b) - u * u + 2.0 * M * u * u * u
    if w0_sq < 0.0:
        return 0, np.nan, 0.0, 0.0
    w = np.sqrt(w0_sq)

    phi = 0.0
    u_capture = 1.0 / (R_S * 1.01)
    u_escape = 1.0 / (2.0 * r_obs)

    # status: 1 escaped, -1 captured, 0 invalid, 2 max-range (treat as escaped)
    status = 2

    while phi < phi_max:
        h = h_max
        remaining = phi_max - phi
        if remaining < h:
            h = remaining
        if h <= 0.0:
            break

        u_prev = u
        w_prev = w

        k1u, k1w = _schwarzschild_orbit_rhs_numba(u_prev, w_prev, M)
        k2u, k2w = _schwarzschild_orbit_rhs_numba(
            u_prev + 0.5 * h * k1u, w_prev + 0.5 * h * k1w, M)
        k3u, k3w = _schwarzschild_orbit_rhs_numba(
            u_prev + 0.5 * h * k2u, w_prev + 0.5 * h * k2w, M)
        k4u, k4w = _schwarzschild_orbit_rhs_numba(
            u_prev + h * k3u, w_prev + h * k3w, M)

        u = u_prev + (h / 6.0) * (k1u + 2.0 * k2u + 2.0 * k3u + k4u)
        w = w_prev + (h / 6.0) * (k1w + 2.0 * k2w + 2.0 * k3w + k4w)
        phi_next = phi + h

        if u_prev < u_capture and u >= u_capture:
            denom = u - u_prev
            frac = 1.0 if denom == 0.0 else (u_capture - u_prev) / denom
            frac = _clip_scalar(frac, 0.0, 1.0)
            phi = phi + frac * h
            w = w_prev + frac * (w - w_prev)
            u = u_capture
            status = -1
            break

        if u_prev > u_escape and u <= u_escape:
            denom = u - u_prev
            frac = 1.0 if denom == 0.0 else (u_escape - u_prev) / denom
            frac = _clip_scalar(frac, 0.0, 1.0)
            phi = phi + frac * h
            w = w_prev + frac * (w - w_prev)
            u = u_escape
            status = 1
            break

        phi = phi_next

    return status, phi, u, w


@njit(cache=True)
def _kerr_initial_conditions_numba(M, a, r_obs, alpha, theta, theta_obs):
    state = np.empty(8, dtype=np.float64)

    r = r_obs
    th = theta_obs
    sin_th = np.sin(th)
    cos_th = np.cos(th)
    sin_th_sq = sin_th * sin_th
    if sin_th_sq < 1e-15:
        sin_th_sq = 1e-15

    Sigma = r * r + a * a * cos_th * cos_th
    Delta = r * r - 2.0 * M * r + a * a
    if Delta <= 0.0 or Sigma <= 0.0:
        return False, state

    sin_alpha = np.sin(alpha)
    sin_screen = np.sin(theta)
    cos_screen = np.cos(theta)

    E = 1.0

    sqrt_Delta = np.sqrt(Delta)
    sqrt_Sigma = np.sqrt(Sigma)
    rho = r * sin_alpha * sqrt_Sigma / sqrt_Delta

    alpha_screen = -rho * sin_screen
    beta_screen = -rho * cos_screen

    xi = -alpha_screen * sin_th
    eta = beta_screen * beta_screen + cos_th * cos_th * (alpha_screen * alpha_screen - a * a)

    L = xi * E
    Q = eta * E * E

    p_t = E
    p_phi = L

    Theta = Q - cos_th * cos_th * (L * L / sin_th_sq - a * a * E * E)
    if Theta < 0.0:
        Theta = 0.0
    p_th_sign = -1.0 if cos_screen > 0.0 else 1.0
    p_theta = p_th_sign * np.sqrt(Theta)

    A_val = (r * r + a * a) * (r * r + a * a) - a * a * Delta * sin_th_sq
    g_tt_inv = -A_val / (Sigma * Delta)
    g_tphi_inv = -2.0 * M * a * r / (Sigma * Delta)
    g_rr_inv = Delta / Sigma
    g_thth_inv = 1.0 / Sigma
    g_phiphi_inv = (Delta - a * a * sin_th_sq) / (Sigma * Delta * sin_th_sq)

    other = (g_tt_inv * p_t * p_t
             + 2.0 * g_tphi_inv * p_t * p_phi
             + g_thth_inv * p_theta * p_theta
             + g_phiphi_inv * p_phi * p_phi)
    p_r_sq = -other / g_rr_inv
    if p_r_sq < 0.0:
        p_r_sq = 0.0
    p_r = -np.sqrt(p_r_sq)

    state[0] = 0.0
    state[1] = r
    state[2] = th
    state[3] = 0.0
    state[4] = p_t
    state[5] = p_r
    state[6] = p_theta
    state[7] = p_phi
    return True, state


@njit(cache=True)
def _kerr_geodesic_equations_numba(state, M, a, r_plus, out):
    t, r, th, phi, p_t, p_r, p_th, p_phi = state

    if r <= r_plus * 1.001:
        for i in range(8):
            out[i] = 0.0
        return

    sin_th = np.sin(th)
    cos_th = np.cos(th)
    sin_th_sq = sin_th * sin_th
    if sin_th_sq < 1e-15:
        sin_th_sq = 1e-15

    Sigma = r * r + a * a * cos_th * cos_th
    Delta = r * r - 2.0 * M * r + a * a
    A = (r * r + a * a) * (r * r + a * a) - a * a * Delta * sin_th_sq

    g_tt_inv = -A / (Sigma * Delta)
    g_tphi_inv = -2.0 * M * a * r / (Sigma * Delta)
    g_rr_inv = Delta / Sigma
    g_thth_inv = 1.0 / Sigma
    g_phiphi_inv = (Delta - a * a * sin_th_sq) / (Sigma * Delta * sin_th_sq)

    dt = g_tt_inv * p_t + g_tphi_inv * p_phi
    dr = g_rr_inv * p_r
    dth = g_thth_inv * p_th
    dphi = g_tphi_inv * p_t + g_phiphi_inv * p_phi

    dSigma_dr = 2.0 * r
    dDelta_dr = 2.0 * r - 2.0 * M
    dA_dr = 4.0 * r * (r * r + a * a) - a * a * dDelta_dr * sin_th_sq

    sigma_delta = Sigma * Delta
    sigma_delta_sq = sigma_delta * sigma_delta
    dg_tt_inv_dr = (-(dA_dr * sigma_delta
                      - A * (dSigma_dr * Delta + Sigma * dDelta_dr))
                    / sigma_delta_sq)
    dg_tphi_inv_dr = (-(2.0 * M * a * (sigma_delta
                        - r * (dSigma_dr * Delta + Sigma * dDelta_dr)))
                      / sigma_delta_sq)
    dg_rr_inv_dr = (dDelta_dr * Sigma - Delta * dSigma_dr) / (Sigma * Sigma)
    dg_thth_inv_dr = -dSigma_dr / (Sigma * Sigma)
    den_phi_dr = Sigma * Delta * sin_th_sq
    dg_phiphi_inv_dr = ((dDelta_dr * den_phi_dr
                         - (Delta - a * a * sin_th_sq)
                         * (dSigma_dr * Delta + Sigma * dDelta_dr) * sin_th_sq)
                        / (den_phi_dr * den_phi_dr))

    dp_r = -0.5 * (dg_tt_inv_dr * p_t * p_t
                   + 2.0 * dg_tphi_inv_dr * p_t * p_phi
                   + dg_rr_inv_dr * p_r * p_r
                   + dg_thth_inv_dr * p_th * p_th
                   + dg_phiphi_inv_dr * p_phi * p_phi)

    dSigma_dth = -2.0 * a * a * sin_th * cos_th
    dA_dth = -a * a * Delta * 2.0 * sin_th * cos_th

    dg_tt_inv_dth = (-(dA_dth * Sigma * Delta - A * dSigma_dth * Delta)
                     / sigma_delta_sq)
    dg_tphi_inv_dth = 2.0 * M * a * r * dSigma_dth / (Sigma * Sigma * Delta)
    dg_rr_inv_dth = -Delta * dSigma_dth / (Sigma * Sigma)
    dg_thth_inv_dth = -dSigma_dth / (Sigma * Sigma)

    num = Delta - a * a * sin_th_sq
    den = Sigma * Delta * sin_th_sq
    dnum_dth = -a * a * 2.0 * sin_th * cos_th
    dden_dth = dSigma_dth * Delta * sin_th_sq + Sigma * Delta * 2.0 * sin_th * cos_th
    dg_phiphi_inv_dth = (dnum_dth * den - num * dden_dth) / (den * den)

    dp_th = -0.5 * (dg_tt_inv_dth * p_t * p_t
                    + 2.0 * dg_tphi_inv_dth * p_t * p_phi
                    + dg_rr_inv_dth * p_r * p_r
                    + dg_thth_inv_dth * p_th * p_th
                    + dg_phiphi_inv_dth * p_phi * p_phi)

    out[0] = dt
    out[1] = dr
    out[2] = dth
    out[3] = dphi
    out[4] = 0.0
    out[5] = dp_r
    out[6] = dp_th
    out[7] = 0.0


@njit(cache=True)
def _rk4_step_kerr_numba(state, h, M, a, r_plus, k1, k2, k3, k4, tmp, out_state):
    _kerr_geodesic_equations_numba(state, M, a, r_plus, k1)
    for i in range(8):
        tmp[i] = state[i] + 0.5 * h * k1[i]

    _kerr_geodesic_equations_numba(tmp, M, a, r_plus, k2)
    for i in range(8):
        tmp[i] = state[i] + 0.5 * h * k2[i]

    _kerr_geodesic_equations_numba(tmp, M, a, r_plus, k3)
    for i in range(8):
        tmp[i] = state[i] + h * k3[i]

    _kerr_geodesic_equations_numba(tmp, M, a, r_plus, k4)
    for i in range(8):
        out_state[i] = state[i] + (h / 6.0) * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i])


@njit(cache=True)
def _kerr_trace_ray_numba(M, a, r_plus, r_obs, alpha, theta, theta_obs,
                          lambda_max, h_max):
    ok, state = _kerr_initial_conditions_numba(M, a, r_obs, alpha, theta, theta_obs)
    if not ok:
        return 0, np.nan, 0

    r_capture = r_plus * 1.01
    r_escape = r_obs * 2.0

    k1 = np.empty(8, dtype=np.float64)
    k2 = np.empty(8, dtype=np.float64)
    k3 = np.empty(8, dtype=np.float64)
    k4 = np.empty(8, dtype=np.float64)
    tmp = np.empty(8, dtype=np.float64)
    next_state = np.empty(8, dtype=np.float64)

    lam = 0.0
    event_status = 2  # 1 escaped, -1 captured, 2 max-range

    while lam < lambda_max:
        h = h_max
        remaining = lambda_max - lam
        if remaining < h:
            h = remaining
        if h <= 0.0:
            break

        r_prev = state[1]

        _rk4_step_kerr_numba(state, h, M, a, r_plus, k1, k2, k3, k4, tmp, next_state)
        r_next = next_state[1]

        if r_prev > r_capture and r_next <= r_capture:
            denom = r_next - r_prev
            frac = 1.0 if denom == 0.0 else (r_capture - r_prev) / denom
            frac = _clip_scalar(frac, 0.0, 1.0)
            for i in range(8):
                state[i] = state[i] + frac * (next_state[i] - state[i])
            lam += frac * h
            event_status = -1
            break

        if r_prev < r_escape and r_next >= r_escape:
            denom = r_next - r_prev
            frac = 1.0 if denom == 0.0 else (r_escape - r_prev) / denom
            frac = _clip_scalar(frac, 0.0, 1.0)
            for i in range(8):
                state[i] = state[i] + frac * (next_state[i] - state[i])
            lam += frac * h
            event_status = 1
            break

        for i in range(8):
            state[i] = next_state[i]
        lam += h

    r_f = state[1]
    th_f = state[2]
    phi_f = state[3]
    p_t_f = state[4]
    p_r_f = state[5]
    p_th_f = state[6]
    p_phi_f = state[7]

    n_half_orbits = int(np.abs(phi_f) // np.pi)

    if r_f <= r_capture * 1.1 or event_status == -1:
        return -1, np.nan, n_half_orbits

    sin_th = np.sin(th_f)
    cos_th = np.cos(th_f)
    sin_th_sq = sin_th * sin_th
    if sin_th_sq < 1e-15:
        sin_th_sq = 1e-15
    Sigma_f = r_f * r_f + a * a * cos_th * cos_th
    Delta_f = r_f * r_f - 2.0 * M * r_f + a * a

    dr_dl = Delta_f / Sigma_f * p_r_f
    dth_dl = p_th_f / Sigma_f
    dphi_dl = (-2.0 * M * a * r_f / (Sigma_f * Delta_f) * p_t_f
               + (Delta_f - a * a * sin_th_sq) / (Sigma_f * Delta_f * sin_th_sq) * p_phi_f)

    sin_phi = np.sin(phi_f)
    cos_phi = np.cos(phi_f)

    vx = (sin_th * cos_phi * dr_dl
          + r_f * cos_th * cos_phi * dth_dl
          - r_f * sin_th * sin_phi * dphi_dl)
    vy = (sin_th * sin_phi * dr_dl
          + r_f * cos_th * sin_phi * dth_dl
          + r_f * sin_th * cos_phi * dphi_dl)
    vz = cos_th * dr_dl - r_f * sin_th * dth_dl

    v_mag = np.sqrt(vx * vx + vy * vy + vz * vz)
    if v_mag < 1e-30:
        return 1, np.nan, n_half_orbits

    final_alpha = np.arccos(_clip_scalar(-vx / v_mag, -1.0, 1.0))
    return 1, final_alpha, n_half_orbits


class Metric(ABC):
    """Base class for spacetime metrics."""

    is_spherically_symmetric = False

    @abstractmethod
    def geodesic_equations(self, lambda_, state):
        """RHS for Hamilton's equations.

        State: [t, r, theta, phi, p_t, p_r, p_theta, p_phi].
        """
        ...

    @abstractmethod
    def initial_conditions(self, r_obs, alpha, theta=0.0,
                           theta_obs=np.pi / 2):
        """Initial 8D state vector for a photon at viewing angle alpha.

        Returns None if no valid trajectory exists.
        """
        ...

    @abstractmethod
    def trace_ray(self, r_obs, alpha, theta=0.0, theta_obs=np.pi / 2,
                  phi_max=50.0):
        """Trace a ray, return (final_alpha, n_half_orbits, outcome).

        final_alpha: deflected viewing angle of escaping photon (radians).
        n_half_orbits: number of half-orbits (winding number).
        outcome: 'escaped', 'captured', or 'invalid'.
        """
        ...

    @abstractmethod
    def alpha_crit(self, r_obs, theta_obs=np.pi / 2):
        """Critical viewing angle in radians."""
        ...

    @abstractmethod
    def capture_radius(self):
        """Inner stopping radius for integration."""
        ...

    def viewing_angle_to_impact_parameter(self, alpha, r_obs,
                                          theta_obs=np.pi / 2):
        """Convert viewing angle to impact parameter. Override for Kerr."""
        raise NotImplementedError


# ============================================================================
# Schwarzschild
# ============================================================================

class Schwarzschild(Metric):
    """Schwarzschild metric: non-rotating black hole of mass M."""

    is_spherically_symmetric = True

    def __init__(self, M=1.0):
        self.M = M
        self.R_S = 2 * M
        self.R_PHOTON = 3 * M
        self.B_CRIT = 3 * np.sqrt(3) * M

    def _f(self, r):
        """Metric function f(r) = 1 - R_S / r."""
        return 1 - self.R_S / r

    def capture_radius(self):
        return self.R_S * 1.01

    def alpha_crit(self, r_obs, theta_obs=np.pi / 2):
        arg = self.B_CRIT * np.sqrt(self._f(r_obs)) / r_obs
        return np.arcsin(np.clip(arg, -1.0, 1.0))

    def viewing_angle_to_impact_parameter(self, alpha, r_obs,
                                          theta_obs=np.pi / 2):
        return r_obs * np.sin(alpha) / np.sqrt(self._f(r_obs))

    # -- Geodesic equations (8D, equatorial) ---------------------------------

    def geodesic_equations(self, lambda_, state):
        t, r, th, phi, p_t, p_r, p_th, p_phi = state

        if r <= self.R_S * 1.001:
            return [0.0] * 8

        f = self._f(r)
        R_S = self.R_S

        dt = p_t / f
        dr = f * p_r
        dth = p_th / r**2
        dphi = p_phi / r**2

        dp_t = 0.0
        dp_r = (-(R_S / (2 * r**2)) * (p_t**2 / f**2)
                - (R_S / (2 * r**2)) * p_r**2
                + (p_th**2 + p_phi**2) / r**3)
        dp_th = 0.0  # equatorial, p_theta stays 0
        dp_phi = 0.0

        return [dt, dr, dth, dphi, dp_t, dp_r, dp_th, dp_phi]

    # -- Initial conditions (8D) ---------------------------------------------

    def initial_conditions(self, r_obs, alpha, theta=0.0,
                           theta_obs=np.pi / 2):
        b = self.viewing_angle_to_impact_parameter(alpha, r_obs)

        f0 = self._f(r_obs)
        E = 1.0
        L = b * E

        p_r_sq = (E**2 / f0 - L**2 / r_obs**2) / f0
        if p_r_sq < 0:
            return None

        p_r = -np.sqrt(p_r_sq)  # inward

        return [0.0, r_obs, np.pi / 2, 0.0,
                E, p_r, 0.0, L]

    # -- Orbit equation (fast path) ------------------------------------------

    def _orbit_equations(self, phi, state):
        u, w = state
        return [w, -u + 3 * self.M * u**2]

    def trace_ray(self, r_obs, alpha, theta=0.0, theta_obs=np.pi / 2,
                  phi_max=50.0):
        """Trace using the orbit equation shortcut.

        Returns (final_alpha, n_half_orbits, outcome).
        """
        status, phi_f, u_f, w_f = _schwarzschild_trace_orbit_numba(
            self.M, self.R_S, r_obs, alpha, phi_max, 0.05)
        if status == 0:
            return np.nan, 0, 'invalid'

        r_f = 1.0 / u_f
        if status == -1 or r_f <= self.R_S * 1.1:
            return np.nan, int(abs(phi_f) // np.pi), 'captured'

        dr_dphi = -w_f / (u_f * u_f)
        heading = np.arctan2(
            dr_dphi * np.sin(phi_f) + r_f * np.cos(phi_f),
            dr_dphi * np.cos(phi_f) - r_f * np.sin(phi_f),
        )
        final_alpha = np.arccos(np.clip(-np.cos(heading), -1.0, 1.0))
        n_half_orbits = int(abs(phi_f) // np.pi)

        return final_alpha, n_half_orbits, 'escaped'


# ============================================================================
# Kerr
# ============================================================================

class Kerr(Metric):
    """Kerr metric in Boyer-Lindquist coordinates: spinning BH with spin a.

    Parameters: M (mass), a (spin, |a| <= M).
    """

    is_spherically_symmetric = False

    def __init__(self, M=1.0, a=0.0):
        if abs(a) > M:
            raise ValueError(f"|a|={abs(a)} exceeds M={M}")
        self.M = M
        self.a = a
        self.r_plus = M + np.sqrt(M**2 - a**2)  # outer horizon

    def _Sigma(self, r, th):
        return r**2 + self.a**2 * np.cos(th)**2

    def _Delta(self, r):
        return r**2 - 2 * self.M * r + self.a**2

    def capture_radius(self):
        return self.r_plus * 1.01

    # -- Critical photon orbits (prograde / retrograde) ----------------------

    def _unstable_photon_r(self):
        """Radii of unstable circular photon orbits (prograde, retrograde)."""
        M, a = self.M, self.a
        if a == 0:
            return 3 * M, 3 * M
        # Bardeen's formula
        r_pro = 2 * M * (1 + np.cos(2 / 3 * np.arccos(-a / M)))
        r_ret = 2 * M * (1 + np.cos(2 / 3 * np.arccos(a / M)))
        return r_pro, r_ret

    def _critical_impact_params(self):
        """Critical impact parameters (b, q) for prograde and retrograde
        photon orbits."""
        M, a = self.M, self.a
        results = []
        for r_ph in self._unstable_photon_r():
            Delta = self._Delta(r_ph)
            xi = (r_ph**2 + a**2) / a - r_ph * Delta / (a * (r_ph - M))
            eta = (r_ph**3 / (a**2 * (r_ph - M)**2)
                   * (4 * M * Delta - r_ph * (r_ph - M)**2))
            results.append((xi, eta))
        return results  # [(xi_pro, eta_pro), (xi_ret, eta_ret)]

    def alpha_crit(self, r_obs, theta_obs=np.pi / 2):
        """Critical viewing angle. For Kerr this returns the larger
        (retrograde) critical angle — the full shadow boundary is
        asymmetric but this gives the conservative envelope."""
        if self.a == 0:
            # Reduce to Schwarzschild formula
            R_S = 2 * self.M
            B_CRIT = 3 * np.sqrt(3) * self.M
            f = 1 - R_S / r_obs
            arg = B_CRIT * np.sqrt(f) / r_obs
            return np.arcsin(np.clip(arg, -1.0, 1.0))

        # Use retrograde (larger) critical impact parameter
        crits = self._critical_impact_params()
        xi_ret, eta_ret = crits[1]
        b_crit = np.sqrt(xi_ret**2 + eta_ret)

        Delta_obs = self._Delta(r_obs)
        Sigma_obs = self._Sigma(r_obs, theta_obs)
        sin_th = np.sin(theta_obs)
        A = (r_obs**2 + self.a**2)**2 - self.a**2 * Delta_obs * sin_th**2
        arg = b_crit * np.sqrt(Sigma_obs * Delta_obs / A) / r_obs
        return np.arcsin(np.clip(arg, -1.0, 1.0))

    def viewing_angle_to_impact_parameter(self, alpha, r_obs,
                                          theta_obs=np.pi / 2):
        if self.a == 0:
            f = 1 - 2 * self.M / r_obs
            return r_obs * np.sin(alpha) / np.sqrt(f)

        Delta = self._Delta(r_obs)
        Sigma = self._Sigma(r_obs, theta_obs)
        sin_th = np.sin(theta_obs)
        A = (r_obs**2 + self.a**2)**2 - self.a**2 * Delta * sin_th**2
        return r_obs * np.sin(alpha) * np.sqrt(A / (Sigma * Delta))

    # -- Geodesic equations (full Kerr Hamiltonian) --------------------------

    def geodesic_equations(self, lambda_, state):
        t, r, th, phi, p_t, p_r, p_th, p_phi = state
        M, a = self.M, self.a

        if r <= self.r_plus * 1.001:
            return [0.0] * 8

        sin_th = np.sin(th)
        cos_th = np.cos(th)
        Sigma = r**2 + a**2 * cos_th**2
        Delta = r**2 - 2 * M * r + a**2
        A = (r**2 + a**2)**2 - a**2 * Delta * sin_th**2

        # Inverse metric components (contravariant)
        g_tt_inv = -A / (Sigma * Delta)
        g_tphi_inv = -2 * M * a * r / (Sigma * Delta)
        g_rr_inv = Delta / Sigma
        g_thth_inv = 1.0 / Sigma
        g_phiphi_inv = (Delta - a**2 * sin_th**2) / (Sigma * Delta * sin_th**2)

        # Hamilton's equations: dx^mu / dlambda = dH/dp_mu = g^{mu nu} p_nu
        dt = g_tt_inv * p_t + g_tphi_inv * p_phi
        dr = g_rr_inv * p_r
        dth = g_thth_inv * p_th
        dphi = g_tphi_inv * p_t + g_phiphi_inv * p_phi

        # dp_mu / dlambda = -dH/dx^mu = -(1/2) dg^{ab}/dx^mu p_a p_b
        # Partial derivatives w.r.t. r
        dSigma_dr = 2 * r
        dDelta_dr = 2 * r - 2 * M
        dA_dr = 4 * r * (r**2 + a**2) - a**2 * dDelta_dr * sin_th**2

        dg_tt_inv_dr = (-(dA_dr * Sigma * Delta
                          - A * (dSigma_dr * Delta + Sigma * dDelta_dr))
                        / (Sigma * Delta)**2)
        dg_tphi_inv_dr = (-(2 * M * a * (Sigma * Delta
                            - r * (dSigma_dr * Delta + Sigma * dDelta_dr)))
                          / (Sigma * Delta)**2)
        dg_rr_inv_dr = (dDelta_dr * Sigma - Delta * dSigma_dr) / Sigma**2
        dg_thth_inv_dr = -dSigma_dr / Sigma**2
        dg_phiphi_inv_dr = ((dDelta_dr * Sigma * Delta * sin_th**2
                             - (Delta - a**2 * sin_th**2)
                             * (dSigma_dr * Delta + Sigma * dDelta_dr)
                             * sin_th**2)
                            / (Sigma * Delta * sin_th**2)**2)

        dp_r = -0.5 * (dg_tt_inv_dr * p_t**2
                        + 2 * dg_tphi_inv_dr * p_t * p_phi
                        + dg_rr_inv_dr * p_r**2
                        + dg_thth_inv_dr * p_th**2
                        + dg_phiphi_inv_dr * p_phi**2)

        # Partial derivatives w.r.t. theta
        dSigma_dth = -2 * a**2 * sin_th * cos_th
        dA_dth = -a**2 * Delta * 2 * sin_th * cos_th

        dg_tt_inv_dth = (-(dA_dth * Sigma * Delta
                           - A * dSigma_dth * Delta)
                         / (Sigma * Delta)**2)
        dg_tphi_inv_dth = (2 * M * a * r * dSigma_dth
                           / (Sigma * Delta)**2 * Delta)
        # Simplify: dg_tphi_inv_dth = 2*M*a*r*dSigma_dth / (Sigma**2 * Delta)
        dg_tphi_inv_dth = 2 * M * a * r * dSigma_dth / (Sigma**2 * Delta)
        dg_rr_inv_dth = -Delta * dSigma_dth / Sigma**2
        dg_thth_inv_dth = -dSigma_dth / Sigma**2

        # g_phiphi_inv = (Delta - a^2 sin^2) / (Sigma Delta sin^2)
        num = Delta - a**2 * sin_th**2
        den = Sigma * Delta * sin_th**2
        dnum_dth = -a**2 * 2 * sin_th * cos_th
        dden_dth = (dSigma_dth * Delta * sin_th**2
                    + Sigma * Delta * 2 * sin_th * cos_th)
        dg_phiphi_inv_dth = (dnum_dth * den - num * dden_dth) / den**2

        dp_th = -0.5 * (dg_tt_inv_dth * p_t**2
                         + 2 * dg_tphi_inv_dth * p_t * p_phi
                         + dg_rr_inv_dth * p_r**2
                         + dg_thth_inv_dth * p_th**2
                         + dg_phiphi_inv_dth * p_phi**2)

        dp_t = 0.0    # t is cyclic
        dp_phi = 0.0  # phi is cyclic

        return [dt, dr, dth, dphi, dp_t, dp_r, dp_th, dp_phi]

    # -- Initial conditions --------------------------------------------------

    def initial_conditions(self, r_obs, alpha, theta=0.0,
                           theta_obs=np.pi / 2):
        """Compute initial state for a photon at the observer.

        theta is the azimuthal screen angle (0 = up, pi/2 = right on screen).
        theta_obs is the observer's polar angle (inclination from spin axis).
        """
        M, a = self.M, self.a
        r = r_obs
        th = theta_obs
        sin_th = np.sin(th)
        cos_th = np.cos(th)

        Sigma = r**2 + a**2 * cos_th**2
        Delta = r**2 - 2 * M * r + a**2
        A = (r**2 + a**2)**2 - a**2 * Delta * sin_th**2

        sin_alpha = np.sin(alpha)
        sin_screen = np.sin(theta)
        cos_screen = np.cos(theta)

        E = 1.0

        # Screen → celestial impact parameters (Bardeen 1973 convention):
        # rho = effective impact parameter on the observer's sky
        sqrt_Delta = np.sqrt(Delta)
        sqrt_Sigma = np.sqrt(Sigma)
        rho = r * sin_alpha * sqrt_Sigma / sqrt_Delta

        # Celestial coordinates (alpha_screen toward phi, beta_screen toward pole)
        alpha_screen = -rho * sin_screen
        beta_screen = -rho * cos_screen

        # Conserved quantities: xi = L/E, eta = Q/E^2
        # alpha_screen = -xi / sin(theta_obs)
        # beta_screen^2 = eta + a^2 cos^2(theta_obs) - xi^2 cot^2(theta_obs)
        xi = -alpha_screen * sin_th
        eta = (beta_screen**2
               + cos_th**2 * (alpha_screen**2 - a**2))

        L = xi * E
        Q = eta * E**2

        p_t = E
        p_phi = L

        # p_theta from Carter constant:
        # Q = p_theta^2 + cos^2(theta) * (L^2/sin^2(theta) - a^2 E^2)
        Theta = Q - cos_th**2 * (L**2 / sin_th**2 - a**2 * E**2)
        if Theta < 0:
            # Can happen near edge; clamp
            Theta = 0.0
        p_th_sign = -1.0 if cos_screen > 0 else 1.0
        p_theta = p_th_sign * np.sqrt(Theta)

        # p_r from null condition: g^{mu nu} p_mu p_nu = 0
        # Solve directly to avoid sign-convention issues with the R(r) potential
        A_val = (r**2 + a**2)**2 - a**2 * Delta * sin_th**2
        g_tt_inv = -A_val / (Sigma * Delta)
        g_tphi_inv = -2 * M * a * r / (Sigma * Delta)
        g_rr_inv = Delta / Sigma
        g_thth_inv = 1.0 / Sigma
        g_phiphi_inv = ((Delta - a**2 * sin_th**2)
                        / (Sigma * Delta * sin_th**2))

        other = (g_tt_inv * p_t**2 + 2 * g_tphi_inv * p_t * p_phi
                 + g_thth_inv * p_theta**2 + g_phiphi_inv * p_phi**2)
        p_r_sq = -other / g_rr_inv
        if p_r_sq < 0:
            p_r_sq = 0.0
        p_r = -np.sqrt(p_r_sq)  # inward

        return [0.0, r, th, 0.0,
                p_t, p_r, p_theta, p_phi]

    # -- Ray tracing (full numerical integration) ----------------------------

    def trace_ray(self, r_obs, alpha, theta=0.0, theta_obs=np.pi / 2,
                  phi_max=50.0):
        """Trace a ray numerically.

        Returns (final_alpha, n_half_orbits, outcome).
        """
        status, final_alpha, n_half_orbits = _kerr_trace_ray_numba(
            self.M, self.a, self.r_plus, r_obs, alpha, theta, theta_obs,
            5000.0, 1.0)
        if status == 0:
            return np.nan, 0, 'invalid'
        if status == -1:
            return np.nan, n_half_orbits, 'captured'
        return final_alpha, n_half_orbits, 'escaped'
