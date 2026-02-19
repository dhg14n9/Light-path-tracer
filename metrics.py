"""
Pluggable spacetime metrics for the geodesic tracer.

Each metric encapsulates all metric-specific physics: geodesic equations,
initial conditions, ray tracing, and critical angles.

Uniform 8D state vector: [t, r, theta, phi, p_t, p_r, p_theta, p_phi].
"""

from abc import ABC, abstractmethod

import numpy as np
from scipy.integrate import solve_ivp


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
        f0 = self._f(r_obs)
        b = r_obs * np.sin(alpha) / np.sqrt(f0)

        u0 = 1.0 / r_obs
        w0_sq = 1.0 / b**2 - u0**2 + 2 * self.M * u0**3
        if w0_sq < 0:
            return np.nan, 0, 'invalid'
        w0 = np.sqrt(w0_sq)

        R_S = self.R_S

        def event_captured(phi, state):
            return state[0] - 1.0 / (R_S * 1.01)
        event_captured.terminal = True
        event_captured.direction = 1

        def event_escaped(phi, state):
            return state[0] - 1.0 / (2 * r_obs)
        event_escaped.terminal = True
        event_escaped.direction = -1

        sol = solve_ivp(
            self._orbit_equations,
            [0, phi_max],
            [u0, w0],
            method='RK45',
            events=[event_captured, event_escaped],
            rtol=1e-10,
            atol=1e-12,
            max_step=0.05,
        )

        u_f = sol.y[0, -1]
        w_f = sol.y[1, -1]
        phi_f = sol.t[-1]
        r_f = 1.0 / u_f

        if r_f <= R_S * 1.1:
            return np.nan, int(abs(phi_f) // np.pi), 'captured'

        dr_dphi = -w_f / u_f**2
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
        state0 = self.initial_conditions(r_obs, alpha, theta, theta_obs)
        if state0 is None:
            return np.nan, 0, 'invalid'

        r_capture = self.capture_radius()
        r_escape = r_obs * 2.0
        lambda_max = 5000.0

        def event_captured(lam, state):
            return state[1] - r_capture
        event_captured.terminal = True
        event_captured.direction = -1

        def event_escaped(lam, state):
            return state[1] - r_escape
        event_escaped.terminal = True
        event_escaped.direction = 1

        sol = solve_ivp(
            self.geodesic_equations,
            [0, lambda_max],
            state0,
            method='RK45',
            events=[event_captured, event_escaped],
            rtol=1e-10,
            atol=1e-12,
            max_step=1.0,
        )

        r_f = sol.y[1, -1]
        th_f = sol.y[2, -1]
        phi_f = sol.y[3, -1]
        p_r_f = sol.y[5, -1]
        p_th_f = sol.y[6, -1]
        p_phi_f = sol.y[7, -1]
        p_t_f = sol.y[4, -1]

        n_half_orbits = int(abs(phi_f) // np.pi)

        if r_f <= r_capture * 1.1:
            return np.nan, n_half_orbits, 'captured'

        # Compute final_alpha from 3D Cartesian velocity at escape point.
        # Observer at (r_obs, pi/2, 0) looks toward -x.
        # final_alpha = angle between photon velocity and -x axis.
        Sigma_f = self._Sigma(r_f, th_f)
        Delta_f = self._Delta(r_f)
        sin_th = np.sin(th_f)
        cos_th = np.cos(th_f)

        # Coordinate velocities from Hamilton's eqs
        dr_dl = Delta_f / Sigma_f * p_r_f
        dth_dl = p_th_f / Sigma_f
        dphi_dl = (-2 * self.M * self.a * r_f / (Sigma_f * Delta_f)
                   * p_t_f
                   + (Delta_f - self.a**2 * sin_th**2)
                   / (Sigma_f * Delta_f * sin_th**2)
                   * p_phi_f)

        # Cartesian velocity: x = r sin(th) cos(phi), ...
        sin_phi = np.sin(phi_f)
        cos_phi = np.cos(phi_f)

        vx = (sin_th * cos_phi * dr_dl
              + r_f * cos_th * cos_phi * dth_dl
              - r_f * sin_th * sin_phi * dphi_dl)
        vy = (sin_th * sin_phi * dr_dl
              + r_f * cos_th * sin_phi * dth_dl
              + r_f * sin_th * cos_phi * dphi_dl)
        vz = cos_th * dr_dl - r_f * sin_th * dth_dl

        v_mag = np.sqrt(vx**2 + vy**2 + vz**2)
        if v_mag < 1e-30:
            return np.nan, n_half_orbits, 'escaped'

        # Angle from -x axis (observer's line of sight toward BH)
        final_alpha = np.arccos(np.clip(-vx / v_mag, -1.0, 1.0))

        return final_alpha, n_half_orbits, 'escaped'
