# pylance: disable=all


"""
Geodesic ray tracer for Schwarzschild black holes.

This script traces null geodesics (light rays) in the Schwarzschild spacetime.
The initial conditions are parameterized by the viewing angle from the observer's
perspective, making it suitable for generating black hole shadow images.

Units: Geometric units where G = c = 1. All lengths are in units of black hole mass M.

Author: [Your name]
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


# =============================================================================
# Physical Constants and Parameters
# =============================================================================

M = 1.0              # Black hole mass (sets the length scale)
R_S = 2 * M          # Schwarzschild radius (event horizon)
R_PHOTON = 3 * M     # Photon sphere radius
B_CRIT = 3 * np.sqrt(3) * M  # Critical impact parameter (~5.196 M)


# =============================================================================
# Geodesic Equations
# =============================================================================

def geodesic_equations(lambda_, state):
    """
    Equations of motion for null geodesics in Schwarzschild spacetime.

    We work in the equatorial plane (theta = pi/2) using the Hamiltonian formulation.

    Parameters
    ----------
    lambda_ : float
        Affine parameter along the geodesic.
    state : array_like
        State vector [t, r, phi, p_t, p_r, p_phi] where:
        - t, r, phi are the coordinates
        - p_t, p_r, p_phi are the conjugate momenta

    Returns
    -------
    derivatives : list
        Time derivatives of the state vector.

    Notes
    -----
    The Hamiltonian for null geodesics is:
        H = (1/2) g^{μν} p_μ p_ν = 0

    For Schwarzschild metric in equatorial plane:
        H = -p_t²/(2f) + f p_r²/2 + p_phi²/(2r²) = 0

    where f = 1 - r_s/r.
    """
    t, r, phi, p_t, p_r, p_phi = state

    # Stop if we're too close to the singularity
    if r <= R_S * 1.001:
        return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    # Metric function
    f = 1 - R_S / r

    # Hamilton's equations: dx^μ/dλ = ∂H/∂p_μ
    dt_dl = p_t / f
    dr_dl = f * p_r
    dphi_dl = p_phi / r**2

    # Hamilton's equations: dp_μ/dλ = -∂H/∂x^μ
    dp_t_dl = 0.0  # Energy is conserved (t is cyclic)
    dp_r_dl = -(R_S / (2 * r**2)) * (p_t**2 / f**2) \
              - (R_S / (2 * r**2)) * p_r**2 \
              + p_phi**2 / r**3
    dp_phi_dl = 0.0  # Angular momentum is conserved (phi is cyclic)

    return [dt_dl, dr_dl, dphi_dl, dp_t_dl, dp_r_dl, dp_phi_dl]


# =============================================================================
# Initial Conditions
# =============================================================================

def viewing_angle_to_impact_parameter(alpha, r_obs):
    """
    Convert viewing angle to impact parameter.

    Parameters
    ----------
    alpha : float
        Viewing angle in radians. alpha=0 means looking directly at the black hole.
        Positive alpha means looking "to the side".
    r_obs : float
        Observer's radial coordinate.

    Returns
    -------
    b : float
        Impact parameter (b = L/E in geometric units).

    Notes
    -----
    The relationship accounts for gravitational effects on the local frame:
        b = r_obs * sin(alpha) / sqrt(1 - r_s/r_obs)

    For an observer at infinity, this reduces to b = r_obs * sin(alpha).
    """
    f_obs = 1 - R_S / r_obs
    b = r_obs * np.sin(alpha) / np.sqrt(f_obs)
    return b


def initial_conditions_from_angle(r_obs, alpha):
    """
    Generate initial conditions for a photon based on viewing angle.

    The photon starts at the observer's position and is traced backward
    (i.e., in the direction the photon came from to reach the observer's eye).

    Parameters
    ----------
    r_obs : float
        Observer's radial coordinate.
    alpha : float
        Viewing angle in radians from the radial direction toward the black hole.
        alpha=0 means looking directly at the black hole.

    Returns
    -------
    state0 : list or None
        Initial state vector [t, r, phi, p_t, p_r, p_phi], or None if invalid.
    """
    # Convert viewing angle to impact parameter
    b = viewing_angle_to_impact_parameter(alpha, r_obs)

    # Initial position
    t0 = 0.0
    r0 = r_obs
    phi0 = 0.0

    # Metric function at observer
    f0 = 1 - R_S / r0

    # Conserved quantities (set E=1 without loss of generality for null geodesics)
    E = 1.0
    L = b * E  # Angular momentum

    # Momenta
    p_t = E
    p_phi = L

    # Solve null condition for p_r: g^{μν} p_μ p_ν = 0
    # => -p_t²/f + f*p_r² + p_phi²/r² = 0
    # => p_r² = (p_t²/f - p_phi²/r²) / f
    p_r_squared = (p_t**2 / f0 - p_phi**2 / r0**2) / f0

    if p_r_squared < 0:
        return None  # No valid trajectory for this configuration

    # Negative p_r: photon moving inward (toward the black hole)
    # This traces the ray backward from the observer
    p_r = -np.sqrt(p_r_squared)

    return [t0, r0, phi0, p_t, p_r, p_phi]


# =============================================================================
# Ray Integrator
# =============================================================================

def integrate_geodesic(state0, lambda_max=1000.0, r_stop_inner=None, r_stop_outer=None):
    """
    Integrate the geodesic equations from given initial conditions.

    Parameters
    ----------
    state0 : list
        Initial state vector [t, r, phi, p_t, p_r, p_phi].
    lambda_max : float
        Maximum affine parameter for integration.
    r_stop_inner : float, optional
        Stop integration if r falls below this value (default: just above horizon).
    r_stop_outer : float, optional
        Stop integration if r exceeds this value (default: 2x initial radius).

    Returns
    -------
    solution : OdeSolution
        Solution object from scipy.integrate.solve_ivp.
    outcome : str
        'captured' if ray fell into the black hole, 'escaped' if it escaped.
    """
    if r_stop_inner is None:
        r_stop_inner = R_S * 1.01
    if r_stop_outer is None:
        r_stop_outer = state0[1] * 2.0

    # Event: ray captured by black hole
    def event_captured(lambda_, state):
        return state[1] - r_stop_inner
    event_captured.terminal = True
    event_captured.direction = -1  # Trigger when r decreases through threshold

    # Event: ray escaped to large radius
    def event_escaped(lambda_, state):
        return state[1] - r_stop_outer
    event_escaped.terminal = True
    event_escaped.direction = 1  # Trigger when r increases through threshold

    # Integrate
    solution = solve_ivp(
        geodesic_equations,
        [0, lambda_max],
        state0,
        method='RK45',
        events=[event_captured, event_escaped],
        max_step=1.0,
        rtol=1e-8,
        atol=1e-10,
        dense_output=True
    )

    # Determine outcome
    final_r = solution.y[1, -1]
    if final_r <= r_stop_inner * 1.1:
        outcome = 'captured'
    else:
        outcome = 'escaped'

    return solution, outcome


def trace_ray(r_obs, alpha, **kwargs):
    """
    Convenience function to trace a single ray from viewing angle.

    Parameters
    ----------
    r_obs : float
        Observer's radial coordinate.
    alpha : float
        Viewing angle in radians.
    **kwargs : dict
        Additional arguments passed to integrate_geodesic.

    Returns
    -------
    solution : OdeSolution or None
        Solution object, or None if initial conditions are invalid.
    outcome : str
        'captured', 'escaped', or 'invalid'.
    """
    state0 = initial_conditions_from_angle(r_obs, alpha)

    if state0 is None:
        return None, 'invalid'

    return integrate_geodesic(state0, **kwargs)


# =============================================================================
# Visualization
# =============================================================================

def plot_trajectories(r_obs, angles_deg, ax=None):
    """
    Plot photon trajectories for various viewing angles.

    Parameters
    ----------
    r_obs : float
        Observer's radial coordinate.
    angles_deg : array_like
        List of viewing angles in degrees.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, creates new figure.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The axes with the plot.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))

    # Draw black hole and photon sphere
    theta = np.linspace(0, 2 * np.pi, 200)
    ax.fill(R_S * np.cos(theta), R_S * np.sin(theta), 'k', label='Event horizon')
    ax.plot(R_PHOTON * np.cos(theta), R_PHOTON * np.sin(theta),
            'r--', linewidth=1.5, label='Photon sphere')

    # Draw observer position
    ax.plot(r_obs, 0, 'go', markersize=10, label=f'Observer (r={r_obs}M)')

    # Trace and plot each ray
    for alpha_deg in angles_deg:
        alpha = np.radians(alpha_deg)
        solution, outcome = trace_ray(r_obs, alpha)

        if solution is None:
            continue

        # Convert to Cartesian
        r = solution.y[1]
        phi = solution.y[2]
        x = r * np.cos(phi)
        y = r * np.sin(phi)

        color = 'steelblue' if outcome == 'escaped' else 'crimson'
        linestyle = '-' if outcome == 'escaped' else '--'
        ax.plot(x, y, color=color, linestyle=linestyle, linewidth=1.2,
                label=f'α={alpha_deg}° ({outcome})')

    # Critical angle for reference
    alpha_crit = np.degrees(np.arcsin(B_CRIT * np.sqrt(1 - R_S / r_obs) / r_obs))
    ax.set_title(f'Photon trajectories (critical angle ≈ {alpha_crit:.2f}°)')

    ax.set_xlabel('x / M')
    ax.set_ylabel('y / M')
    ax.set_aspect('equal')
    ax.legend(loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3)

    return ax


# =============================================================================
# Main
# =============================================================================

if __name__ == '__main__':
    # Observer parameters
    r_obs = 50.0 * M

    # Viewing angles to trace (in degrees)
    # Critical angle at r_obs=50M is approximately 5.97 degrees
    angles = [0, 2, 4, 5, 5.5, 5.97, 6.5, 8, 10, 15]

    print("=" * 60)
    print("Schwarzschild Black Hole Geodesic Tracer")
    print("=" * 60)
    print(f"Black hole mass:       M = {M}")
    print(f"Schwarzschild radius:  r_s = {R_S} M")
    print(f"Photon sphere:         r_ph = {R_PHOTON} M")
    print(f"Critical impact param: b_crit = {B_CRIT:.4f} M")
    print(f"Observer radius:       r_obs = {r_obs} M")
    print("=" * 60)

    # Calculate and print critical angle for this observer
    alpha_crit = np.arcsin(B_CRIT * np.sqrt(1 - R_S / r_obs) / r_obs)
    print(f"Critical viewing angle: {np.degrees(alpha_crit):.4f}°")
    print("  (rays with |α| < α_crit are captured)")
    print("=" * 60)

    # Trace rays and report outcomes
    print("\nTracing rays:")
    print("-" * 40)
    for alpha_deg in angles:
        alpha = np.radians(alpha_deg)
        b = viewing_angle_to_impact_parameter(alpha, r_obs)
        solution, outcome = trace_ray(r_obs, alpha)

        status = "CAPTURED" if outcome == 'captured' else "ESCAPED"
        print(f"  α = {alpha_deg:6.2f}°  →  b = {b:6.3f} M  →  {status}")

    # Plot
    print("\nGenerating plot...")
    fig, ax = plt.subplots(figsize=(12, 10))
    plot_trajectories(r_obs, angles, ax=ax)

    # Set reasonable axis limits
    ax.set_xlim(-r_obs * 0.3, r_obs * 1.2)
    ax.set_ylim(-r_obs * 0.5, r_obs * 0.5)

    plt.tight_layout()
    plt.savefig('geodesic_trajectories.png', dpi=150, bbox_inches='tight')
    print("Saved: geodesic_trajectories.png")
    plt.show()
