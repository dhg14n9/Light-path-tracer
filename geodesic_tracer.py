"""
Geodesic ray tracer for arbitrary spacetime metrics.

Generic integration and visualization routines. Metric-specific physics
(equations of motion, initial conditions, orbit shortcuts) live in
metrics.py.

Units: Geometric units where G = c = 1.
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

from metrics import Schwarzschild


# =============================================================================
# Ray Integrator
# =============================================================================

def integrate_geodesic(metric, state0, lambda_max=1000.0,
                       r_stop_inner=None, r_stop_outer=None):
    """
    Integrate the geodesic equations from given initial conditions.

    Parameters
    ----------
    metric : Metric
        Spacetime metric providing geodesic_equations and capture_radius.
    state0 : list
        Initial 8D state vector.
    lambda_max : maximum affine parameter.
    r_stop_inner : stop if r falls below this (default: metric.capture_radius()).
    r_stop_outer : stop if r exceeds this (default: 2x initial radius).

    Returns
    -------
    solution : OdeSolution
    outcome : 'captured' or 'escaped'
    """
    if r_stop_inner is None:
        r_stop_inner = metric.capture_radius()
    if r_stop_outer is None:
        r_stop_outer = state0[1] * 2.0

    def event_captured(lambda_, state):
        return state[1] - r_stop_inner
    event_captured.terminal = True
    event_captured.direction = -1

    def event_escaped(lambda_, state):
        return state[1] - r_stop_outer
    event_escaped.terminal = True
    event_escaped.direction = 1

    solution = solve_ivp(
        metric.geodesic_equations,
        [0, lambda_max],
        state0,
        method='RK45',
        events=[event_captured, event_escaped],
        max_step=1.0,
        rtol=1e-8,
        atol=1e-10,
        dense_output=True,
    )

    final_r = solution.y[1, -1]
    outcome = 'captured' if final_r <= r_stop_inner * 1.1 else 'escaped'
    return solution, outcome


def trace_ray(metric, r_obs, alpha, **kwargs):
    """Trace a single ray from viewing angle using the full Hamiltonian.

    Returns (solution, outcome) or (None, 'invalid').
    """
    state0 = metric.initial_conditions(r_obs, alpha)
    if state0 is None:
        return None, 'invalid'
    return integrate_geodesic(metric, state0, **kwargs)


# =============================================================================
# Visualization
# =============================================================================

def plot_trajectories(metric, r_obs, angles_deg, ax=None):
    """
    Plot photon trajectories for various viewing angles.

    Parameters
    ----------
    metric : Metric
    r_obs : observer radial coordinate
    angles_deg : list of viewing angles in degrees
    ax : matplotlib axes (optional)
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))

    # Draw black hole and photon sphere
    theta = np.linspace(0, 2 * np.pi, 200)
    r_horizon = metric.capture_radius()
    ax.fill(r_horizon * np.cos(theta), r_horizon * np.sin(theta),
            'k', label='Event horizon')

    if hasattr(metric, 'R_PHOTON'):
        r_ph = metric.R_PHOTON
        ax.plot(r_ph * np.cos(theta), r_ph * np.sin(theta),
                'r--', linewidth=1.5, label='Photon sphere')

    ax.plot(r_obs, 0, 'go', markersize=10, label=f'Observer (r={r_obs}M)')

    for alpha_deg in angles_deg:
        alpha = np.radians(alpha_deg)
        solution, outcome = trace_ray(metric, r_obs, alpha)

        if solution is None:
            continue

        r = solution.y[1]
        phi = solution.y[3]  # 8D: phi is index 3
        x = r * np.cos(phi)
        y = r * np.sin(phi)

        color = 'steelblue' if outcome == 'escaped' else 'crimson'
        linestyle = '-' if outcome == 'escaped' else '--'
        ax.plot(x, y, color=color, linestyle=linestyle, linewidth=1.2,
                label=f'α={alpha_deg}° ({outcome})')

    alpha_crit = np.degrees(metric.alpha_crit(r_obs))
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
    metric = Schwarzschild(M=1.0)
    r_obs = 50.0 * metric.M

    angles = [0, 2, 4, 5, 5.5, 5.97, 6.5, 8, 10, 15]

    print("=" * 60)
    print("Geodesic Tracer")
    print("=" * 60)
    print(f"Metric: {type(metric).__name__}")
    print(f"Observer radius: r_obs = {r_obs} M")

    ac = metric.alpha_crit(r_obs)
    print(f"Critical viewing angle: {np.degrees(ac):.4f}°")
    print("=" * 60)

    print("\nTracing rays:")
    print("-" * 40)
    for alpha_deg in angles:
        alpha = np.radians(alpha_deg)
        b = metric.viewing_angle_to_impact_parameter(alpha, r_obs)
        solution, outcome = trace_ray(metric, r_obs, alpha)
        status = "CAPTURED" if outcome == 'captured' else "ESCAPED"
        print(f"  α = {alpha_deg:6.2f}°  →  b = {b:6.3f} M  →  {status}")

    print("\nGenerating plot...")
    fig, ax = plt.subplots(figsize=(12, 10))
    plot_trajectories(metric, r_obs, angles, ax=ax)

    ax.set_xlim(-r_obs * 0.3, r_obs * 1.2)
    ax.set_ylim(-r_obs * 0.5, r_obs * 0.5)

    plt.tight_layout()
    plt.savefig('geodesic_trajectories.png', dpi=150, bbox_inches='tight')
    print("Saved: geodesic_trajectories.png")
    plt.show()
