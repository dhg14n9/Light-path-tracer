"""
Example script demonstrating geodesic tracing around a Schwarzschild black hole.
"""

import numpy as np
import matplotlib.pyplot as plt

from geodesic_tracer import (
    M, R_S, R_PHOTON, B_CRIT,
    trace_ray, viewing_angle_to_impact_parameter
)


def main():
    # Observer position
    r_obs = 50.0 * M

    # Choose a viewing angle (in degrees)
    alpha_deg = 8.0
    alpha = np.radians(alpha_deg)

    # Trace the ray
    solution, outcome = trace_ray(r_obs, alpha)

    # Print results
    b = viewing_angle_to_impact_parameter(alpha, r_obs)
    print(f"Observer radius:    r_obs = {r_obs} M")
    print(f"Viewing angle:      α = {alpha_deg}°")
    print(f"Impact parameter:   b = {b:.4f} M")
    print(f"Critical b:         b_crit = {B_CRIT:.4f} M")
    print(f"Outcome:            {outcome.upper()}")

    # Extract trajectory
    r = solution.y[1] 
    phi = solution.y[2]
    x = r * np.cos(phi)
    y = r * np.sin(phi)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 10))

    # Black hole
    theta = np.linspace(0, 2 * np.pi, 200)
    ax.fill(R_S * np.cos(theta), R_S * np.sin(theta), 'k', label='Event horizon')
    ax.plot(R_PHOTON * np.cos(theta), R_PHOTON * np.sin(theta),
            'r--', linewidth=1.5, label='Photon sphere')

    # Trajectory
    color = 'steelblue' if outcome == 'escaped' else 'crimson'
    ax.plot(x, y, color=color, linewidth=2, label=f'Photon path ({outcome})')

    # Observer
    ax.plot(r_obs, 0, 'go', markersize=12, label='Observer')

    # Formatting
    ax.set_xlabel('x / M', fontsize=12)
    ax.set_ylabel('y / M', fontsize=12)
    ax.set_title(f'Geodesic trajectory (α = {alpha_deg}°, b = {b:.2f} M)', fontsize=14)

    # Set symmetric axis limits centered on the black hole
    limit = r_obs * 1.1
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_aspect('equal')

    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('example_geodesic.png', dpi=150)
    print(f"\nSaved: example_geodesic.png")
    plt.show()


if __name__ == '__main__':
    main()
