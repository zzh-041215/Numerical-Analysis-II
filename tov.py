from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from data_input import load_reference_data


@dataclass(frozen=True)
class TOVSolution:
    """Solution of the relativistic TOV equations for a polytropic EOS."""
    n: float
    sigma: float          # relativistic parameter
    xi: np.ndarray
    theta: np.ndarray
    mass: np.ndarray      # dimensionless enclosed mass m(xi)
    first_zero: Optional[float]
    total_mass: Optional[float]  # m(xi_1) = dimensionless total mass
    num_steps: int


def solve_tov_rk4(
    n: float,
    sigma: float = 0.0,
    epsilon: float = 1e-4,
    h: float = 1e-3,
    xi_max: Optional[float] = None,
    max_steps: int = 200_000,
) -> TOVSolution:
    """Solve TOV equations with polytropic EOS using RK4.

    Dimensionless TOV equations:
        dtheta/dxi = -(theta^n + 1)(m + sigma * xi^3 * theta^n)
                      / (xi^2 * (1 - 2*sigma*m/xi))
        dm/dxi = xi^2 * theta^n

    When sigma=0, reduces to standard Lane-Emden.

    Parameters
    ----------
    n:
        Polytropic index.
    sigma:
        Relativistic parameter. sigma = P_c/(rho_c * c^2) at center.
        sigma -> 0 recovers Newtonian (standard Lane-Emden).
        Typical values: 0 (Newtonian) to ~0.5 (strongly relativistic).
    epsilon:
        Starting point.
    h:
        Step size.
    xi_max:
        Maximum xi for integration.

    Returns
    -------
    TOVSolution.
    """
    if sigma < 0:
        raise ValueError("sigma must be non-negative.")

    if xi_max is None:
        ref = load_reference_data()
        prop = ref.get_global_property(n)
        xi_max = (prop.xi_1 or 10.0) * (1.0 + sigma) if prop else 10.0

    # Taylor initial conditions (generalized for TOV)
    # Near xi=0: theta = 1 - xi^2/6 + ..., m = xi^3/3 + ...
    theta0 = 1.0 - epsilon**2 / 6.0 + n * epsilon**4 / 120.0
    m0 = epsilon**3 / 3.0

    xi_list = [epsilon]
    theta_list = [theta0]
    m_list = [m0]

    xi = epsilon
    state = np.array([theta0, m0], dtype=float)
    first_zero = None
    total_mass = None
    step_count = 0

    while xi < xi_max and step_count < max_steps:
        step = min(h, xi_max - xi)

        def tov_rhs(xi_s, s):
            th, m = s
            if th <= 0:
                return np.array([0.0, 0.0])

            th_n = th ** n
            denom = xi_s ** 2 * (1.0 - 2.0 * sigma * m / xi_s)

            if denom <= 0:
                return np.array([-1e10, 0.0])

            dth = -(m + sigma * xi_s ** 3 * th_n) / denom
            dm = xi_s ** 2 * th_n

            return np.array([dth, dm])

        k1 = tov_rhs(xi, state)
        k2 = tov_rhs(xi + step / 2, state + step * k1 / 2)
        k3 = tov_rhs(xi + step / 2, state + step * k2 / 2)
        k4 = tov_rhs(xi + step, state + step * k3)

        new_state = state + step * (k1 + 2 * k2 + 2 * k3 + k4) / 6
        new_xi = xi + step
        step_count += 1

        # Check for surface (theta=0) or collapse
        if state[0] > 0 and new_state[0] <= 0:
            w = abs(state[0]) / (abs(state[0]) + abs(new_state[0]))
            first_zero = float(xi + w * step)
            total_mass = float(state[1] + w * (new_state[1] - state[1]))
            xi_list.append(first_zero)
            theta_list.append(0.0)
            m_list.append(total_mass)
            break

        # Check for gravitational collapse (denom <= 0 in rhs)
        if new_state[0] < -1e5:
            break

        xi_list.append(new_xi)
        theta_list.append(float(new_state[0]))
        m_list.append(float(new_state[1]))

        xi = new_xi
        state = new_state

    return TOVSolution(
        n=n, sigma=sigma, xi=np.array(xi_list),
        theta=np.array(theta_list), mass=np.array(m_list),
        first_zero=first_zero, total_mass=total_mass,
        num_steps=step_count,
    )


def mass_radius_curve(
    n: float,
    sigma_values: list[float],
    epsilon: float = 1e-4,
    h: float = 2e-3,
) -> tuple[list[float], list[float], list[float]]:
    """Compute the mass-radius curve for a range of sigma values.

    Returns (sigma_list, radius_list, mass_list).
    """
    radii = []
    masses = []
    valid_sigmas = []

    for sigma in sigma_values:
        try:
            sol = solve_tov_rk4(n=n, sigma=sigma, epsilon=epsilon, h=h)
            if sol.first_zero is not None and sol.total_mass is not None:
                valid_sigmas.append(sigma)
                radii.append(sol.first_zero)
                masses.append(sol.total_mass)
        except Exception:
            continue

    return valid_sigmas, radii, masses


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from pathlib import Path

    print("=" * 70)
    print("TOV Equation Solver (Relativistic Polytropes)")
    print("=" * 70)

    # Compare Newtonian (sigma=0) with standard LE
    print("\n--- Newtonian limit verification (sigma=0 vs standard LE) ---")
    for n_val in (0.0, 1.0, 3.0):
        sol_tov = solve_tov_rk4(n=n_val, sigma=0.0, epsilon=1e-4, h=1e-3)
        ref = load_reference_data()
        prop = ref.get_global_property(n_val)

        print(f"  n={n_val:g}:")
        if sol_tov.first_zero is not None:
            print(f"    TOV xi_1 = {sol_tov.first_zero:.8f}")
            print(f"    TOV mass = {sol_tov.total_mass:.8f}")
        else:
            print(f"    TOV: no finite surface found")
        print(f"    LE  xi_1 = {prop.xi_1:.8f}" if prop else "    LE  xi_1 = N/A")
        print(f"    LE  mass = {prop.mass:.8f}" if prop else "    LE  mass = N/A")

    # Mass-radius curve for n=1.5 (non-relativistic degenerate)
    print("\n--- Mass-Radius curve for n=1.5 (white dwarf-like) ---")
    sigma_vals = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]
    sigmas, radii, masses = mass_radius_curve(1.5, sigma_vals)
    for s, r, m in zip(sigmas, radii, masses):
        print(f"  sigma={s:.2f}: R={r:.4f}, M={m:.4f}")

    # Plot
    output_dir = Path(__file__).resolve().parent / "output_initial"
    output_dir.mkdir(exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Density profiles for different sigma
    n_plot = 1.5
    for s in (0.0, 0.1, 0.2, 0.3):
        sol = solve_tov_rk4(n=n_plot, sigma=s, epsilon=1e-4, h=2e-3)
        ax1.plot(sol.xi, sol.theta, linewidth=1.5,
                 label=f"sigma={s:.1f}")
    ax1.set_xlabel(r"$\xi$")
    ax1.set_ylabel(r"$\theta(\xi)$")
    ax1.set_title(f"TOV: Density profiles, n={n_plot}")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Mass-radius curves
    for n_mr in (1.0, 1.5, 2.0):
        sigs, rads, mass = mass_radius_curve(n_mr, sigma_vals,
                                              epsilon=1e-4, h=2e-3)
        if rads and mass:
            ax2.plot(rads, mass, "o-", linewidth=1.2,
                     label=f"n={n_mr:g}")
    ax2.set_xlabel("Radius (dimensionless)")
    ax2.set_ylabel("Mass (dimensionless)")
    ax2.set_title("TOV Mass-Radius Relations")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "tov_overview.png", dpi=200)
    plt.close()
    print(f"\nPlot saved to {output_dir / 'tov_overview.png'}")
