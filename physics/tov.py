from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import sys
from pathlib import Path

import numpy as np

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

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

    # Plot (2x2 layout)
    output_dir = Path(__file__).resolve().parent.parent / "output_initial"
    output_dir.mkdir(exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))

    # --- Top-Left: Density profiles for different sigma ---
    ax1 = axes[0, 0]
    n_plot = 1.5
    sigma_colors = {0.0: "blue", 0.1: "green", 0.2: "orange", 0.3: "red"}
    for s in (0.0, 0.1, 0.2, 0.3):
        sol = solve_tov_rk4(n=n_plot, sigma=s, epsilon=1e-4, h=2e-3)
        ax1.plot(sol.xi, sol.theta, linewidth=1.5, color=sigma_colors[s],
                 label=f"σ={s:.1f}")
    ax1.set_xlabel(r"$\xi$")
    ax1.set_ylabel(r"$\theta(\xi)$")
    ax1.set_title(f"Density Profiles: n={n_plot} at various σ")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.annotate(
        r"$\sigma = P_c / (\rho_c c^2)$" + "\n"
        r"$\sigma = 0$ → Newtonian LE" + "\n"
        r"$\sigma > 0$ → GR correction" + "\n"
        "Larger σ → more compact star",
        xy=(0.97, 0.97), xycoords="axes fraction", fontsize=8,
        ha="right", va="top",
        bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

    # --- Top-Right: Mass-Radius curves ---
    ax2 = axes[0, 1]
    sigma_vals = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]
    for n_mr in (1.0, 1.5, 2.0):
        sigs, rads, masses = mass_radius_curve(n_mr, sigma_vals, epsilon=1e-4, h=2e-3)
        if rads and masses:
            ax2.plot(rads, masses, "o-", linewidth=1.2, label=f"n={n_mr:g}")
            # Annotate σ direction on the last curve
            if n_mr == 1.5 and len(rads) >= 2:
                mid = len(rads) // 2
                ax2.annotate("σ↑", (rads[mid], masses[mid]),
                             fontsize=9, color="red",
                             arrowprops=dict(arrowstyle="->", color="red"))
    ax2.set_xlabel("Radius (dimensionless)")
    ax2.set_ylabel("Mass (dimensionless)")
    ax2.set_title("Mass-Radius Relation")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    ax2.annotate("σ increases →\nstar becomes more compact\n(mass increases at fixed radius)",
                 xy=(0.03, 0.97), xycoords="axes fraction", fontsize=8,
                 va="top", bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

    # --- Bottom-Left: Newtonian limit comparison ---
    ax3 = axes[1, 0]
    ax3.axis("off")
    ref = load_reference_data()
    table_data = []
    for n_val in (0.0, 1.0, 3.0):
        sol_tov = solve_tov_rk4(n=n_val, sigma=0.0, epsilon=1e-4, h=1e-3)
        prop = ref.get_global_property(n_val)
        tov_xi1 = f"{sol_tov.first_zero:.6f}" if sol_tov.first_zero else "N/A"
        le_xi1 = f"{prop.xi_1:.6f}" if prop and prop.xi_1 else "N/A"
        tov_mass = f"{sol_tov.total_mass:.6f}" if sol_tov.total_mass else "N/A"
        le_mass = f"{prop.mass:.6f}" if prop and prop.mass else "N/A"
        table_data.append([f"n={n_val:g}", tov_xi1, le_xi1, tov_mass, le_mass])

    col_labels = ["n", "TOV ξ₁", "LE ξ₁", "TOV mass", "LE mass"]
    table = ax3.table(cellText=table_data, colLabels=col_labels,
                      cellLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.8)
    ax3.set_title("Newtonian Limit (σ=0): TOV vs Standard LE", fontsize=11,
                  fontweight="bold")

    # --- Bottom-Right: Physical explanation ---
    ax4 = axes[1, 1]
    ax4.axis("off")
    explanation = (
        "TOV Equations (Relativistic Polytropes)\n\n"
        "Equations (dimensionless):\n"
        "  dθ/dξ = -(θⁿ+1)(m+σξ³θⁿ) / [ξ²(1-2σm/ξ)]\n"
        "  dm/dξ = ξ² θⁿ\n\n"
        "Parameters:\n"
        "  σ = P_c/(ρ_c c²)  — relativistic factor\n"
        "  σ=0 → Newtonian Lane-Emden\n"
        "  σ~0.3-0.5 → neutron star regime\n\n"
        "Key Physics:\n"
        "  - GR gravity is stronger → stars are\n"
        "    more compact at same central density\n"
        "  - There exists a maximum mass for\n"
        "    stable configurations (Chandrasekhar limit)\n"
        "  - n=3, σ→critical → mass approaches constant\n"
        "    (independent of central density)"
    )
    ax4.text(0.1, 0.5, explanation, transform=ax4.transAxes, fontsize=9,
             verticalalignment="center", fontfamily="monospace",
             bbox=dict(boxstyle="round", facecolor="lightcyan", alpha=0.8))

    fig.suptitle("TOV Equation: Relativistic Polytropes", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / "tov_overview.png", dpi=200)
    plt.close()
    print(f"\nPlot saved to {output_dir / 'tov_overview.png'}")
