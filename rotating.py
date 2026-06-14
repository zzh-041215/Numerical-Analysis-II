from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from data_input import load_reference_data


@dataclass(frozen=True)
class RotatingSolution:
    """Solution of Lane-Emden equation with uniform rotation (first-order).

    (1/xi^2) d/dxi (xi^2 dtheta/dxi) + theta^n = lambda
    where lambda is the rotation parameter.
    """
    n: float
    lambd: float         # rotation parameter
    xi: np.ndarray
    theta: np.ndarray
    theta_prime: np.ndarray
    first_zero: Optional[float]
    num_steps: int


def solve_rotating_rk4(
    n: float,
    lambd: float = 0.0,
    epsilon: float = 1e-4,
    h: float = 1e-3,
    xi_max: Optional[float] = None,
    max_steps: int = 200_000,
) -> RotatingSolution:
    """Solve the Lane-Emden equation with rotation using RK4.

    The rotation parameter lambda represents the effect of uniform rotation
    at first order. lambda=0 recovers the standard Lane-Emden equation.

    For the centrifugal force: lambda = omega^2 / (2*pi*G*rho_c)

    Parameters
    ----------
    n:
        Polytropic index.
    lambd:
        Rotation parameter. Must satisfy lambda < 1 for a finite star
        (otherwise centrifugal forces exceed gravity).
    """
    if lambd >= 1.0:
        raise ValueError("lambda >= 1: star is unbound (centrifugal disruption).")

    if xi_max is None:
        ref = load_reference_data()
        prop = ref.get_global_property(n)
        # Rotation expands the star, so increase xi_max
        xi_max = (prop.xi_1 or 10.0) * (1.0 + lambd) + h if prop else 10.0

    # Taylor initial conditions (rotation modifies leading coefficients)
    # theta(ξ) = 1 - (1-λ)ξ²/6 + n(1-λ)ξ⁴/120 + ...
    # theta'(ξ) = -(1-λ)ξ/3 + n(1-λ)ξ³/30 + ...
    theta0 = 1.0 - (1.0 - lambd) * epsilon**2 / 6.0 + n * (1.0 - lambd) * epsilon**4 / 120.0
    theta_prime0 = -(1.0 - lambd) * epsilon / 3.0 + n * (1.0 - lambd) * epsilon**3 / 30.0

    xi_list = [epsilon]
    theta_list = [theta0]
    tp_list = [theta_prime0]

    xi = epsilon
    state = np.array([theta0, theta_prime0], dtype=float)
    first_zero = None
    step_count = 0

    while xi < xi_max and step_count < max_steps:
        step = min(h, xi_max - xi)

        def rhs(xi_s, s):
            t, tp = s
            if t < 0 and not float(n).is_integer():
                t_pow = 0.0
            else:
                t_pow = t ** n
            return np.array([tp, -2.0 * tp / xi_s - t_pow + lambd])

        k1 = rhs(xi, state)
        k2 = rhs(xi + step / 2, state + step * k1 / 2)
        k3 = rhs(xi + step / 2, state + step * k2 / 2)
        k4 = rhs(xi + step, state + step * k3)

        new_state = state + step * (k1 + 2 * k2 + 2 * k3 + k4) / 6
        new_xi = xi + step
        step_count += 1

        if state[0] > 0 and new_state[0] <= 0:
            w = abs(state[0]) / (abs(state[0]) + abs(new_state[0]))
            first_zero = float(xi + w * step)
            xi_list.append(first_zero)
            theta_list.append(0.0)
            tp_list.append(float(new_state[1]))
            break

        xi_list.append(new_xi)
        theta_list.append(float(new_state[0]))
        tp_list.append(float(new_state[1]))

        xi = new_xi
        state = new_state

    return RotatingSolution(
        n=n, lambd=lambd, xi=np.array(xi_list),
        theta=np.array(theta_list), theta_prime=np.array(tp_list),
        first_zero=first_zero, num_steps=step_count,
    )


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from pathlib import Path

    print("=" * 70)
    print("Rotating Polytrope Solver")
    print("=" * 70)

    # Effect of rotation on stellar radius
    print("\n--- Effect of rotation on stellar radius (n=1.5) ---")
    n_test = 1.5
    ref = load_reference_data()
    prop = ref.get_global_property(n_test)
    xi_ref = prop.xi_1 if prop else 3.654

    for lambd_val in (0.0, 0.05, 0.1, 0.15, 0.2, 0.25):
        sol = solve_rotating_rk4(n=n_test, lambd=lambd_val,
                                  epsilon=1e-4, h=2e-3)
        if sol.first_zero:
            ratio = sol.first_zero / xi_ref
            print(f"  lambda={lambd_val:.2f}: xi_1={sol.first_zero:.6f}, "
                  f"R/R_0={ratio:.4f}")

    # Plot
    output_dir = Path(__file__).resolve().parent / "output_initial"
    output_dir.mkdir(exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Density profiles for different rotation rates
    for lambd_val in (0.0, 0.1, 0.2, 0.3):
        sol = solve_rotating_rk4(n=1.5, lambd=lambd_val,
                                  epsilon=1e-4, h=2e-3)
        ax1.plot(sol.xi, sol.theta, linewidth=1.5,
                 label=f"lambda={lambd_val:.1f}")
    ax1.set_xlabel(r"$\xi$")
    ax1.set_ylabel(r"$\theta(\xi)$")
    ax1.set_title("Rotating Polytrope: Density Profiles (n=1.5)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Radius vs rotation
    lambd_vals = np.linspace(0, 0.4, 15)
    for n_val in (0.5, 1.5, 3.0):
        radii = []
        valid_lambdas = []
        for lam in lambd_vals:
            try:
                sol = solve_rotating_rk4(n=n_val, lambd=lam,
                                          epsilon=1e-4, h=2e-3)
                if sol.first_zero:
                    valid_lambdas.append(lam)
                    radii.append(sol.first_zero)
            except Exception:
                continue
        if radii:
            ax2.plot(valid_lambdas, radii, "o-", linewidth=1.2,
                     label=f"n={n_val:g}")
    ax2.set_xlabel(r"$\lambda$ (rotation)")
    ax2.set_ylabel(r"$\xi_1$ (stellar radius)")
    ax2.set_title("Stellar Radius vs Rotation Rate")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "rotating_overview.png", dpi=200)
    plt.close()
    print(f"\nPlot saved to {output_dir / 'rotating_overview.png'}")
