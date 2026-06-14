from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from data_input import load_reference_data


@dataclass(frozen=True)
class ContinuedSolution:
    """Lane-Emden solution continued beyond the first zero."""
    n: float
    xi: np.ndarray
    theta: np.ndarray
    theta_prime: np.ndarray
    zeros: list[float]    # positions of sign changes
    num_zeros: int


def solve_continued_rk4(
    n: float,
    epsilon: float = 1e-4,
    h: float = 1e-3,
    xi_max: float = 30.0,
    max_zeros: int = 10,
    max_steps: int = 500_000,
) -> ContinuedSolution:
    """Solve Lane-Emden beyond the first zero using RK4.

    For non-integer n, we use |theta|^n * sign(theta) to avoid complex values
    when theta becomes negative (physical analytic continuation).

    Parameters
    ----------
    n:
        Polytropic index. Must be < 5 for finite first zero.
    max_zeros:
        Maximum number of zero crossings to track.
    xi_max:
        Maximum xi to integrate to.
    """
    if n >= 5.0:
        raise ValueError("n >= 5: no finite first zero.")

    # Taylor initial conditions
    theta0 = 1.0 - epsilon**2 / 6.0 + n * epsilon**4 / 120.0
    theta_prime0 = -epsilon / 3.0 + n * epsilon**3 / 30.0

    xi_list = [epsilon]
    theta_list = [theta0]
    tp_list = [theta_prime0]

    xi = epsilon
    state = np.array([theta0, theta_prime0], dtype=float)
    zeros = []
    prev_sign = 1.0  # theta starts positive
    step_count = 0

    while xi < xi_max and step_count < max_steps:
        step = min(h, xi_max - xi)

        def rhs(xi_s, s):
            t, tp = s
            # Physical continuation for non-integer n:
            # theta^n -> |theta|^n * sign(theta)
            if float(n).is_integer():
                t_pow = t ** n
            else:
                t_pow = abs(t) ** n * np.sign(t) if t != 0 else 0.0
            return np.array([tp, -2.0 * tp / xi_s - t_pow])

        k1 = rhs(xi, state)
        k2 = rhs(xi + step / 2, state + step * k1 / 2)
        k3 = rhs(xi + step / 2, state + step * k2 / 2)
        k4 = rhs(xi + step, state + step * k3)

        new_state = state + step * (k1 + 2 * k2 + 2 * k3 + k4) / 6
        new_xi = xi + step
        step_count += 1

        # Detect zero crossings
        if state[0] * new_state[0] < 0:
            w = abs(state[0]) / (abs(state[0]) + abs(new_state[0]))
            zero_xi = xi + w * step
            zeros.append(float(zero_xi))
            if len(zeros) >= max_zeros:
                xi_list.append(zero_xi)
                theta_list.append(0.0)
                tp_list.append(float(new_state[1]))
                break

        xi_list.append(new_xi)
        theta_list.append(float(new_state[0]))
        tp_list.append(float(new_state[1]))

        xi = new_xi
        state = new_state

    return ContinuedSolution(
        n=n,
        xi=np.array(xi_list),
        theta=np.array(theta_list),
        theta_prime=np.array(tp_list),
        zeros=zeros,
        num_zeros=len(zeros),
    )


def asymptotic_behavior(n: float, xi: float) -> str:
    """Describe asymptotic behavior at large xi for given n.

    n < 5: finite radius, oscillatory beyond first zero (for integer n)
    n = 5: theta ~ 1/xi as xi -> infinity, never crosses zero
    n > 5: theta ~ C * xi^{(2-n)/(n-1)}, infinite radius, physically uninteresting
    """
    if n < 5.0:
        return f"n={n:g} < 5: finite radius, theta(xi_1)=0"
    elif abs(n - 5.0) < 1e-10:
        return f"n=5: infinite radius, theta ~ 1/xi asymptotically"
    else:
        return f"n={n:g} > 5: infinite radius, algebraically decaying"


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from pathlib import Path

    print("=" * 70)
    print("Beyond-First-Zero Continuation")
    print("=" * 70)

    for n_val in (0.0, 1.0, 2.0):
        print(f"\n--- n = {n_val:g} ---")
        print(f"  {asymptotic_behavior(n_val, 0.0)}")

        sol = solve_continued_rk4(n_val, epsilon=1e-4, h=1e-2,
                                  xi_max=25.0, max_zeros=5)
        print(f"  Number of zeros found: {sol.num_zeros}")
        for i, z in enumerate(sol.zeros):
            print(f"    Zero {i+1}: xi = {z:.8f}")
        print(f"  Final theta(xi={sol.xi[-1]:.1f}) = {sol.theta[-1]:.6f}")

    # Plot
    output_dir = Path(__file__).resolve().parent / "output_initial"
    output_dir.mkdir(exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    for n_val in (0.0, 1.0, 2.0):
        sol = solve_continued_rk4(n_val, epsilon=1e-4, h=5e-3,
                                  xi_max=20.0, max_zeros=4)
        ax.plot(sol.xi, sol.theta, linewidth=1.2, label=f"n={n_val:g}")
        for z in sol.zeros:
            ax.axvline(z, color="gray", alpha=0.2, linewidth=0.5)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xlabel(r"$\xi$")
    ax.set_ylabel(r"$\theta(\xi)$")
    ax.set_title("Lane-Emden: Beyond First Zero")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Zero positions vs n
    ax = axes[1]
    n_values = np.linspace(0.0, 4.5, 10)
    for i_zero in range(3):
        zero_positions = []
        for n_val in n_values:
            try:
                sol = solve_continued_rk4(n_val, epsilon=1e-4, h=1e-2,
                                          xi_max=50.0, max_zeros=i_zero + 1)
                if len(sol.zeros) > i_zero:
                    zero_positions.append(sol.zeros[i_zero])
                else:
                    zero_positions.append(np.nan)
            except Exception:
                zero_positions.append(np.nan)
        ax.plot(n_values, zero_positions, "o-", linewidth=1.2,
                label=f"Zero {i_zero + 1}")

    ax.set_xlabel(r"$n$")
    ax.set_ylabel(r"$\xi$ (zero position)")
    ax.set_title("Zero Positions vs Polytropic Index")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "continuation_overview.png", dpi=200)
    plt.close()
    print(f"\nPlot saved to {output_dir / 'continuation_overview.png'}")
