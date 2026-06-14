from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from data_input import load_reference_data


@dataclass(frozen=True)
class AdaptiveRKSolution:
    """Solution of the Lane-Emden equation by adaptive-step Runge-Kutta."""
    n: float
    xi: np.ndarray
    theta: np.ndarray
    theta_prime: np.ndarray
    first_zero: Optional[float]
    num_steps: int
    num_rhs_evals: int
    accepted_steps: int
    rejected_steps: int
    min_h_used: float
    max_h_used: float


# Dormand-Prince 5(4) Butcher tableau
# The 7-stage embedded pair with orders 5 and 4
DP_A = np.array([
    [0, 0, 0, 0, 0, 0],
    [1/5, 0, 0, 0, 0, 0],
    [3/40, 9/40, 0, 0, 0, 0],
    [44/45, -56/15, 32/9, 0, 0, 0],
    [19372/6561, -25360/2187, 64448/6561, -212/729, 0, 0],
    [9017/3168, -355/33, 46732/5247, 49/176, -5103/18656, 0],
], dtype=float)

DP_B5 = np.array([35/384, 0, 500/1113, 125/192, -2187/6784, 11/84, 0], dtype=float)  # 5th order
DP_B4 = np.array([5179/57600, 0, 7571/16695, 393/640, -92097/339200, 187/2100, 1/40], dtype=float)  # 4th order
DP_C = np.array([0, 1/5, 3/10, 4/5, 8/9, 1, 1], dtype=float)


def taylor_initial_conditions(n: float, epsilon: float) -> tuple[float, float]:
    """Approximate theta(epsilon) and theta'(epsilon) near xi=0 using Taylor expansion."""
    if epsilon <= 0:
        raise ValueError("epsilon must be positive.")
    theta = 1.0 - epsilon**2 / 6.0 + n * epsilon**4 / 120.0
    theta_prime = -epsilon / 3.0 + n * epsilon**3 / 30.0
    return theta, theta_prime


def lane_emden_rhs(xi: float, state: np.ndarray, n: float) -> np.ndarray:
    """Right-hand side of the first-order Lane-Emden system."""
    if xi <= 0:
        raise ValueError("xi must be positive; start from epsilon > 0.")
    theta, theta_prime = state
    if theta < 0.0 and not float(n).is_integer():
        theta_power = 0.0  # avoid complex; signal will be caught upstream
    else:
        theta_power = theta ** n
    return np.array([theta_prime, -2.0 * theta_prime / xi - theta_power], dtype=float)


def _dp_step(xi: float, state: np.ndarray, h: float, n: float) -> tuple[np.ndarray, np.ndarray, float]:
    """One Dormand-Prince step returning (y5, y4, error_estimate).

    Implements the Dormand-Prince 5(4) embedded Runge-Kutta pair.
    k_i = f(xi + c_i*h, y + h * sum_j a_{ij} * k_j)  for i = 0..6
    y5 = y + h * sum_i b5_i * k_i    (5th order)
    y4 = y + h * sum_i b4_i * k_i    (4th order)
    """
    k = np.zeros((7, 2), dtype=float)

    # k0 = f(xi, state)  (c0 = 0)
    k[0] = lane_emden_rhs(xi, state, n)

    for i in range(1, 7):
        xi_stage = xi + DP_C[i] * h
        state_stage = state.copy()
        for j in range(i):
            state_stage += h * DP_A[i - 1, j] * k[j]
        k[i] = lane_emden_rhs(xi_stage, state_stage, n)

    y5 = state.copy()
    y4 = state.copy()
    for i in range(7):
        y5 += h * DP_B5[i] * k[i]
        y4 += h * DP_B4[i] * k[i]

    err = float(np.max(np.abs(y5 - y4)))
    return y5, y4, err


def _linear_first_zero(xi_left: float, theta_left: float, xi_right: float, theta_right: float) -> float:
    """Linear interpolation of theta's first zero."""
    denom = abs(theta_left) + abs(theta_right)
    if denom == 0:
        return xi_left
    return xi_left + abs(theta_left) / denom * (xi_right - xi_left)


def _linear_first_zero_from_slope(xi: float, theta: float, theta_prime: float) -> float:
    """Extrapolate to first zero using slope."""
    if theta_prime >= 0:
        return xi
    return xi - theta / theta_prime


def solve_lane_emden_adaptive(
    n: float,
    epsilon: float = 1e-6,
    h0: float = 1e-3,
    tolerance: float = 1e-8,
    xi_max: Optional[float] = None,
    stop_at_zero: bool = True,
    safety_factor: float = 0.9,
    h_min: float = 1e-12,
    h_max: float = 1.0,
    max_steps: int = 1_000_000,
    zero_tolerance: float = 1e-10,
) -> AdaptiveRKSolution:
    """Solve the Lane-Emden equation using adaptive-step Dormand-Prince 5(4).

    Parameters
    ----------
    n:
        Polytropic index, 0 <= n <= 5.
    epsilon:
        Starting point near xi=0.
    h0:
        Initial step size.
    tolerance:
        Local error tolerance per step.
    xi_max:
        Maximum xi for integration. If None, uses reference first zero.
    stop_at_zero:
        Stop integration when theta crosses zero.
    safety_factor:
        Safety factor for step size update (typical: 0.8-0.9).
    h_min, h_max:
        Bounds on step size.
    max_steps:
        Maximum number of steps.
    zero_tolerance:
        Threshold for detecting near-zero theta.

    Returns
    -------
    AdaptiveRKSolution with the full solution trajectory.
    """
    if not (0.0 <= n <= 5.0):
        raise ValueError("n must satisfy 0 <= n <= 5.")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive.")

    reference_data = load_reference_data()
    reference_zero = reference_data.get_first_zero(n)

    if xi_max is None:
        if reference_zero is not None:
            xi_max = reference_zero + h_max
        else:
            xi_max = 10.0
    if xi_max <= epsilon:
        raise ValueError("xi_max must be greater than epsilon.")

    theta0, theta_prime0 = taylor_initial_conditions(n, epsilon)

    # Storage for trajectory
    xi_list = [epsilon]
    theta_list = [theta0]
    theta_prime_list = [theta_prime0]

    xi = epsilon
    state = np.array([theta0, theta_prime0], dtype=float)
    h = h0
    first_zero: Optional[float] = None

    step_count = 0
    accepted = 0
    rejected = 0
    num_rhs_evals = 0
    min_h = h0
    max_h = h0

    while xi < xi_max and step_count < max_steps:
        # Clamp step to not overshoot xi_max
        if xi + h > xi_max:
            h = xi_max - xi

        # Check for near-zero theta before stepping
        if stop_at_zero and 0.0 < state[0] <= zero_tolerance:
            first_zero = _linear_first_zero_from_slope(xi, state[0], state[1])
            xi_list.append(first_zero)
            theta_list.append(0.0)
            theta_prime_list.append(float(state[1]))
            break

        # Attempt step
        try:
            y5, y4, err = _dp_step(xi, state, h, n)
            num_rhs_evals += 7  # 7 stages per step
        except (TypeError, ValueError):
            # Step failed (e.g., negative theta for non-integer n)
            h *= 0.5
            rejected += 1
            step_count += 1
            if h < h_min:
                break
            continue

        if not np.all(np.isfinite(y5)):
            h *= 0.5
            rejected += 1
            step_count += 1
            if h < h_min:
                break
            continue

        # Check for zero crossing
        if stop_at_zero and state[0] > 0.0 and y5[0] <= 0.0:
            first_zero = _linear_first_zero(xi, state[0], xi + h, y5[0])
            xi_list.append(first_zero)
            theta_list.append(0.0)
            theta_prime_list.append(float(y5[1]))
            accepted += 1
            break

        # Step size control
        if err <= tolerance:
            # Accept step
            xi += h
            state = y5
            xi_list.append(xi)
            theta_list.append(float(state[0]))
            theta_prime_list.append(float(state[1]))
            accepted += 1

            min_h = min(min_h, h)
            max_h = max(max_h, h)

            # Increase step size for next step
            if err > 0:
                h_new = safety_factor * h * (tolerance / err) ** 0.2
                # Limit growth to factor of 5
                h = min(h_new, 5.0 * h, h_max)
            else:
                h = min(5.0 * h, h_max)
        else:
            # Reject step
            rejected += 1
            if err > 0:
                h_new = safety_factor * h * (tolerance / err) ** 0.2
                h = max(h_new, 0.1 * h, h_min)
            else:
                h *= 0.5

        step_count += 1

    return AdaptiveRKSolution(
        n=n,
        xi=np.array(xi_list, dtype=float),
        theta=np.array(theta_list, dtype=float),
        theta_prime=np.array(theta_prime_list, dtype=float),
        first_zero=first_zero,
        num_steps=step_count,
        num_rhs_evals=num_rhs_evals,
        accepted_steps=accepted,
        rejected_steps=rejected,
        min_h_used=min_h,
        max_h_used=max_h,
    )


if __name__ == "__main__":
    import math
    import time
    from pathlib import Path

    import importlib.util
    import sys
    rk_path = Path(__file__).resolve().parent / "Ronge-Kutta.py"
    spec = importlib.util.spec_from_file_location("rk4_solver", rk_path)
    rk4_mod = importlib.util.module_from_spec(spec)
    sys.modules["rk4_solver"] = rk4_mod
    spec.loader.exec_module(rk4_mod)
    solve_lane_emden_rk4 = rk4_mod.solve_lane_emden_rk4

    print("=" * 70)
    print("Adaptive RK5(4) vs Fixed-Step RK4 Comparison")
    print("=" * 70)

    for n_val in (0.0, 1.0, 3.0):
        print(f"\n--- n = {n_val:g} ---")

        # Determine xi_max for fixed-step methods
        if abs(n_val) < 1e-14:
            xi_max_val = math.sqrt(6.0)
            exact = lambda xi: 1.0 - xi**2 / 6.0
            stop_zero = True
        elif abs(n_val - 1.0) < 1e-14:
            xi_max_val = math.pi
            exact = lambda xi: np.ones_like(xi) if np.isscalar(xi) else np.where(xi != 0, np.sin(xi) / xi, 1.0)
            stop_zero = True
        else:
            ref = load_reference_data()
            xi_max_val = ref.get_first_zero(n_val) or 10.0
            exact = None
            stop_zero = (n_val < 5.0)

        # Fixed-step RK4 at various h
        print("  Fixed-step RK4:")
        for h_rk4 in (1e-2, 5e-3, 1e-3, 5e-4):
            try:
                t0 = time.perf_counter()
                sol = solve_lane_emden_rk4(
                    n=n_val, epsilon=1e-4, h=h_rk4,
                    xi_max=xi_max_val, stop_at_zero=stop_zero,
                )
                elapsed = time.perf_counter() - t0
                steps = len(sol.xi) - 1
                if exact is not None:
                    xi_eval = np.linspace(1e-4, min(xi_max_val, sol.xi[-1]), 2000)
                    err = np.max(np.abs(
                        np.interp(xi_eval, sol.xi, sol.theta) - exact(xi_eval)
                    ))
                    print(f"    h={h_rk4:.0e}: {steps} steps, {elapsed:.4f}s, error={err:.2e}")
                else:
                    print(f"    h={h_rk4:.0e}: {steps} steps, {elapsed:.4f}s, xi1={sol.first_zero:.8f}")
            except Exception as e:
                print(f"    h={h_rk4:.0e}: FAILED - {e}")

        # Adaptive RK5(4) at various tolerances
        print("  Adaptive RK5(4):")
        for tol in (1e-6, 1e-8, 1e-10):
            try:
                t0 = time.perf_counter()
                sol = solve_lane_emden_adaptive(
                    n=n_val, epsilon=1e-4, tolerance=tol,
                    xi_max=xi_max_val, stop_at_zero=stop_zero,
                )
                elapsed = time.perf_counter() - t0
                if exact is not None:
                    xi_eval = np.linspace(1e-4, min(xi_max_val, sol.xi[-1]), 2000)
                    err = np.max(np.abs(
                        np.interp(xi_eval, sol.xi, sol.theta) - exact(xi_eval)
                    ))
                    print(
                        f"    tol={tol:.0e}: {sol.accepted_steps} accepted + {sol.rejected_steps} rejected, "
                        f"{elapsed:.4f}s, error={err:.2e}, "
                        f"h∈[{sol.min_h_used:.1e}, {sol.max_h_used:.1e}]"
                    )
                else:
                    print(
                        f"    tol={tol:.0e}: {sol.accepted_steps} accepted + {sol.rejected_steps} rejected, "
                        f"{elapsed:.4f}s, xi1={sol.first_zero:.8f}, "
                        f"h∈[{sol.min_h_used:.1e}, {sol.max_h_used:.1e}]"
                    )
            except Exception as e:
                print(f"    tol={tol:.0e}: FAILED - {e}")

    # Efficiency comparison plot (error vs function evaluations)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from pathlib import Path

    OUTPUT_DIR = Path(__file__).resolve().parent / "output_initial"
    OUTPUT_DIR.mkdir(exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for col, (n_val, xi_max_val) in enumerate([
        (0.0, math.sqrt(6.0)), (1.0, math.pi),
    ]):
        ax = axes[col]
        if abs(n_val) < 1e-14:
            exact_func = lambda xi: 1.0 - xi**2 / 6.0
            stop_zero = True
        else:
            def exact_func(xi):
                r = np.ones_like(xi, dtype=float)
                m = xi != 0
                r[m] = np.sin(xi[m]) / xi[m]
                return r
            stop_zero = True

        # Fixed-step RK4 data
        h_rk4_list = [1e-2, 5e-3, 2.5e-3, 1.25e-3, 6.25e-4]
        rk4_evals = []
        rk4_errs = []
        for h_rk4 in h_rk4_list:
            try:
                sol = solve_lane_emden_rk4(n=n_val, epsilon=1e-4, h=h_rk4,
                                           xi_max=xi_max_val, stop_at_zero=stop_zero)
                xi_eval = np.linspace(1e-4, min(xi_max_val, sol.xi[-1]), 2000)
                err = np.max(np.abs(
                    np.interp(xi_eval, sol.xi, sol.theta) - exact_func(xi_eval)
                ))
                rk4_evals.append(4 * (len(sol.xi) - 1))
                rk4_errs.append(err)
            except Exception:
                continue

        ax.loglog(rk4_evals, rk4_errs, "ko-", linewidth=1.2, label="RK4 (fixed)")

        # Adaptive RK5(4) data
        tol_list = [1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9]
        ad_evals = []
        ad_errs = []
        for tol in tol_list:
            try:
                sol = solve_lane_emden_adaptive(
                    n=n_val, epsilon=1e-4, tolerance=tol,
                    xi_max=xi_max_val, stop_at_zero=stop_zero,
                )
                xi_eval = np.linspace(1e-4, min(xi_max_val, sol.xi[-1]), 2000)
                err = np.max(np.abs(
                    np.interp(xi_eval, sol.xi, sol.theta) - exact_func(xi_eval)
                ))
                ad_evals.append(sol.num_rhs_evals)
                ad_errs.append(err)
            except Exception:
                continue

        ax.loglog(ad_evals, ad_errs, "rs-", linewidth=1.2, label="RK5(4) adaptive")
        ax.set_xlabel("RHS evaluations")
        ax.set_ylabel(r"$\|\theta_{num} - \theta_{exact}\|_\infty$")
        ax.set_title(f"Efficiency: n={n_val:g}")
        ax.legend()
        ax.grid(True, which="both", alpha=0.3)

    fig.suptitle("Fixed-RK4 vs Adaptive-RK5(4): Accuracy vs Cost", fontsize=13)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "adaptive_rk_efficiency.png", dpi=200)
    plt.close()
    print(f"\nPlot saved to {OUTPUT_DIR / 'adaptive_rk_efficiency.png'}")
