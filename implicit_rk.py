from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from data_input import load_reference_data


@dataclass(frozen=True)
class ImplicitRKSolution:
    """Solution of Lane-Emden using implicit Runge-Kutta (Radau IIA)."""
    n: float
    xi: np.ndarray
    theta: np.ndarray
    theta_prime: np.ndarray
    first_zero: Optional[float]
    num_steps: int
    num_newton_iters: int


# 3-stage Radau IIA Butcher tableau (order 5, L-stable)
# This method is particularly well-suited for stiff ODEs.
RADAU_C = np.array([(4.0 - np.sqrt(6.0)) / 10.0,
                     (4.0 + np.sqrt(6.0)) / 10.0,
                     1.0])
RADAU_A = np.array([
    [(88.0 - 7.0 * np.sqrt(6.0)) / 360.0,
     (296.0 - 169.0 * np.sqrt(6.0)) / 1800.0,
     (-2.0 + 3.0 * np.sqrt(6.0)) / 225.0],
    [(296.0 + 169.0 * np.sqrt(6.0)) / 1800.0,
     (88.0 + 7.0 * np.sqrt(6.0)) / 360.0,
     (-2.0 - 3.0 * np.sqrt(6.0)) / 225.0],
    [(16.0 - np.sqrt(6.0)) / 36.0,
     (16.0 + np.sqrt(6.0)) / 36.0,
     1.0 / 9.0],
])
RADAU_B = np.array([(16.0 - np.sqrt(6.0)) / 36.0,
                    (16.0 + np.sqrt(6.0)) / 36.0,
                    1.0 / 9.0])


def taylor_initial_conditions(n: float, epsilon: float) -> tuple[float, float]:
    """Taylor expansion near xi=0."""
    theta = 1.0 - epsilon**2 / 6.0 + n * epsilon**4 / 120.0
    theta_prime = -epsilon / 3.0 + n * epsilon**3 / 30.0
    return theta, theta_prime


def _radau_step(state: np.ndarray, xi: float, h: float, n: float,
                max_newton: int = 10, newton_tol: float = 1e-12) -> tuple[np.ndarray, int]:
    """One Radau IIA step: solve for stage values via simplified Newton.

    Returns (new_state, newton_iterations).
    """
    s = 3  # stages
    dim = 2  # (theta, theta_prime)

    # Initial guess for stages: use explicit Euler
    Z = np.zeros((s, dim))
    for i in range(s):
        xi_s = xi + RADAU_C[i] * h
        def rhs_simple(t, tp):
            if t < 0 and not float(n).is_integer():
                return np.array([tp, -2.0 * tp / xi_s])
            return np.array([tp, -2.0 * tp / xi_s - t ** n])

        Z[i] = state + RADAU_C[i] * h * rhs_simple(state[0], state[1])

    # Simplified Newton iteration
    total_iters = 0
    for _ in range(max_newton):
        max_correction = 0.0
        for i in range(s):
            xi_s = xi + RADAU_C[i] * h

            def rhs(t, tp):
                if t < 0 and not float(n).is_integer():
                    t_pow = 0.0
                else:
                    t_pow = t ** n
                return np.array([tp, -2.0 * tp / xi_s - t_pow])

            # Stage equation residual
            stage_res = Z[i] - state
            for j in range(s):
                stage_res -= h * RADAU_A[i, j] * rhs(Z[j][0], Z[j][1])

            # Jacobian approximation (use diagonal dominance)
            J_diag = np.eye(dim) - h * RADAU_A[i, i] * np.array([
                [0, 1],
                [-n * Z[i][0] ** (n - 1) if Z[i][0] > 0 else 0, -2.0 / xi_s],
            ])

            try:
                correction = np.linalg.solve(J_diag, -stage_res)
            except np.linalg.LinAlgError:
                correction = -stage_res / (1.0 + h)

            Z[i] += correction
            max_correction = max(max_correction, np.max(np.abs(correction)))

        total_iters += 1
        if max_correction < newton_tol:
            break

    # Combine stages
    new_state = state.copy()
    for i in range(s):
        xi_s = xi + RADAU_C[i] * h
        def rhs(t, tp):
            if t < 0 and not float(n).is_integer():
                t_pow = 0.0
            else:
                t_pow = t ** n
            return np.array([tp, -2.0 * tp / xi_s - t_pow])
        new_state += h * RADAU_B[i] * rhs(Z[i][0], Z[i][1])

    return new_state, total_iters


def solve_lane_emden_radau(
    n: float,
    epsilon: float = 1e-6,
    h: float = 1e-2,
    xi_max: Optional[float] = None,
    stop_at_zero: bool = True,
    max_steps: int = 100_000,
) -> ImplicitRKSolution:
    """Solve Lane-Emden using 3-stage Radau IIA (implicit RK, order 5, L-stable).

    Radau IIA is L-stable (perfect damping of stiff modes), making it
    suitable for n close to 5 where the problem becomes moderately stiff.
    """
    if xi_max is None:
        ref = load_reference_data()
        prop = ref.get_global_property(n)
        xi_max = (prop.xi_1 or 10.0) + h if prop else 10.0

    theta0, theta_prime0 = taylor_initial_conditions(n, epsilon)

    xi_list = [epsilon]
    theta_list = [theta0]
    tp_list = [theta_prime0]

    xi = epsilon
    state = np.array([theta0, theta_prime0], dtype=float)
    first_zero = None
    step_count = 0
    total_newton = 0

    while xi < xi_max and step_count < max_steps:
        step = min(h, xi_max - xi)

        # Check for near-zero
        if stop_at_zero and state[0] <= 1e-10 and state[1] < 0:
            first_zero = float(xi - state[0] / state[1])
            xi_list.append(first_zero)
            theta_list.append(0.0)
            tp_list.append(float(state[1]))
            break

        new_state, n_iters = _radau_step(state, xi, step, n)
        new_xi = xi + step
        step_count += 1
        total_newton += n_iters

        if stop_at_zero and state[0] > 0 and new_state[0] <= 0:
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

    return ImplicitRKSolution(
        n=n, xi=np.array(xi_list), theta=np.array(theta_list),
        theta_prime=np.array(tp_list), first_zero=first_zero,
        num_steps=step_count, num_newton_iters=total_newton,
    )


if __name__ == "__main__":
    import math
    import time

    print("=" * 70)
    print("Implicit RK (Radau IIA) vs Explicit RK4")
    print("=" * 70)

    # Focus on n close to 5 where stiffness matters
    for n_val in (0.0, 3.0, 4.5):
        print(f"\n--- n = {n_val:g} ---")

        if abs(n_val) < 1e-14:
            xi_max_val = math.sqrt(6.0)
        elif abs(n_val - 1.0) < 1e-14:
            xi_max_val = math.pi
        else:
            ref = load_reference_data()
            xi_max_val = ref.get_first_zero(n_val) or 10.0

        # Implicit Radau
        print("  Radau IIA (implicit, L-stable):")
        for h_test in (5e-2, 2e-2, 1e-2):
            try:
                t0 = time.perf_counter()
                sol = solve_lane_emden_radau(n=n_val, epsilon=1e-4, h=h_test,
                                             xi_max=xi_max_val)
                elapsed = time.perf_counter() - t0
                print(f"    h={h_test:.0e}: {sol.num_steps} steps, "
                      f"{sol.num_newton_iters} Newton iters, "
                      f"{elapsed:.4f}s, xi_1={sol.first_zero}")
            except Exception as e:
                print(f"    h={h_test:.0e}: FAILED - {e}")

        # Compare with explicit RK4
        print("  RK4 (explicit):")
        from pathlib import Path
        import importlib.util, sys
        rk_path = Path(__file__).resolve().parent / "Ronge-Kutta.py"
        spec = importlib.util.spec_from_file_location("rk_temp2", rk_path)
        rk_mod = importlib.util.module_from_spec(spec)
        sys.modules["rk_temp2"] = rk_mod
        spec.loader.exec_module(rk_mod)

        for h_test in (5e-2, 2e-2, 1e-2):
            try:
                t0 = time.perf_counter()
                sol = rk_mod.solve_lane_emden_rk4(n=n_val, epsilon=1e-4, h=h_test,
                                                   xi_max=xi_max_val)
                elapsed = time.perf_counter() - t0
                print(f"    h={h_test:.0e}: {len(sol.xi) - 1} steps, "
                      f"{elapsed:.4f}s, xi_1={sol.first_zero}")
            except Exception as e:
                print(f"    h={h_test:.0e}: FAILED - {e}")
