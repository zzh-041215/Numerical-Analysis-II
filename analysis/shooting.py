from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import sys
from pathlib import Path

import numpy as np

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from data_input import load_reference_data


@dataclass(frozen=True)
class ShootingResult:
    """Result of solving the Lane-Emden equation via shooting method.

    The shooting method treats the Lane-Emden equation as a BVP:
    - theta(0) = 1, theta'(0) = 0 (inner BC, handled by Taylor expansion)
    - theta(xi_1) = 0 (outer BC, target condition)

    The unknown initial condition (theta'(epsilon) near the center) is adjusted
    until the outer BC is satisfied. Since theta'(epsilon) is fixed by Taylor
    expansion, we instead vary the parameter n slightly, or more practically,
    we shoot for the correct xi_1 given a fixed n.
    """
    n: float
    xi: np.ndarray
    theta: np.ndarray
    theta_prime: np.ndarray
    xi_1: float
    shooting_iterations: int
    converged: bool
    residual: float


def taylor_initial_conditions(n: float, epsilon: float) -> tuple[float, float]:
    """Taylor expansion near xi=0."""
    theta = 1.0 - epsilon**2 / 6.0 + n * epsilon**4 / 120.0
    theta_prime = -epsilon / 3.0 + n * epsilon**3 / 30.0
    return theta, theta_prime


def _rk4_integrate(
    n: float,
    epsilon: float,
    h: float,
    xi_target: float,
    theta_prime_epsilon: Optional[float] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    """Integrate from epsilon to xi_target using RK4.

    If theta_prime_epsilon is None, uses Taylor expansion value.

    Returns (xi, theta, theta_prime, theta_at_target, theta_prime_at_target).
    """
    if theta_prime_epsilon is None:
        theta0, tp0 = taylor_initial_conditions(n, epsilon)
    else:
        theta0 = 1.0 - epsilon**2 / 6.0 + n * epsilon**4 / 120.0
        tp0 = theta_prime_epsilon

    xi_arr = [epsilon]
    theta_arr = [theta0]
    tp_arr = [tp0]

    xi = epsilon
    state = np.array([theta0, tp0], dtype=float)

    while xi < xi_target:
        step = min(h, xi_target - xi)
        if xi <= 0:
            break

        # RK4 step
        def rhs(xi_s, s):
            t, tp = s
            if t < 0 and not float(n).is_integer():
                t_pow = 0.0
            else:
                t_pow = float(t) ** n
            return np.array([tp, -2.0 * tp / xi_s - t_pow])

        k1 = rhs(xi, state)
        k2 = rhs(xi + step / 2, state + step * k1 / 2)
        k3 = rhs(xi + step / 2, state + step * k2 / 2)
        k4 = rhs(xi + step, state + step * k3)

        state = state + step * (k1 + 2 * k2 + 2 * k3 + k4) / 6
        xi += step

        xi_arr.append(xi)
        theta_arr.append(float(state[0]))
        tp_arr.append(float(state[1]))

    return (
        np.array(xi_arr), np.array(theta_arr), np.array(tp_arr),
        float(state[0]), float(state[1]),
    )


def shoot_to_surface(
    n: float,
    epsilon: float = 1e-4,
    h: float = 1e-3,
    xi_guess: Optional[float] = None,
    max_iterations: int = 20,
    tolerance: float = 1e-10,
) -> ShootingResult:
    """Shooting method: find xi_1 such that theta(xi_1) = 0.

    This is the standard shooting approach for the Lane-Emden problem.
    Given n, we integrate outward and find where theta crosses zero.
    This is essentially what the RK4 solver already does with stop_at_zero=True,
    but here we frame it as a proper shooting/BVP method.

    The "shooting parameter" is xi_1 (the outer boundary location).
    We use secant method to solve theta(xi_1) = 0.

    Parameters
    ----------
    n:
        Polytropic index.
    epsilon:
        Starting point.
    h:
        Integration step size.
    xi_guess:
        Initial guess for xi_1. If None, uses reference data.
    max_iterations:
        Maximum shooting iterations.
    tolerance:
        Convergence tolerance on |theta(xi_1)|.

    Returns
    -------
    ShootingResult.
    """
    # Initial guess
    if xi_guess is None:
        ref = load_reference_data()
        prop = ref.get_global_property(n)
        if prop is not None and prop.xi_1 is not None:
            xi_guess = prop.xi_1
        else:
            xi_guess = 5.0  # fallback

    # Secant method setup
    xi_prev = xi_guess * 0.9
    xi_curr = xi_guess

    _, _, _, theta_prev, _ = _rk4_integrate(n, epsilon, h, xi_prev)
    converged = False
    iterations = 0
    final_xi = None
    final_theta = None
    final_theta_prime = None
    final_xi_arr = None

    for iterations in range(1, max_iterations + 1):
        xi_arr, theta_arr, tp_arr, theta_curr, tp_curr = _rk4_integrate(
            n, epsilon, h, xi_curr,
        )

        if abs(theta_curr) < tolerance:
            converged = True
            final_xi = xi_curr
            final_theta = theta_arr
            final_theta_prime = tp_arr
            final_xi_arr = xi_arr
            break

        # Secant update
        if abs(theta_curr - theta_prev) < 1e-15:
            # Secant would divide by zero; take a Newton-like step
            xi_next = xi_curr - theta_curr / tp_curr if tp_curr != 0 else xi_curr * 0.99
        else:
            xi_next = xi_curr - theta_curr * (xi_curr - xi_prev) / (theta_curr - theta_prev)

        # Keep xi_next in reasonable bounds
        xi_next = max(xi_curr * 0.5, min(xi_curr * 1.5, xi_next))

        xi_prev = xi_curr
        theta_prev = theta_curr
        xi_curr = xi_next

    if not converged:
        # Return best available
        xi_arr, theta_arr, tp_arr, theta_curr, tp_curr = _rk4_integrate(
            n, epsilon, h, xi_curr,
        )
        final_xi = xi_curr
        final_theta = theta_arr
        final_theta_prime = tp_arr
        final_xi_arr = xi_arr

    # Find exact zero by interpolation
    for i in range(1, len(final_theta)):
        if final_theta[i] <= 0.0:
            w = abs(final_theta[i - 1]) / (abs(final_theta[i - 1]) + abs(final_theta[i]))
            xi_1_exact = final_xi_arr[i - 1] + w * (final_xi_arr[i] - final_xi_arr[i - 1])
            final_xi = xi_1_exact
            break

    return ShootingResult(
        n=n,
        xi=final_xi_arr,
        theta=final_theta,
        theta_prime=final_theta_prime,
        xi_1=float(final_xi) if final_xi is not None else xi_curr,
        shooting_iterations=iterations,
        converged=converged,
        residual=float(abs(theta_curr)),
    )


def shoot_to_surface_parameter(
    n: float,
    epsilon: float = 1e-4,
    h: float = 1e-3,
    xi_max: Optional[float] = None,
    theta_prime_adjustment: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Variant: shoot by adjusting theta'(epsilon) slightly for a given xi_max.

    This demonstrates the classical shooting approach where the initial slope
    is the shooting parameter and xi_max is fixed.

    Parameters
    ----------
    n, epsilon, h:
        Standard parameters.
    xi_max:
        Fixed outer boundary. If None, uses reference xi_1.
    theta_prime_adjustment:
        Adjustment to the Taylor theta'(epsilon). Used for secant iteration.

    Returns
    -------
    (xi, theta, theta_prime) for the integrated solution.
    """
    if xi_max is None:
        ref = load_reference_data()
        prop = ref.get_global_property(n)
        xi_max = prop.xi_1 if (prop and prop.xi_1) else 5.0

    tp_epsilon = taylor_initial_conditions(n, epsilon)[1] + theta_prime_adjustment
    xi_arr, theta_arr, tp_arr, _, _ = _rk4_integrate(
        n, epsilon, h, xi_max, theta_prime_epsilon=tp_epsilon,
    )
    return xi_arr, theta_arr, tp_arr


if __name__ == "__main__":
    import math

    print("=" * 70)
    print("Shooting Method for Lane-Emden Equation")
    print("=" * 70)

    for n_test in (0.0, 1.0, 2.0, 3.0):
        print(f"\n--- n = {n_test:g} ---")

        # Get reference value
        ref = load_reference_data()
        prop = ref.get_global_property(n_test)
        xi_ref = prop.xi_1 if prop else None

        # Shooting method
        result = shoot_to_surface(n_test, epsilon=1e-4, h=1e-3)

        print(f"  Shooting result:  xi_1 = {result.xi_1:.8f}")
        if xi_ref is not None:
            print(f"  Reference value:  xi_1 = {xi_ref:.8f}")
            print(f"  Absolute error:   {abs(result.xi_1 - xi_ref):.2e}")
        print(f"  Iterations:       {result.shooting_iterations}")
        print(f"  Converged:        {result.converged}")
        print(f"  Residual:         {result.residual:.2e}")

    # Demonstrate classical shooting: vary theta'(epsilon) to hit theta(xi_max)=0
    print(f"\n--- Classical shooting demonstration (n=3) ---")
    n_demo = 3.0
    prop = load_reference_data().get_global_property(n_demo)
    xi_target = prop.xi_1 if prop else 6.897

    # Secant on theta'(epsilon)
    tp0, tp1 = -0.1, -0.05  # two initial guesses for theta'(epsilon)
    eps = 1e-4

    xi1, t1, _, theta_end0, _ = _rk4_integrate(n_demo, eps, 1e-3, xi_target,
                                                theta_prime_epsilon=tp0)
    _, _, _, theta_end1, _ = _rk4_integrate(n_demo, eps, 1e-3, xi_target,
                                            theta_prime_epsilon=tp1)

    for it in range(10):
        if abs(theta_end1) < 1e-10:
            break
        tp_new = tp1 - theta_end1 * (tp1 - tp0) / (theta_end1 - theta_end0 + 1e-15)
        tp0, tp1 = tp1, tp_new
        theta_end0 = theta_end1
        _, _, _, theta_end1, _ = _rk4_integrate(n_demo, eps, 1e-3, xi_target,
                                                theta_prime_epsilon=tp1)

    print(f"  Taylor theta'(epsilon) = {taylor_initial_conditions(n_demo, eps)[1]:.6f}")
    print(f"  Adjusted theta'(epsilon) = {tp1:.6f}")
    print(f"  theta(xi_max) residual = {theta_end1:.2e}")
    print(f"  (Adjustment needed because Taylor expansion order is limited)")
