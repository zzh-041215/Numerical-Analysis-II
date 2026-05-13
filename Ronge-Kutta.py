from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from data_input import load_reference_data


@dataclass(frozen=True)
class LaneEmdenSolution:
    n: float
    xi: np.ndarray
    theta: np.ndarray
    theta_prime: np.ndarray
    first_zero: Optional[float]


def taylor_initial_conditions(n: float, epsilon: float) -> tuple[float, float]:
    """Approximate theta(epsilon) and theta'(epsilon) near xi=0."""
    if epsilon <= 0:
        raise ValueError("epsilon must be positive.")

    theta = 1.0 - epsilon**2 / 6.0 + n * epsilon**4 / 120.0
    theta_prime = -epsilon / 3.0 + n * epsilon**3 / 30.0
    return theta, theta_prime


def lane_emden_rhs(xi: float, state: np.ndarray, n: float) -> np.ndarray:
    """Right-hand side of the first-order Lane-Emden system."""
    if xi == 0:
        raise ValueError("xi=0 is singular; start from a positive epsilon.")

    theta, theta_prime = state
    if theta < 0.0 and not float(n).is_integer():
        raise TypeError("theta is negative and n is non-integer, resulting in complex value.")
    else:
        theta_power = theta**n

    return np.array(
        [
            theta_prime,
            -2.0 * theta_prime / xi - theta_power,
        ],
        dtype=float,
    )


def rk4_step(xi: float, state: np.ndarray, h: float, n: float) -> np.ndarray:
    """One classical fourth-order Runge-Kutta step."""
    k1 = lane_emden_rhs(xi, state, n)
    k2 = lane_emden_rhs(xi + h / 2.0, state + h * k1 / 2.0, n)
    k3 = lane_emden_rhs(xi + h / 2.0, state + h * k2 / 2.0, n)
    k4 = lane_emden_rhs(xi + h, state + h * k3, n)
    return state + h * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0


def _linear_first_zero(
    xi_left: float, theta_left: float, xi_right: float, theta_right: float,
) -> float:
    """Linear interpolation of theta's first zero between two adjacent points."""
    weight = abs(theta_left) / (abs(theta_left) + abs(theta_right))
    return xi_left + weight * (xi_right - xi_left)


def _linear_first_zero_from_slope(xi: float, theta: float, theta_prime: float) -> float:
    if theta_prime >= 0:
        return xi
    return xi - theta / theta_prime


def solve_lane_emden_rk4(
    n: float,
    epsilon: float = 1e-6,
    h: float = 1e-3,
    xi_max: Optional[float] = None,
    stop_at_zero: bool = True,
    zero_tolerance: float = 1e-10,
    max_steps: int = 1_000_000,
) -> LaneEmdenSolution:
    """Solve the Lane-Emden equation by RK4.

    Parameters
    ----------
    n:
        Polytropic index. This implementation is intended for 0 <= n < 5.
    epsilon:
        Positive starting point. The initial value at xi=epsilon is produced
        by Taylor expansion around xi=0.
    h:
        Fixed RK4 step size.
    xi_max:
        End point of integration. If omitted, the reference first zero is used
        when available; otherwise a conservative default is used.
    stop_at_zero:
        Stop once theta first crosses zero.
    zero_tolerance:
        Stop by linear extrapolation when theta is already very close to zero.
        This avoids evaluating theta**n for negative theta when n is non-integer.
    max_steps:
        Safety cap for long integrations, especially when n is close to 5.
    """
    if not (0.0 <= n < 5.0):
        raise ValueError("n must satisfy 0 <= n < 5.")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive.")
    if h <= 0:
        raise ValueError("h must be positive.")
    if zero_tolerance <= 0:
        raise ValueError("zero_tolerance must be positive.")
    if max_steps <= 0:
        raise ValueError("max_steps must be positive.")

    reference_data = load_reference_data()
    reference_zero = reference_data.get_first_zero(n)

    if xi_max is None:
        if reference_zero is not None:
            xi_max = reference_zero + h
        else:
            xi_max = 10.0
    if xi_max <= epsilon:
        raise ValueError("xi_max must be greater than epsilon.")

    theta0, theta_prime0 = taylor_initial_conditions(n, epsilon)
    xi_values = [epsilon]
    theta_values = [theta0]
    theta_prime_values = [theta_prime0]

    xi = epsilon
    state = np.array([theta0, theta_prime0], dtype=float)
    first_zero: Optional[float] = None

    step_count = 0
    while xi < xi_max and step_count < max_steps:
        if stop_at_zero and 0.0 < state[0] <= zero_tolerance:
            first_zero = _linear_first_zero_from_slope(xi, state[0], state[1])
            xi_values.append(first_zero)
            theta_values.append(0.0)
            theta_prime_values.append(float(state[1]))
            break

        step = min(h, xi_max - xi)
        new_state = rk4_step(xi, state, step, n)
        new_xi = xi + step
        step_count += 1

        if stop_at_zero and not np.all(np.isfinite(new_state)):
            first_zero = _linear_first_zero_from_slope(xi, state[0], state[1])
            xi_values.append(first_zero)
            theta_values.append(0.0)
            theta_prime_values.append(float(state[1]))
            break

        if stop_at_zero and state[0] > 0.0 and new_state[0] <= 0.0:
            first_zero = _linear_first_zero(xi, state[0], new_xi, new_state[0])
            xi_values.append(first_zero)
            theta_values.append(0.0)
            theta_prime_values.append(float(new_state[1]))
            break

        xi_values.append(new_xi)
        theta_values.append(float(new_state[0]))
        theta_prime_values.append(float(new_state[1]))

        xi = new_xi
        state = new_state

    return LaneEmdenSolution(
        n=n,
        xi=np.array(xi_values, dtype=float),
        theta=np.array(theta_values, dtype=float),
        theta_prime=np.array(theta_prime_values, dtype=float),
        first_zero=first_zero,
    )
