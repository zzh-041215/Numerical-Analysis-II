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
class FiniteDifferenceSolution:
    n: float
    xi: np.ndarray
    theta: np.ndarray
    theta_prime: np.ndarray
    first_zero: Optional[float]
    converged: bool
    iterations: int
    residual_norm: float
    update_norm: float


def taylor_initial_conditions(n: float, epsilon: float) -> tuple[float, float]:
    """Approximate theta(epsilon) and theta'(epsilon) near the singular center."""
    if epsilon <= 0:
        raise ValueError("epsilon must be positive.")

    theta = 1.0 - epsilon**2 / 6.0 + n * epsilon**4 / 120.0
    theta_prime = -epsilon / 3.0 + n * epsilon**3 / 30.0
    return theta, theta_prime


def initial_guess(
    xi: np.ndarray,
    theta_left: float,
    theta_right: float,
) -> np.ndarray:
    """Build a positive initial profile for Newton iteration."""
    t = (xi - xi[0]) / (xi[-1] - xi[0])
    lane_emden_n5_shape = 1.0 / np.sqrt(1.0 + xi * xi / 3.0)
    lane_emden_n5_shape *= theta_left / lane_emden_n5_shape[0]
    guess = theta_right + (lane_emden_n5_shape - theta_right) * (1.0 - t)
    guess[0] = theta_left
    guess[-1] = theta_right
    return guess


def build_residual(theta: np.ndarray, xi: np.ndarray, n: float) -> np.ndarray:
    """Residual of the centered finite-difference Lane-Emden equations."""
    h = xi[1] - xi[0]
    h2 = h * h
    xi_mid = xi[1:-1]
    theta_left = theta[:-2]
    theta_mid = theta[1:-1]
    theta_right = theta[2:]

    return (
        (theta_right - 2.0 * theta_mid + theta_left) / h2
        + (theta_right - theta_left) / (xi_mid * h)
        + theta_mid**n
    )


#Build the Jacobian of the finite-difference equations in sparse tridiagonal form.
def build_jacobian_tridiagonal(
    theta: np.ndarray,
    xi: np.ndarray,
    n: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the tridiagonal Jacobian in sparse banded storage.

    The returned arrays are lower, diagonal and upper. This keeps the Newton
    linear solve in O(N) memory and O(N) operations with the Thomas algorithm.
    """
    h = xi[1] - xi[0]
    h2 = h * h
    xi_mid = xi[1:-1]
    theta_mid = theta[1:-1]

    lower_full = 1.0 / h2 - 1.0 / (xi_mid * h)
    diagonal = -2.0 / h2 + n * theta_mid ** (n - 1.0)
    upper_full = 1.0 / h2 + 1.0 / (xi_mid * h)

    lower = lower_full[1:].copy()
    upper = upper_full[:-1].copy()
    return lower, diagonal.copy(), upper

# Using the Thomas algorithm to solve the tridiagonal linear system
def solve_tridiagonal(
    lower: np.ndarray,
    diagonal: np.ndarray,
    upper: np.ndarray,
    rhs: np.ndarray,
) -> np.ndarray:
    n_unknowns = diagonal.size
    # marginal test
    if rhs.size != n_unknowns:
        raise ValueError("rhs and diagonal must have the same length.")
    if lower.size != max(0, n_unknowns - 1) or upper.size != max(0, n_unknowns - 1):
        raise ValueError("lower and upper must have length len(diagonal)-1.")

    a = lower.astype(float, copy=True)
    b = diagonal.astype(float, copy=True)
    c = upper.astype(float, copy=True)
    d = rhs.astype(float, copy=True)

    for i in range(1, n_unknowns):
        if abs(b[i - 1]) < 1e-14:
            raise ZeroDivisionError("zero pivot encountered in tridiagonal solve.")
        factor = a[i - 1] / b[i - 1]
        b[i] -= factor * c[i - 1]
        d[i] -= factor * d[i - 1]

    if abs(b[-1]) < 1e-14:
        raise ZeroDivisionError("zero pivot encountered in tridiagonal solve.")

    solution = np.empty_like(d)
    solution[-1] = d[-1] / b[-1]
    for i in range(n_unknowns - 2, -1, -1):
        solution[i] = (d[i] - c[i] * solution[i + 1]) / b[i]

    return solution


def newton_step(theta: np.ndarray, xi: np.ndarray, n: float) -> tuple[np.ndarray, float]:
    """Construct one Newton correction using the tridiagonal sparse Jacobian."""
    residual = build_residual(theta, xi, n)
    lower, diagonal, upper = build_jacobian_tridiagonal(theta, xi, n)
    delta_inner = solve_tridiagonal(lower, diagonal, upper, -residual)
    return delta_inner, float(np.linalg.norm(residual, ord=np.inf))

# Use a damped Newton update to ensure the positivity of the interior profile.
def damped_update(
    theta: np.ndarray,
    xi: np.ndarray,
    n: float,
    delta_inner: np.ndarray,
    residual_norm: float,
    *,
    min_damping: float = 1e-4,
) -> tuple[np.ndarray, float, float]:
    damping = 1.0
    best_theta = theta.copy()
    best_residual = residual_norm

    while damping >= min_damping:
        candidate = theta.copy()
        candidate[1:-1] += damping * delta_inner

        if np.any(candidate[1:-1] <= 0.0):
            damping *= 0.5
            continue

        candidate_residual = float(np.linalg.norm(build_residual(candidate, xi, n), ord=np.inf))
        if candidate_residual < residual_norm:
            return candidate, candidate_residual, damping

        if candidate_residual < best_residual:
            best_theta = candidate
            best_residual = candidate_residual

        damping *= 0.5

    return best_theta, best_residual, 0.0


def compute_theta_prime(theta: np.ndarray, xi: np.ndarray) -> np.ndarray:
    """Compute theta' on the grid with second-order finite differences."""
    if theta.size < 3:
        return np.gradient(theta, xi)
    return np.gradient(theta, xi, edge_order=2)


def find_first_zero(xi: np.ndarray, theta: np.ndarray) -> Optional[float]:
    """Locate the first zero by linear interpolation."""
    for i in range(1, theta.size):
        if theta[i] == 0.0:
            return float(xi[i])
        if theta[i - 1] * theta[i] < 0.0:
            weight = abs(theta[i - 1]) / (abs(theta[i - 1]) + abs(theta[i]))
            return float(xi[i - 1] + weight * (xi[i] - xi[i - 1]))
    return None


def solve_lane_emden_finite_difference(
    n: float,
    epsilon: float = 1e-6,
    num_intervals: int = 1000,
    xi_max: Optional[float] = None,
    theta_right: float = 0.0,
    max_iterations: int = 50,
    residual_tolerance: float = 1e-10,
    update_tolerance: float = 1e-10,
) -> FiniteDifferenceSolution:
    """Solve the Lane-Emden equation with finite differences and Newton iteration.

    The boundary conditions are theta(epsilon) from Taylor expansion and
    theta(xi_max)=theta_right. If xi_max is omitted, the reference first zero
    from polytrope_global_properties.csv is used.

    The Newton stopping rule uses both:
    1. infinity norm of the nonlinear residual;
    2. relative infinity norm of the Newton update.

    The first quantity checks whether the discrete equation is satisfied, while
    the second checks whether the iterate itself has stabilized.
    """
    if not (0.0 <= n <= 5.0):
        raise ValueError("n must satisfy 0 <= n <= 5.")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive.")
    if num_intervals < 2:
        raise ValueError("num_intervals must be at least 2.")
    if max_iterations <= 0:
        raise ValueError("max_iterations must be positive.")
    if residual_tolerance <= 0 or update_tolerance <= 0:
        raise ValueError("tolerances must be positive.")

    if xi_max is None:
        reference_zero = load_reference_data().get_first_zero(n)
        if reference_zero is None:
            raise ValueError("xi_max is required when no reference first zero is available.")
        xi_max = reference_zero
    if xi_max <= epsilon:
        raise ValueError("xi_max must be greater than epsilon.")

    theta_left, _ = taylor_initial_conditions(n, epsilon)
    xi = np.linspace(epsilon, xi_max, num_intervals + 1)
    theta = initial_guess(xi, theta_left, theta_right)

    converged = False
    residual_norm = float(np.linalg.norm(build_residual(theta, xi, n), ord=np.inf))
    update_norm = float("inf")
    iterations = 0

    for iterations in range(1, max_iterations + 1):
        delta_inner, residual_norm = newton_step(theta, xi, n)
        theta_candidate, candidate_residual, damping = damped_update(
            theta,
            xi,
            n,
            delta_inner,
            residual_norm,
        )

        actual_delta = theta_candidate[1:-1] - theta[1:-1]
        update_norm = float(
            np.linalg.norm(actual_delta, ord=np.inf)
            / (1.0 + np.linalg.norm(theta_candidate[1:-1], ord=np.inf))
        )

        theta = theta_candidate
        residual_norm = candidate_residual

        if residual_norm <= residual_tolerance and update_norm <= update_tolerance:
            converged = True
            break
        if damping == 0.0 and update_norm <= update_tolerance:
            converged = residual_norm <= residual_tolerance
            break

    theta_prime = compute_theta_prime(theta, xi)
    first_zero = find_first_zero(xi, theta)
    if first_zero is None and abs(theta[-1]) <= residual_tolerance:
        first_zero = float(xi[-1])

    return FiniteDifferenceSolution(
        n=n,
        xi=xi,
        theta=theta,
        theta_prime=theta_prime,
        first_zero=first_zero,
        converged=converged,
        iterations=iterations,
        residual_norm=residual_norm,
        update_norm=update_norm,
    )
