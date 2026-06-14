from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from data_input import load_reference_data


@dataclass(frozen=True)
class HighOrderFDSolution:
    """Solution of Lane-Emden equation using 4th-order compact FD."""
    n: float
    xi: np.ndarray
    theta: np.ndarray
    theta_prime: np.ndarray
    first_zero: Optional[float]
    converged: bool
    iterations: int
    residual_norm: float


def taylor_initial_conditions(n: float, epsilon: float) -> tuple[float, float]:
    """Taylor expansion near xi=0."""
    theta = 1.0 - epsilon**2 / 6.0 + n * epsilon**4 / 120.0
    theta_prime = -epsilon / 3.0 + n * epsilon**3 / 30.0
    return theta, theta_prime


def initial_guess(xi: np.ndarray, theta_left: float, theta_right: float) -> np.ndarray:
    """Build a positive initial profile for Newton iteration."""
    t = (xi - xi[0]) / (xi[-1] - xi[0])
    n5_shape = 1.0 / np.sqrt(1.0 + xi * xi / 3.0)
    n5_shape *= theta_left / n5_shape[0]
    guess = theta_right + (n5_shape - theta_right) * (1.0 - t)
    guess[0] = theta_left
    guess[-1] = theta_right
    return guess


def build_residual_4th(theta: np.ndarray, xi: np.ndarray, n: float) -> np.ndarray:
    """Fourth-order centered finite-difference residual.

    Uses 5-point stencils:
    theta''_j ≈ (-theta_{j+2} + 16*theta_{j+1} - 30*theta_j + 16*theta_{j-1} - theta_{j-2}) / (12*h^2)
    theta'_j  ≈ (-theta_{j+2} + 8*theta_{j+1} - 8*theta_{j-1} + theta_{j-2}) / (12*h)

    At interior points j=2,...,N-2.
    Near-boundary points (j=1, N-1) use second-order as fallback.
    """
    h = xi[1] - xi[0]
    N = len(xi)
    residual = np.zeros(N - 2)  # interior points only (indices 1..N-2)

    for idx in range(1, N - 1):
        j = idx
        if 2 <= j <= N - 3:
            # Fourth-order centered stencil
            d2 = (-theta[j + 2] + 16.0 * theta[j + 1] - 30.0 * theta[j]
                  + 16.0 * theta[j - 1] - theta[j - 2]) / (12.0 * h * h)
            d1 = (-theta[j + 2] + 8.0 * theta[j + 1] - 8.0 * theta[j - 1]
                  + theta[j - 2]) / (12.0 * h)
        else:
            # Second-order centered stencil for near-boundary points
            d2 = (theta[j + 1] - 2.0 * theta[j] + theta[j - 1]) / (h * h)
            d1 = (theta[j + 1] - theta[j - 1]) / (2.0 * h)

        residual[idx - 1] = d2 + (2.0 * d1) / xi[j] + theta[j] ** n

    return residual


def build_jacobian_banded(theta: np.ndarray, xi: np.ndarray, n: float):
    """Build pentadiagonal Jacobian for 4th-order FD scheme.

    Returns (banded_matrix, lower_bandwidth, upper_bandwidth) for use with
    scipy.linalg.solve_banded.
    """
    h = xi[1] - xi[0]
    N = len(xi)
    M = N - 2  # number of interior unknowns

    # Pentadiagonal: bandwidths = 2
    # Store in LAPACK banded format: row i of banded = diagonal offset by i from main
    banded = np.zeros((5, M))  # 2 lower + main + 2 upper = 5 rows

    for idx in range(1, N - 1):
        j = idx
        row = idx - 1

        if 2 <= j <= N - 3:
            # Fourth-order stencil Jacobian entries
            # d(residual_j)/d(theta_{j-2})
            if row - 2 >= 0:
                banded[0, row] = -1.0 / (12.0 * h * h) + 1.0 / (12.0 * h) * (2.0 / xi[j])
            # d(residual_j)/d(theta_{j-1})
            if row - 1 >= 0:
                banded[1, row] = 16.0 / (12.0 * h * h) - 8.0 / (12.0 * h) * (2.0 / xi[j])
            # d(residual_j)/d(theta_j)
            banded[2, row] = -30.0 / (12.0 * h * h) + n * theta[j] ** (n - 1.0)
            # d(residual_j)/d(theta_{j+1})
            if row + 1 < M:
                banded[3, row] = 16.0 / (12.0 * h * h) + 8.0 / (12.0 * h) * (2.0 / xi[j])
            # d(residual_j)/d(theta_{j+2})
            if row + 2 < M:
                banded[4, row] = -1.0 / (12.0 * h * h) - 1.0 / (12.0 * h) * (2.0 / xi[j])
        else:
            # Second-order tridiagonal Jacobian
            if row - 1 >= 0:
                banded[1, row] = 1.0 / (h * h) - 1.0 / (h) * (1.0 / xi[j])
            banded[2, row] = -2.0 / (h * h) + n * theta[j] ** (n - 1.0)
            if row + 1 < M:
                banded[3, row] = 1.0 / (h * h) + 1.0 / (h) * (1.0 / xi[j])

    return banded


def solve_lane_emden_fd_4th(
    n: float,
    epsilon: float = 1e-6,
    num_intervals: int = 500,
    xi_max: Optional[float] = None,
    theta_right: float = 0.0,
    max_iterations: int = 50,
    residual_tolerance: float = 1e-10,
    update_tolerance: float = 1e-10,
) -> HighOrderFDSolution:
    """Solve Lane-Emden with 4th-order finite differences and Newton iteration."""
    if xi_max is None:
        xi_max = load_reference_data().get_first_zero(n)
        if xi_max is None:
            raise ValueError("xi_max required.")

    theta_left, _ = taylor_initial_conditions(n, epsilon)
    xi = np.linspace(epsilon, xi_max, num_intervals + 1)
    theta = initial_guess(xi, theta_left, theta_right)

    converged = False
    residual_norm = float("inf")
    update_norm = float("inf")
    iterations = 0

    for iterations in range(1, max_iterations + 1):
        residual = build_residual_4th(theta, xi, n)
        residual_norm = float(np.linalg.norm(residual, ord=np.inf))

        if residual_norm < residual_tolerance and iterations > 1:
            converged = True
            break

        # Build banded Jacobian and solve with SciPy
        try:
            from scipy.linalg import solve_banded
            banded = build_jacobian_banded(theta, xi, n)
            delta = solve_banded((2, 2), banded, -residual)
        except ImportError:
            # Fallback: use simple Gauss-Seidel-like update
            h = xi[1] - xi[0]
            delta = -residual / (-2.0 / (h * h) + n * np.maximum(theta[1:-1], 1e-6) ** (n - 1.0))

        # Damped update
        damping = 1.0
        while damping > 1e-6:
            candidate = theta.copy()
            candidate[1:-1] += damping * delta
            if np.all(candidate[1:-1] >= 0.0):
                break
            damping *= 0.5

        actual_delta = candidate[1:-1] - theta[1:-1]
        update_norm = float(
            np.linalg.norm(actual_delta, ord=np.inf)
            / (1.0 + np.linalg.norm(candidate[1:-1], ord=np.inf))
        )

        theta = candidate

        if update_norm < update_tolerance and residual_norm < residual_tolerance:
            converged = True
            break

    theta_prime = np.gradient(theta, xi, edge_order=2)
    first_zero = None
    for i in range(1, len(theta)):
        if theta[i] <= 0:
            if theta[i - 1] > 0:
                w = abs(theta[i - 1]) / (abs(theta[i - 1]) + abs(theta[i]))
                first_zero = float(xi[i - 1] + w * (xi[i] - xi[i - 1]))
            break
    if first_zero is None and abs(theta[-1]) < 1e-10:
        first_zero = float(xi[-1])

    return HighOrderFDSolution(
        n=n, xi=xi, theta=theta, theta_prime=theta_prime,
        first_zero=first_zero, converged=converged,
        iterations=iterations, residual_norm=residual_norm,
    )


if __name__ == "__main__":
    import math

    print("=" * 70)
    print("Fourth-Order Finite Difference Solver")
    print("=" * 70)

    for n_val in (0.0, 1.0, 3.0):
        print(f"\n--- n = {n_val:g} ---")

        if abs(n_val) < 1e-14:
            xi_max_val = math.sqrt(6.0)
            exact_theta = lambda xi: 1.0 - xi**2 / 6.0
        elif abs(n_val - 1.0) < 1e-14:
            xi_max_val = math.pi
            def exact_theta(xi):
                r = np.ones_like(xi)
                m = xi != 0
                r[m] = np.sin(xi[m]) / xi[m]
                return r
        else:
            xi_max_val = load_reference_data().get_first_zero(n_val)
            exact_theta = None

        # Test 4th-order FD at different resolutions
        for N in (100, 200, 400, 800):
            try:
                sol = solve_lane_emden_fd_4th(n=n_val, num_intervals=N,
                                              xi_max=xi_max_val)
                h_actual = sol.xi[1] - sol.xi[0]
                if exact_theta is not None:
                    xi_eval = np.linspace(1e-6, xi_max_val, 2000)
                    err = np.max(np.abs(
                        np.interp(xi_eval, sol.xi, sol.theta) - exact_theta(xi_eval)
                    ))
                    print(f"  N={N:4d}, h={h_actual:.6f}, error={err:.2e}, "
                          f"iter={sol.iterations}, conv={sol.converged}")
                else:
                    print(f"  N={N:4d}, h={h_actual:.6f}, xi_1={sol.first_zero:.8f}, "
                          f"iter={sol.iterations}, conv={sol.converged}")
            except Exception as e:
                print(f"  N={N:4d}: FAILED - {e}")
