from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from data_input import load_reference_data


@dataclass(frozen=True)
class IsothermalSolution:
    """Solution of the isothermal Lane-Emden equation.

    The isothermal sphere equation is the n -> infinity limit:
        (1/xi^2) d/dxi (xi^2 dpsi/dxi) = exp(-psi)
    with BCs: psi(0) = 0, psi'(0) = 0.
    """
    xi: np.ndarray
    psi: np.ndarray          # theta analogue: density = rho_c * exp(-psi)
    psi_prime: np.ndarray    # derivative
    num_steps: int
    xi_max: float


def taylor_isothermal(epsilon: float) -> tuple[float, float]:
    """Taylor expansion of the isothermal solution near xi=0.

    psi(xi) = xi^2/6 - xi^4/120 + xi^6/1890 + O(xi^8)
    psi'(xi) = xi/3 - xi^3/30 + xi^5/315 + O(xi^7)
    """
    psi = epsilon**2 / 6.0 - epsilon**4 / 120.0 + epsilon**6 / 1890.0
    psi_prime = epsilon / 3.0 - epsilon**3 / 30.0 + epsilon**5 / 315.0
    return psi, psi_prime


def isothermal_rhs(xi: float, state: np.ndarray) -> np.ndarray:
    """RHS of the isothermal first-order system.

    y1' = y2
    y2' = -2*y2/xi + exp(-y1)
    """
    psi, psi_prime = state
    return np.array([psi_prime, -2.0 * psi_prime / xi + np.exp(-psi)], dtype=float)


def rk4_step_isothermal(xi: float, state: np.ndarray, h: float) -> np.ndarray:
    """One classical RK4 step for the isothermal equation."""
    k1 = isothermal_rhs(xi, state)
    k2 = isothermal_rhs(xi + h / 2.0, state + h * k1 / 2.0)
    k3 = isothermal_rhs(xi + h / 2.0, state + h * k2 / 2.0)
    k4 = isothermal_rhs(xi + h, state + h * k3)
    return state + h * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0


def solve_isothermal_rk4(
    epsilon: float = 1e-6,
    h: float = 1e-3,
    xi_max: float = 50.0,
    max_steps: int = 1_000_000,
) -> IsothermalSolution:
    """Solve the isothermal Lane-Emden equation using RK4.

    The isothermal sphere has no finite boundary (xi_1 = infinity).
    Integration proceeds to xi_max.

    Parameters
    ----------
    epsilon:
        Starting point near xi=0.
    h:
        Fixed step size.
    xi_max:
        Maximum xi for integration.
    max_steps:
        Safety cap on number of steps.

    Returns
    -------
    IsothermalSolution.
    """
    if epsilon <= 0:
        raise ValueError("epsilon must be positive.")
    if xi_max <= epsilon:
        raise ValueError("xi_max must be greater than epsilon.")

    psi0, psi_prime0 = taylor_isothermal(epsilon)

    n_steps_est = int(np.ceil((xi_max - epsilon) / h))
    xi_list = [epsilon]
    psi_list = [psi0]
    psi_prime_list = [psi_prime0]

    xi = epsilon
    state = np.array([psi0, psi_prime0], dtype=float)
    step_count = 0

    while xi < xi_max and step_count < max_steps:
        step = min(h, xi_max - xi)
        new_state = rk4_step_isothermal(xi, state, step)
        xi += step
        state = new_state
        step_count += 1

        xi_list.append(xi)
        psi_list.append(float(state[0]))
        psi_prime_list.append(float(state[1]))

    return IsothermalSolution(
        xi=np.array(xi_list, dtype=float),
        psi=np.array(psi_list, dtype=float),
        psi_prime=np.array(psi_prime_list, dtype=float),
        num_steps=step_count,
        xi_max=xi_max,
    )


def isothermal_asymptotic(xi: np.ndarray) -> np.ndarray:
    """Asymptotic solution for the isothermal sphere at large xi.

    psi ~ ln(xi^2/2) - 2*ln(ln(xi^2/2)) + ...
    Valid for xi >> 1.
    """
    xi_safe = np.maximum(xi, 1.1)
    log_arg = xi_safe ** 2 / 2.0
    return np.log(log_arg) - 2.0 * np.log(np.maximum(np.log(log_arg), 1e-10))


def solve_isothermal_fd(
    epsilon: float = 1e-6,
    num_intervals: int = 1000,
    xi_max: float = 50.0,
    max_iterations: int = 50,
    tolerance: float = 1e-10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
    """Solve the isothermal equation using finite differences + Newton.

    The discretized equation at interior point j is:
        (psi_{j+1} - 2*psi_j + psi_{j-1})/h^2
        + (psi_{j+1} - psi_{j-1})/(xi_j * h)
        - exp(-psi_j) = 0

    Returns (xi, psi, psi_prime, converged).
    """
    psi_left, _ = taylor_isothermal(epsilon)
    xi = np.linspace(epsilon, xi_max, num_intervals + 1)
    h = xi[1] - xi[0]

    # Initial guess: blend between taylor and asymptotic
    psi_guess = np.zeros(num_intervals + 1)
    psi_guess[0] = psi_left
    for j in range(1, num_intervals + 1):
        t = xi[j] / xi_max
        psi_guess[j] = psi_left * (1 - t) + isothermal_asymptotic(np.array([xi[j]]))[0] * t

    # Newton iteration (tridiagonal Jacobian)
    for iteration in range(max_iterations):
        # Build residual at interior points
        psi = psi_guess
        residual = np.zeros(num_intervals - 1)
        for j in range(1, num_intervals):
            residual[j - 1] = (
                (psi[j + 1] - 2.0 * psi[j] + psi[j - 1]) / h ** 2
                + (psi[j + 1] - psi[j - 1]) / (xi[j] * h)
                - np.exp(-psi[j])
            )

        res_norm = float(np.linalg.norm(residual, ord=np.inf))
        if res_norm < tolerance:
            psi_prime = np.gradient(psi, xi, edge_order=2)
            return xi, psi, psi_prime, True

        # Build tridiagonal Jacobian
        lower = np.zeros(num_intervals - 1)
        diag = np.zeros(num_intervals - 1)
        upper = np.zeros(num_intervals - 1)

        for j in range(1, num_intervals):
            idx = j - 1
            diag[idx] = -2.0 / h ** 2 + np.exp(-psi[j])
            if idx > 0:
                lower[idx - 1] = 1.0 / h ** 2 - 1.0 / (xi[j] * h)
                upper[idx - 1] = 1.0 / h ** 2 + 1.0 / (xi[j] * h)

        # Thomas algorithm
        a = lower.copy()
        b = diag.copy()
        c_arr = upper.copy()
        d = -residual.copy()

        for i in range(1, num_intervals - 1):
            factor = a[i - 1] / b[i - 1]
            b[i] -= factor * c_arr[i - 1]
            d[i] -= factor * d[i - 1]

        delta = np.zeros(num_intervals - 1)
        delta[-1] = d[-1] / b[-1]
        for i in range(num_intervals - 3, -1, -1):
            delta[i] = (d[i] - c_arr[i] * delta[i + 1]) / b[i]

        # Damped update
        damping = 1.0
        while damping > 1e-8:
            candidate = psi.copy()
            candidate[1:-1] += damping * delta
            if np.all(candidate[1:-1] >= 0):
                psi_guess = candidate
                break
            damping *= 0.5

        if damping <= 1e-8:
            psi_guess[1:-1] += damping * delta
            break

    psi_prime = np.gradient(psi_guess, xi, edge_order=2)
    return xi, psi_guess, psi_prime, False


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from pathlib import Path

    print("=" * 70)
    print("Isothermal Sphere (n -> infinity) Solver")
    print("=" * 70)

    # Solve with RK4
    print("\nSolving isothermal sphere with RK4...")
    sol = solve_isothermal_rk4(epsilon=1e-4, h=1e-2, xi_max=30.0)
    print(f"  Steps: {sol.num_steps}")
    print(f"  psi(xi_max={sol.xi_max}) = {sol.psi[-1]:.6f}")
    print(f"  Asymptotic at xi_max: {isothermal_asymptotic(np.array([sol.xi_max]))[0]:.6f}")

    # Solve with FD
    print("\nSolving isothermal sphere with FD...")
    xi_fd, psi_fd, psi_prime_fd, converged = solve_isothermal_fd(
        epsilon=1e-3, num_intervals=500, xi_max=30.0,
    )
    print(f"  Converged: {converged}")
    print(f"  psi(xi_max) = {psi_fd[-1]:.6f}")

    # Plot
    output_dir = Path(__file__).resolve().parent / "output_initial"
    output_dir.mkdir(exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # psi vs xi
    ax1.plot(sol.xi, sol.psi, linewidth=1.5, label="Isothermal RK4")
    ax1.plot(xi_fd, psi_fd, "--", linewidth=1.5, label="Isothermal FD")
    xi_asy = np.logspace(0.3, 1.5, 200)
    ax1.plot(xi_asy, isothermal_asymptotic(xi_asy), ":", linewidth=1.5, label="Asymptotic")
    ax1.set_xlabel(r"$\xi$")
    ax1.set_ylabel(r"$\psi(\xi)$")
    ax1.set_title("Isothermal Sphere Solution")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Density profile: rho/rho_c = exp(-psi)
    ax2.semilogy(sol.xi, np.exp(-sol.psi), linewidth=1.5, label="Isothermal RK4")
    ax2.set_xlabel(r"$\xi$")
    ax2.set_ylabel(r"$\rho/\rho_c = e^{-\psi}$")
    ax2.set_title("Isothermal Density Profile")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "isothermal_overview.png", dpi=200)
    plt.close()
    print(f"\nPlot saved to {output_dir / 'isothermal_overview.png'}")
