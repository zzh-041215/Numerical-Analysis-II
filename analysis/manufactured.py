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
class ManufacturedSolution:
    """A manufactured solution for verifying Lane-Emden solvers.

    Given a chosen smooth function theta_man(xi) that satisfies the boundary
    conditions theta(0)=1, theta'(0)=0, we compute the source term
        S(xi) = theta_man'' + (2/xi) * theta_man' + theta_man^n
    such that theta_man exactly solves the modified equation
        theta'' + (2/xi) * theta' + theta^n = S(xi).

    Any numerical solver can then be tested by solving this modified equation
    and comparing with the known theta_man. This allows rigorous convergence
    order verification for arbitrary n values.
    """
    n: float
    xi_max: float

    def theta(self, xi: np.ndarray) -> np.ndarray:
        """The manufactured solution theta_man(xi)."""
        raise NotImplementedError

    def theta_prime(self, xi: np.ndarray) -> np.ndarray:
        """Analytic derivative of theta_man."""
        raise NotImplementedError

    def theta_double_prime(self, xi: np.ndarray) -> np.ndarray:
        """Analytic second derivative of theta_man."""
        raise NotImplementedError

    def source_term(self, xi: np.ndarray) -> np.ndarray:
        """Compute the forcing source term S(xi)."""
        tp = self.theta_prime(xi)
        tpp = self.theta_double_prime(xi)
        t = self.theta(xi)
        return tpp + (2.0 / xi) * tp + t ** self.n


class CosineBumpSolution(ManufacturedSolution):
    """theta(xi) = 1 + A * (cos(pi * xi / xi_max) - 1) / 2.

    Satisfies theta(0) = 1, theta'(0) = 0 (by evenness of cosine).
    The amplitude A controls how much theta varies.
    """

    def __init__(self, n: float, xi_max: float, amplitude: float = 0.9):
        super().__init__(n, xi_max)
        self.A = amplitude
        self._k = np.pi / xi_max

    def theta(self, xi: np.ndarray) -> np.ndarray:
        return 1.0 + self.A * (np.cos(self._k * xi) - 1.0) / 2.0

    def theta_prime(self, xi: np.ndarray) -> np.ndarray:
        return -self.A * self._k * np.sin(self._k * xi) / 2.0

    def theta_double_prime(self, xi: np.ndarray) -> np.ndarray:
        return -self.A * self._k ** 2 * np.cos(self._k * xi) / 2.0


class ExponentialDecaySolution(ManufacturedSolution):
    """theta(xi) = exp(-alpha * xi^2) + beta * xi^2 * exp(-gamma * xi).

    Chosen to satisfy BCs and decay monotonically, resembling real LE solutions.
    The parameters are tuned to keep theta positive over [0, xi_max].
    """

    def __init__(
        self, n: float, xi_max: float,
        alpha: float = 0.05, beta: float = 0.0, gamma: float = 1.0,
    ):
        super().__init__(n, xi_max)
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def theta(self, xi: np.ndarray) -> np.ndarray:
        return (
            np.exp(-self.alpha * xi ** 2)
            + self.beta * xi ** 2 * np.exp(-self.gamma * xi)
        )

    def theta_prime(self, xi: np.ndarray) -> np.ndarray:
        term1 = -2.0 * self.alpha * xi * np.exp(-self.alpha * xi ** 2)
        term2 = self.beta * xi * np.exp(-self.gamma * xi) * (2.0 - self.gamma * xi)
        return term1 + term2

    def theta_double_prime(self, xi: np.ndarray) -> np.ndarray:
        e_a = np.exp(-self.alpha * xi ** 2)
        term1 = 2.0 * self.alpha * (2.0 * self.alpha * xi ** 2 - 1.0) * e_a
        e_g = np.exp(-self.gamma * xi)
        term2 = self.beta * e_g * (
            2.0 - 4.0 * self.gamma * xi + self.gamma ** 2 * xi ** 2
        )
        return term1 + term2


class PolynomialBumpSolution(ManufacturedSolution):
    """theta(xi) = 1 + a2*xi^2 + a4*xi^4 + a6*xi^6.

    Simple polynomial that satisfies BCs exactly. Coefficients chosen to make
    theta(xi_max) > 0 and give a physically plausible shape.
    """

    def __init__(self, n: float, xi_max: float):
        super().__init__(n, xi_max)
        # Choose coefficients: theta(xi_max) should be positive but small
        L = xi_max
        # Solve for a2, a4, a6 such that:
        # theta(L) = target, theta(L) smooth
        target = 0.1
        # Simple choice: a2 = -1/6 (like Taylor), a4 = n/120, a6 = correction
        self.a2 = -1.0 / 6.0
        self.a4 = n / 120.0
        # Set a6 so theta(L) = target
        self.a6 = (target - 1.0 - self.a2 * L ** 2 - self.a4 * L ** 4) / L ** 6

    def theta(self, xi: np.ndarray) -> np.ndarray:
        x2 = xi ** 2
        return 1.0 + self.a2 * x2 + self.a4 * x2 ** 2 + self.a6 * x2 ** 3

    def theta_prime(self, xi: np.ndarray) -> np.ndarray:
        return 2.0 * self.a2 * xi + 4.0 * self.a4 * xi ** 3 + 6.0 * self.a6 * xi ** 5

    def theta_double_prime(self, xi: np.ndarray) -> np.ndarray:
        return 2.0 * self.a2 + 12.0 * self.a4 * xi ** 2 + 30.0 * self.a6 * xi ** 4


def solve_manufactured_rk4(
    manufactured: ManufacturedSolution,
    epsilon: float = 1e-4,
    h: float = 1e-3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Solve the manufactured Lane-Emden equation using RK4.

    The modified system is:
        y1' = y2
        y2' = -2*y2/xi - y1^n + S(xi)
    where S(xi) is the source term from the manufactured solution.

    Returns (xi, theta, theta_prime) arrays.
    """
    # Taylor initial conditions (inline to avoid import issues)
    n = manufactured.n
    theta0 = 1.0 - epsilon**2 / 6.0 + n * epsilon**4 / 120.0
    theta_prime0 = -epsilon / 3.0 + n * epsilon**3 / 30.0

    n_steps = max(10, int(np.ceil((manufactured.xi_max - epsilon) / h)))
    h_actual = (manufactured.xi_max - epsilon) / n_steps

    xi_arr = np.linspace(epsilon, manufactured.xi_max, n_steps + 1)
    theta_arr = np.zeros(n_steps + 1)
    theta_prime_arr = np.zeros(n_steps + 1)
    theta_arr[0] = theta0
    theta_prime_arr[0] = theta_prime0

    state = np.array([theta0, theta_prime0], dtype=float)

    for i in range(n_steps):
        xi_i = xi_arr[i]
        h_step = h_actual

        # RK4 step with source term
        def rhs(xi_s, s):
            t, tp = s
            if t < 0 and abs(float(n) - round(float(n))) > 1e-10:
                t_pow = 0.0
            else:
                t_pow = float(abs(t)) ** n
            src = manufactured.source_term(np.array([float(xi_s)]))[0]
            return np.array([
                float(tp),
                -2.0 * float(tp) / float(xi_s) - t_pow + src,
            ])

        k1 = rhs(xi_i, state)
        k2 = rhs(xi_i + h_step / 2, state + h_step * k1 / 2)
        k3 = rhs(xi_i + h_step / 2, state + h_step * k2 / 2)
        k4 = rhs(xi_i + h_step, state + h_step * k3)

        state = state + h_step * (k1 + 2 * k2 + 2 * k3 + k4) / 6
        theta_arr[i + 1] = float(state[0])
        theta_prime_arr[i + 1] = float(state[1])

    return xi_arr, theta_arr, theta_prime_arr


def verify_convergence_order(
    manufactured: ManufacturedSolution,
    solver_func,
    h_values: list[float],
    epsilon: float = 1e-4,
    eval_points: int = 2000,
) -> dict:
    """Verify the convergence order of a solver using a manufactured solution.

    Parameters
    ----------
    manufactured:
        The manufactured solution.
    solver_func:
        Function with signature solver_func(manufactured, epsilon, h) -> (xi, theta, theta').
    h_values:
        List of step sizes to test.
    epsilon:
        Starting point.
    eval_points:
        Number of evaluation points for error computation.

    Returns
    -------
    dict with errors, hs, observed_order, r_squared.
    """
    errors = []
    effective_hs = []

    xi_eval = np.linspace(epsilon, manufactured.xi_max, eval_points)
    theta_exact = manufactured.theta(xi_eval)

    for h in h_values:
        try:
            xi_num, theta_num, _ = solver_func(manufactured, epsilon, h)
            theta_interp = np.interp(xi_eval, xi_num, theta_num)
            err = float(np.max(np.abs(theta_interp - theta_exact)))
            errors.append(err)

            # Effective step size
            n_intervals = int(np.ceil((manufactured.xi_max - epsilon) / h))
            effective_hs.append((manufactured.xi_max - epsilon) / n_intervals)
        except Exception as e:
            print(f"    [WARNING] h={h:.0e} failed: {e}")
            continue

    if len(errors) < 2:
        return {"errors": errors, "hs": effective_hs, "observed_order": None}

    log_h = np.log(effective_hs)
    log_err = np.log(errors)
    coeffs = np.polyfit(log_h, log_err, 1)
    p = float(coeffs[0])

    predicted = np.polyval(coeffs, log_h)
    ss_res = np.sum((log_err - predicted) ** 2)
    ss_tot = np.sum((log_err - np.mean(log_err)) ** 2)
    r_squared = float(1 - ss_res / ss_tot) if ss_tot > 0 else 1.0

    return {
        "errors": errors,
        "hs": effective_hs,
        "observed_order": p,
        "r_squared": r_squared,
    }


if __name__ == "__main__":
    import math

    print("=" * 70)
    print("Manufactured Solution Verification (MMS)")
    print("=" * 70)

    for n_test in (0.0, 1.5, 3.0):
        xi_max_test = 5.0
        print(f"\n--- n = {n_test:g}, xi_max = {xi_max_test} ---")

        # Use cosine bump solution
        ms = CosineBumpSolution(n_test, xi_max_test, amplitude=0.5)

        # Verify that the manufactured solution satisfies BCs
        xi_check = np.array([0.0, xi_max_test])
        print(f"  theta(0) = {ms.theta(np.array([0.0]))[0]:.6f}")
        print(f"  theta'(0) = {ms.theta_prime(np.array([1e-8]))[0]:.2e}")
        print(f"  theta(xi_max) = {ms.theta(np.array([xi_max_test]))[0]:.6f}")

        # Convergence study
        h_vals = [0.2, 0.1, 0.05, 0.025, 0.0125]
        result = verify_convergence_order(ms, solve_manufactured_rk4, h_vals)

        if result["observed_order"] is not None:
            print(f"  Observed order: p = {result['observed_order']:.3f} (R^2={result['r_squared']:.4f})")
            print(f"  Expected order: p = 4.0 (RK4)")
            print(f"  Errors: {', '.join(f'{e:.2e}' for e in result['errors'])}")
        else:
            print("  Insufficient data for order estimation")

    # Test with non-integer n to demonstrate MMS capability
    print(f"\n--- Testing non-integer n values ---")
    for n_test in (1.5, 2.5, 3.5):
        xi_max_test = 5.0
        ms = CosineBumpSolution(n_test, xi_max_test, amplitude=0.5)
        h_vals = [0.2, 0.1, 0.05, 0.025]
        result = verify_convergence_order(ms, solve_manufactured_rk4, h_vals)
        if result["observed_order"] is not None:
            print(f"  n={n_test:g}: p = {result['observed_order']:.3f} (R^2={result['r_squared']:.4f})")
        else:
            print(f"  n={n_test:g}: insufficient data")

    # Convergence order verification plot
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from pathlib import Path

    OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output_initial"
    OUTPUT_DIR.mkdir(exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Convergence curves for multiple n
    colors = {0.0: "blue", 1.5: "green", 3.0: "red"}
    h_vals_fine = [0.2, 0.1, 0.05, 0.025, 0.0125]
    for n_test in (0.0, 1.5, 3.0):
        ms = CosineBumpSolution(n_test, 5.0, amplitude=0.5)
        result = verify_convergence_order(ms, solve_manufactured_rk4, h_vals_fine)
        if result["errors"]:
            ax1.loglog(result["hs"], result["errors"], "o-",
                       color=colors[n_test], linewidth=1.5,
                       label=f"n={n_test:g} (p={result['observed_order']:.2f})" if result['observed_order'] else f"n={n_test:g}")

    # Reference O(h^2) line
    h_ref = np.array([0.01, 0.3])
    ax1.loglog(h_ref, 0.5 * h_ref**2, "k:", linewidth=1, alpha=0.5, label=r"$\sim h^2$")
    ax1.loglog(h_ref, 0.01 * h_ref**4, "k--", linewidth=1, alpha=0.5, label=r"$\sim h^4$")
    ax1.set_xlabel("h")
    ax1.set_ylabel(r"$\|\theta_{num} - \theta_{MMS}\|_\infty$")
    ax1.set_title("MMS Convergence Verification")
    ax1.legend(fontsize=8)
    ax1.grid(True, which="both", alpha=0.3)

    # Observed order vs n
    n_range = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
    orders = []
    n_ok = []
    for n_test in n_range:
        ms = CosineBumpSolution(n_test, 5.0, amplitude=0.5)
        result = verify_convergence_order(ms, solve_manufactured_rk4, h_vals_fine[:4])
        if result["observed_order"] is not None:
            n_ok.append(n_test)
            orders.append(result["observed_order"])

    ax2.plot(n_ok, orders, "o-", linewidth=1.5)
    ax2.axhline(4.0, color="red", linestyle="--", alpha=0.5, label="Expected RK4 (p=4)")
    ax2.axhline(2.0, color="orange", linestyle="--", alpha=0.5, label="Observed (p≈2)")
    ax2.set_xlabel("n")
    ax2.set_ylabel("Observed convergence order")
    ax2.set_title("Convergence Order vs Polytropic Index")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 5)

    fig.suptitle("Manufactured Solution Verification (MMS)", fontsize=13)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "manufactured_convergence.png", dpi=200)
    plt.close()
    print(f"\nPlot saved to {OUTPUT_DIR / 'manufactured_convergence.png'}")
