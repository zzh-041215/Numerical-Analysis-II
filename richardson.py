from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from data_input import load_reference_data


@dataclass(frozen=True)
class RichardsonResult:
    """Result of Richardson extrapolation on a Lane-Emden solution pair."""
    n: float
    xi: np.ndarray
    theta: np.ndarray
    theta_prime: np.ndarray
    first_zero: Optional[float]
    error_estimate: Optional[np.ndarray]
    method_order: int


def richardson_extrapolate(
    coarse_xi: np.ndarray,
    coarse_theta: np.ndarray,
    coarse_theta_prime: np.ndarray,
    fine_xi: np.ndarray,
    fine_theta: np.ndarray,
    fine_theta_prime: np.ndarray,
    n: float,
    coarse_first_zero: Optional[float],
    fine_first_zero: Optional[float],
    method_order: int,
) -> RichardsonResult:
    """Perform Richardson extrapolation on two solutions at different resolutions.

    For a method of order p, the extrapolated value is:
        theta_rich = (2^p * theta_fine - theta_coarse) / (2^p - 1)

    The error estimate is:
        |theta_rich - theta_fine|
    which approximates the remaining error after extrapolation.

    Parameters
    ----------
    coarse_xi, coarse_theta, coarse_theta_prime:
        Solution on the coarser grid (step size h).
    fine_xi, fine_theta, fine_theta_prime:
        Solution on the finer grid (step size h/2).
    n:
        Polytropic index (for metadata).
    coarse_first_zero, fine_first_zero:
        First zero locations from each grid.
    method_order:
        Theoretical order of the underlying method (4 for RK4, 2 for FD).

    Returns
    -------
    RichardsonResult with the extrapolated solution on the fine grid.
    """
    if method_order <= 0:
        raise ValueError("method_order must be positive.")

    # Interpolate coarse solution onto the fine grid
    coarse_theta_on_fine = np.interp(fine_xi, coarse_xi, coarse_theta)
    coarse_theta_prime_on_fine = np.interp(fine_xi, coarse_xi, coarse_theta_prime)

    # Richardson extrapolation formula
    factor = 2.0 ** method_order
    weight_fine = factor / (factor - 1.0)
    weight_coarse = -1.0 / (factor - 1.0)

    theta_rich = weight_fine * fine_theta + weight_coarse * coarse_theta_on_fine
    theta_prime_rich = weight_fine * fine_theta_prime + weight_coarse * coarse_theta_prime_on_fine

    # Error estimate: difference between extrapolated and fine solution
    error_estimate = np.abs(theta_rich - fine_theta)

    # Extrapolate first zero
    first_zero_rich: Optional[float] = None
    if coarse_first_zero is not None and fine_first_zero is not None:
        first_zero_rich = weight_fine * fine_first_zero + weight_coarse * coarse_first_zero

    return RichardsonResult(
        n=n,
        xi=fine_xi.copy(),
        theta=theta_rich,
        theta_prime=theta_prime_rich,
        first_zero=first_zero_rich,
        error_estimate=error_estimate,
        method_order=method_order,
    )


def richardson_extrapolate_rk4(
    coarse_solution,
    fine_solution,
) -> RichardsonResult:
    """Convenience wrapper for RK4 Richardson extrapolation (p=4).

    Expects solutions with attributes: xi, theta, theta_prime, first_zero, n.
    The fine solution should use step size h/2 relative to coarse.
    """
    return richardson_extrapolate(
        coarse_xi=coarse_solution.xi,
        coarse_theta=coarse_solution.theta,
        coarse_theta_prime=coarse_solution.theta_prime,
        fine_xi=fine_solution.xi,
        fine_theta=fine_solution.theta,
        fine_theta_prime=fine_solution.theta_prime,
        n=coarse_solution.n,
        coarse_first_zero=coarse_solution.first_zero,
        fine_first_zero=fine_solution.first_zero,
        method_order=4,
    )


def richardson_extrapolate_fd(
    coarse_solution,
    fine_solution,
) -> RichardsonResult:
    """Convenience wrapper for FD Richardson extrapolation (p=2).

    Expects solutions with attributes: xi, theta, theta_prime, first_zero, n.
    The fine solution should use step size h/2 relative to coarse.
    """
    return richardson_extrapolate(
        coarse_xi=coarse_solution.xi,
        coarse_theta=coarse_solution.theta,
        coarse_theta_prime=coarse_solution.theta_prime,
        fine_xi=fine_solution.xi,
        fine_theta=fine_solution.theta,
        fine_theta_prime=fine_solution.theta_prime,
        n=coarse_solution.n,
        coarse_first_zero=coarse_solution.first_zero,
        fine_first_zero=fine_solution.first_zero,
        method_order=2,
    )


def compute_convergence_order(
    errors: list[float],
    hs: list[float],
) -> tuple[float, float]:
    """Estimate convergence order and its 95% confidence interval from error data.

    Uses log-log linear regression: log(error) = p * log(h) + C
    Returns (observed_order, r_squared).
    """
    if len(errors) < 2:
        raise ValueError("Need at least two data points to estimate order.")

    log_h = np.log(hs)
    log_err = np.log(errors)
    coeffs = np.polyfit(log_h, log_err, 1)
    p = float(coeffs[0])

    # Compute R² for goodness-of-fit
    predicted = np.polyval(coeffs, log_h)
    ss_res = np.sum((log_err - predicted) ** 2)
    ss_tot = np.sum((log_err - np.mean(log_err)) ** 2)
    r_squared = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 1.0

    return p, r_squared


def verify_richardson_order(
    exact_theta_func,
    n: float,
    rk4_module,
    fd_module,
    epsilons: tuple[float, ...] = (1e-3, 1e-4, 1e-5),
    h_pairs: tuple[tuple[float, float], ...] = (
        (1e-2, 5e-3),
        (5e-3, 2.5e-3),
        (2.5e-3, 1.25e-3),
    ),
    xi_max: Optional[float] = None,
) -> dict:
    """Systematic verification of Richardson extrapolation effectiveness.

    For each method (RK4, FD), runs pairs of coarse/fine solutions,
    applies Richardson extrapolation, and measures the convergence order
    before and after extrapolation.

    Returns a dict with all results for analysis.
    """
    import math

    if xi_max is None:
        if abs(n) < 1e-14:
            xi_max = math.sqrt(6.0)
        elif abs(n - 1.0) < 1e-14:
            xi_max = math.pi
        elif abs(n - 5.0) < 1e-14:
            xi_max = 10.0
        else:
            ref = load_reference_data()
            xi_max = ref.get_first_zero(n)
            if xi_max is None:
                raise ValueError(f"Cannot determine xi_max for n={n}.")

    results = {"n": n, "rk4": {}, "fd": {}}

    for method_name, module, base_order in [
        ("rk4", rk4_module, 4),
        ("fd", fd_module, 2),
    ]:
        raw_errors = []
        rich_errors = []
        hs_fine = []

        for h_coarse, h_fine in h_pairs:
            best_error = float("inf")
            best_epsilon = epsilons[0]

            # Find best epsilon for this h pair
            for eps in epsilons:
                try:
                    if method_name == "rk4":
                        sol = module.solve_lane_emden_rk4(
                            n=n, epsilon=eps, h=h_fine, xi_max=xi_max,
                        )
                    else:
                        num_intervals = max(20, int(math.ceil((xi_max - eps) / h_fine)))
                        sol = module.solve_lane_emden_finite_difference(
                            n=n, epsilon=eps, num_intervals=num_intervals,
                            xi_max=xi_max, theta_right=0.0,
                        )
                    xi_eval = np.linspace(eps, xi_max, 2000)
                    theta_num = np.interp(xi_eval, sol.xi, sol.theta)
                    theta_ref = exact_theta_func(xi_eval)
                    err = float(np.max(np.abs(theta_num - theta_ref)))
                    if err < best_error:
                        best_error = err
                        best_epsilon = eps
                except Exception:
                    continue

            if best_error == float("inf"):
                continue

            # Run coarse and fine solutions with best epsilon
            try:
                if method_name == "rk4":
                    coarse = module.solve_lane_emden_rk4(
                        n=n, epsilon=best_epsilon, h=h_coarse, xi_max=xi_max,
                    )
                    fine = module.solve_lane_emden_rk4(
                        n=n, epsilon=best_epsilon, h=h_fine, xi_max=xi_max,
                    )
                    rich = richardson_extrapolate_rk4(coarse, fine)
                else:
                    n_coarse = max(20, int(math.ceil((xi_max - best_epsilon) / h_coarse)))
                    n_fine = max(20, int(math.ceil((xi_max - best_epsilon) / h_fine)))
                    coarse = module.solve_lane_emden_finite_difference(
                        n=n, epsilon=best_epsilon, num_intervals=n_coarse,
                        xi_max=xi_max, theta_right=0.0,
                    )
                    fine = module.solve_lane_emden_finite_difference(
                        n=n, epsilon=best_epsilon, num_intervals=n_fine,
                        xi_max=xi_max, theta_right=0.0,
                    )
                    rich = richardson_extrapolate_fd(coarse, fine)
            except Exception:
                continue

            # Compute errors
            xi_eval = np.linspace(best_epsilon, xi_max, 2000)
            theta_ref = exact_theta_func(xi_eval)

            fine_theta_interp = np.interp(xi_eval, fine.xi, fine.theta)
            rich_theta_interp = np.interp(xi_eval, rich.xi, rich.theta)

            raw_err = float(np.max(np.abs(fine_theta_interp - theta_ref)))
            rich_err = float(np.max(np.abs(rich_theta_interp - theta_ref)))

            raw_errors.append(raw_err)
            rich_errors.append(rich_err)
            hs_fine.append(h_fine)

        if len(raw_errors) >= 2:
            p_raw, r2_raw = compute_convergence_order(raw_errors, hs_fine)
            p_rich, r2_rich = compute_convergence_order(rich_errors, hs_fine) if len(rich_errors) >= 2 else (None, None)
            results[method_name] = {
                "raw_errors": raw_errors,
                "rich_errors": rich_errors,
                "hs": hs_fine,
                "observed_order_raw": p_raw,
                "r_squared_raw": r2_raw,
                "observed_order_rich": p_rich,
                "r_squared_rich": r2_rich,
                "expected_order_raw": base_order,
                "expected_order_rich": base_order + 2,
            }

    return results


if __name__ == "__main__":
    import importlib.util
    import sys
    from pathlib import Path

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    BASE = Path(__file__).resolve().parent
    OUTPUT_DIR = BASE / "output_initial"
    OUTPUT_DIR.mkdir(exist_ok=True)

    # Load solver modules
    for mod_name, mod_path in [
        ("rk_solver", BASE / "Ronge-Kutta.py"),
        ("fd_solver", BASE / "finite-difference.py"),
    ]:
        spec = importlib.util.spec_from_file_location(mod_name, mod_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[mod_name] = module
        spec.loader.exec_module(module)

    rk4_mod = sys.modules["rk_solver"]
    fd_mod = sys.modules["fd_solver"]

    import math

    print("=" * 70)
    print("Richardson Extrapolation Verification")
    print("=" * 70)

    for n_val in (0.0, 1.0, 5.0):
        if abs(n_val) < 1e-14:
            xi_max_val = math.sqrt(6.0)
            exact_theta = lambda xi, nv=0.0: 1.0 - xi**2 / 6.0
        elif abs(n_val - 1.0) < 1e-14:
            xi_max_val = math.pi
            exact_theta = lambda xi, nv=1.0: np.ones_like(xi) if np.isscalar(xi) or xi[0] == 0 else np.sin(xi) / xi
        else:
            xi_max_val = 10.0
            exact_theta = lambda xi, nv=5.0: 1.0 / np.sqrt(1.0 + xi**2 / 3.0)

        if abs(n_val - 1.0) < 1e-14:
            def exact_theta_n1(xi):
                result = np.ones_like(xi, dtype=float)
                mask = xi != 0.0
                result[mask] = np.sin(xi[mask]) / xi[mask]
                return result
            current_exact = exact_theta_n1
        else:
            current_exact = exact_theta

        print(f"\n--- n = {n_val:g} ---")

        results = verify_richardson_order(
            current_exact, n_val, rk4_mod, fd_mod, xi_max=xi_max_val,
        )

        for method in ("rk4", "fd"):
            data = results[method]
            if "observed_order_raw" not in data:
                print(f"  {method}: insufficient successful runs")
                continue
            print(f"  {method}:")
            print(f"    Raw order:       p = {data['observed_order_raw']:.3f} (R^2={data['r_squared_raw']:.4f}), expected p = {data['expected_order_raw']}")
            if data["observed_order_rich"] is not None:
                print(f"    Richardson order: p = {data['observed_order_rich']:.3f} (R^2={data['r_squared_rich']:.4f}), expected p ~ {data['expected_order_rich']}")
                # Improvement factor
                if data["raw_errors"] and data["rich_errors"]:
                    ratio = data["raw_errors"][-1] / data["rich_errors"][-1] if data["rich_errors"][-1] > 0 else float("inf")
                    print(f"    Error reduction at finest h: {ratio:.1f}x")

    # Generate convergence comparison plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for col, n_val in enumerate((0.0, 1.0, 5.0)):
        ax = axes[col]
        if abs(n_val) < 1e-14:
            current_exact = lambda xi: 1.0 - xi**2 / 6.0
            xi_max_val = math.sqrt(6.0)
        elif abs(n_val - 1.0) < 1e-14:
            def current_exact(xi):
                r = np.ones_like(xi, dtype=float)
                m = xi != 0
                r[m] = np.sin(xi[m]) / xi[m]
                return r
            xi_max_val = math.pi
        else:
            current_exact = lambda xi: 1.0 / np.sqrt(1.0 + xi**2 / 3.0)
            xi_max_val = 10.0

        results = verify_richardson_order(current_exact, n_val, rk4_mod, fd_mod, xi_max=xi_max_val)
        for method, marker, label in [("rk4", "o-", "RK4"), ("fd", "s--", "FD")]:
            data = results.get(method, {})
            if "raw_errors" in data and data["raw_errors"]:
                ax.loglog(data["hs"], data["raw_errors"], marker + "k", linewidth=1.2, alpha=0.4, label=f"{label} raw")
            if "rich_errors" in data and data["rich_errors"]:
                ax.loglog(data["hs"], data["rich_errors"], marker, linewidth=1.5, label=f"{label} Richardson")
        ax.set_xlabel("h")
        ax.set_ylabel(r"$\|\theta_{num} - \theta_{exact}\|_\infty$")
        ax.set_title(f"n = {n_val:g}")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8)

    fig.suptitle("Richardson Extrapolation: Error vs Step Size", fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "richardson_convergence.png", dpi=200, bbox_inches="tight")
    plt.close()
    print(f"\nPlot saved to {OUTPUT_DIR / 'richardson_convergence.png'}")
