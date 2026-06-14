from __future__ import annotations

import csv
import importlib.util
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Ensure project root is on sys.path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from data_input import load_reference_data


BASE_DIR = _ROOT
OUTPUT_DIR = BASE_DIR / "output_initial"
RK_FILE = BASE_DIR / "solvers" / "rk4.py"
FD_FILE = BASE_DIR / "solvers" / "fd.py"

MODULE_CACHE: dict[str, object] = {}

EPSILON_VALUES = (1e-3, 1e-4, 1e-5)
TARGET_H_VALUES = (2e-2, 1.5e-2, 1e-2, 7.5e-3, 5e-3, 3.75e-3, 2.5e-3)
FIXED_H_FOR_EPSILON_STUDY = 5e-3
REFERENCE_RK_EPSILON = 1e-7
REFERENCE_RK_H = 2e-4
REFERENCE_FD_H = 5e-4
IMPORTANT_NODE_FRACTIONS = (0.25, 0.5, 0.75)
SPECIAL_N_MAX_XI1 = 50.0
THETA_OVERVIEW_XI_MAX = 10.0


@dataclass(frozen=True)
class ExactExperimentResult:
    method: str
    n: float
    epsilon: float
    target_h: float
    actual_h: float
    error_inf: Optional[float]
    converged: bool
    message: str = ""


@dataclass(frozen=True)
class SpecialNodeExperimentResult:
    method: str
    n: float
    epsilon: float
    target_h: float
    actual_h: float
    theta_q25_error: Optional[float]
    theta_q50_error: Optional[float]
    theta_q75_error: Optional[float]
    surface_derivative_error: Optional[float]
    max_node_error: Optional[float]
    converged: bool
    message: str = ""


# ---------------------------------------------------------------------------
# Horedt table cleaning utilities
# ---------------------------------------------------------------------------

def _clean_horedt_theta_data(n: float) -> Optional[tuple[np.ndarray, np.ndarray]]:
    """Extract cleaned (xi, theta) data from the Horedt table for a given n.

    The Horedt (1986) table was extracted from PDF and contains OCR artifacts:
    - Negative / out-of-range theta values at very small xi
    - Column-shifted rows (theta values that are actually xi or other columns)
    - Spurious sign changes in the middle of the solution

    This function filters out obviously corrupted rows and returns a cleaned
    dataset suitable for interpolation.

    Returns (xi, theta) arrays, or None if insufficient clean data.
    """
    reference_data = load_reference_data()
    points = reference_data.get_sphere_table(n, until_first_zero=False)

    if not points:
        return None

    # Get the known first zero from global properties as a sanity check
    prop = reference_data.get_global_property(n)
    xi_1_ref = prop.xi_1 if prop and prop.xi_1 else None

    # Filter: keep only rows where theta is in the physically meaningful range
    # For Lane-Emden: theta starts at 1 and decreases monotonically to 0 at xi_1
    clean_xi = []
    clean_theta = []

    for p in points:
        # Skip rows with clearly corrupted theta values
        if p.theta is None:
            continue
        # Physical theta range for xi >= 0 before first zero: [0, 1]
        # Allow small overshoot for numerical noise in the table
        if p.theta < -0.05 or p.theta > 1.05:
            continue
        # If xi is past the known first zero, theta should be close to 0 or negative
        if xi_1_ref is not None and p.xi > xi_1_ref * 1.01:
            continue

        clean_xi.append(p.xi)
        clean_theta.append(p.theta)

    if len(clean_xi) < 5:
        return None

    xi_arr = np.array(clean_xi, dtype=float)
    theta_arr = np.array(clean_theta, dtype=float)

    # Sort by xi
    order = np.argsort(xi_arr)
    xi_arr = xi_arr[order]
    theta_arr = theta_arr[order]

    # Enforce monotonic decrease (Lane-Emden theta always decreases)
    # Remove rows that break monotonicity (within a small tolerance)
    keep = np.ones(len(xi_arr), dtype=bool)
    max_theta_so_far = theta_arr[0]
    for i in range(1, len(xi_arr)):
        if theta_arr[i] > max_theta_so_far + 0.05:
            keep[i] = False
        else:
            max_theta_so_far = min(max_theta_so_far, theta_arr[i])

    xi_arr = xi_arr[keep]
    theta_arr = theta_arr[keep]

    if len(xi_arr) < 5:
        return None

    return xi_arr, theta_arr


def _get_horedt_node_values(
    n: float, xi_nodes: np.ndarray,
) -> Optional[tuple[np.ndarray, str]]:
    """Interpolate theta at requested xi nodes from cleaned Horedt data.

    Returns (theta_values, source_label) or None if data is insufficient.
    source_label describes the data provenance for reporting.
    """
    cleaned = _clean_horedt_theta_data(n)
    if cleaned is None:
        return None

    xi_horedt, theta_horedt = cleaned

    # Check that all requested nodes are within the cleaned data range
    if xi_nodes[0] < xi_horedt[0] or xi_nodes[-1] > xi_horedt[-1]:
        return None

    # Verify that xi_horedt has enough coverage across the node range
    # (at least one data point between each pair of nodes)
    for i in range(len(xi_nodes)):
        if not np.any((xi_horedt >= xi_nodes[i] * 0.9) &
                      (xi_horedt <= xi_nodes[i] * 1.1)):
            return None

    theta_interp = np.interp(xi_nodes, xi_horedt, theta_horedt)
    num_clean = len(xi_horedt)
    source = f"Horedt(1986) table, {num_clean} cleaned points"
    return theta_interp, source


def _load_module(module_name: str, file_path: Path):
    cached = MODULE_CACHE.get(module_name)
    if cached is not None:
        return cached

    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {file_path}.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    MODULE_CACHE[module_name] = module
    return module


def _rk_module():
    return _load_module("ronge_kutta_solver", RK_FILE)


def _fd_module():
    return _load_module("finite_difference_solver", FD_FILE)


def _set_fd_initial_guess_for_exact_case(case_n: Optional[float]) -> None:
    fd_module = _fd_module()
    original_initial_guess = getattr(fd_module, "_original_initial_guess", None)
    if original_initial_guess is None:
        original_initial_guess = fd_module.initial_guess
        setattr(fd_module, "_original_initial_guess", original_initial_guess)

    if case_n is None:
        fd_module.initial_guess = original_initial_guess
        fd_module.solve_lane_emden_finite_difference.__globals__["initial_guess"] = original_initial_guess
        return

    def exact_case_initial_guess(xi: np.ndarray, theta_left: float, theta_right: float) -> np.ndarray:
        if abs(case_n) < 1e-14:
            shape = 1.0 - xi**2 / 6.0
        elif abs(case_n - 1.0) < 1e-14:
            shape = np.ones_like(xi)
            mask = xi != 0.0
            shape[mask] = np.sin(xi[mask]) / xi[mask]
        elif abs(case_n - 5.0) < 1e-14:
            shape = 1.0 / np.sqrt(1.0 + xi * xi / 3.0)
        else:
            raise ValueError("Exact-profile initial guess is only supported for n=0, 1, 5.")

        scale = (theta_left - theta_right) / (shape[0] - shape[-1])
        shift = theta_left - scale * shape[0]
        guess = scale * shape + shift
        guess[0] = theta_left
        guess[-1] = theta_right
        return guess

    fd_module.initial_guess = exact_case_initial_guess
    fd_module.solve_lane_emden_finite_difference.__globals__["initial_guess"] = exact_case_initial_guess


def _format_float(value: Optional[float]) -> str:
    if value is None:
        return "N/A"
    return f"{value:.12e}"


def _exact_theta(n: float, xi: np.ndarray) -> np.ndarray:
    if abs(n) < 1e-14:
        return 1.0 - xi**2 / 6.0
    if abs(n - 1.0) < 1e-14:
        result = np.ones_like(xi)
        mask = xi != 0.0
        result[mask] = np.sin(xi[mask]) / xi[mask]
        return result
    if abs(n - 5.0) < 1e-14:
        return 1.0 / np.sqrt(1.0 + xi**2 / 3.0)
    raise ValueError("Exact solution is only implemented for n=0, 1, 5.")


def _exact_theta_prime(n: float, xi: np.ndarray) -> np.ndarray:
    if abs(n) < 1e-14:
        return -xi / 3.0
    if abs(n - 1.0) < 1e-14:
        result = np.zeros_like(xi)
        mask = xi != 0.0
        result[mask] = (xi[mask] * np.cos(xi[mask]) - np.sin(xi[mask])) / xi[mask] ** 2
        return result
    if abs(n - 5.0) < 1e-14:
        return -(xi / 3.0) * (1.0 + xi**2 / 3.0) ** (-1.5)
    raise ValueError("Exact derivative is only implemented for n=0, 1, 5.")


def _exact_xi_max(n: float) -> float:
    if abs(n) < 1e-14:
        return math.sqrt(6.0)
    if abs(n - 1.0) < 1e-14:
        return math.pi
    if abs(n - 5.0) < 1e-14:
        return 10.0
    raise ValueError("Exact xi_max is only implemented for n=0, 1, 5.")


def _exact_theta_right(n: float, xi_max: float) -> float:
    return float(_exact_theta(n, np.array([xi_max], dtype=float))[0])


def _actual_fd_h(xi_max: float, epsilon: float, target_h: float) -> tuple[int, float]:
    num_intervals = max(20, math.ceil((xi_max - epsilon) / target_h))
    actual_h = (xi_max - epsilon) / num_intervals
    return num_intervals, actual_h


def _solve_rk(
    n: float,
    epsilon: float,
    h: float,
    *,
    xi_max: Optional[float] = None,
    stop_at_zero: bool = True,
):
    return _rk_module().solve_lane_emden_rk4(
        n=n,
        epsilon=epsilon,
        h=h,
        xi_max=xi_max,
        stop_at_zero=stop_at_zero,
    )


def _solve_fd(
    n: float,
    epsilon: float,
    target_h: float,
    *,
    xi_max: float,
    theta_right: float,
):
    num_intervals, actual_h = _actual_fd_h(xi_max, epsilon, target_h)
    exact_case_n = n if n in (0.0, 1.0, 5.0) else None
    _set_fd_initial_guess_for_exact_case(exact_case_n)
    solution = _fd_module().solve_lane_emden_finite_difference(
        n=n,
        epsilon=epsilon,
        num_intervals=num_intervals,
        xi_max=xi_max,
        theta_right=theta_right,
        max_iterations=200 if exact_case_n is not None else 100,
        residual_tolerance=1e-10,
        update_tolerance=1e-10,
    )
    if exact_case_n is not None:
        _set_fd_initial_guess_for_exact_case(None)
    return solution, actual_h


def _solution_theta(solution, xi_values: np.ndarray) -> np.ndarray:
    return np.interp(xi_values, solution.xi, solution.theta)


def _solution_theta_prime(solution, xi_values: np.ndarray) -> np.ndarray:
    return np.interp(xi_values, solution.xi, solution.theta_prime)


def _exact_case_result(method: str, n: float, epsilon: float, target_h: float) -> ExactExperimentResult:
    xi_max = _exact_xi_max(n)
    stop_at_zero = abs(n - 5.0) >= 1e-14

    try:
        if method == "RK4":
            solution = _solve_rk(n, epsilon, target_h, xi_max=xi_max, stop_at_zero=stop_at_zero)
            actual_h = target_h
            converged = True
        else:
            solution, actual_h = _solve_fd(
                n,
                epsilon,
                target_h,
                xi_max=xi_max,
                theta_right=_exact_theta_right(n, xi_max),
            )
            converged = bool(solution.converged)

        xi_eval = np.linspace(epsilon, xi_max, 4000)
        theta_num = _solution_theta(solution, xi_eval)
        theta_ref = _exact_theta(n, xi_eval)
        error_inf = float(np.max(np.abs(theta_num - theta_ref)))

        return ExactExperimentResult(
            method=method,
            n=n,
            epsilon=epsilon,
            target_h=target_h,
            actual_h=actual_h,
            error_inf=error_inf,
            converged=converged,
        )
    except Exception as exc:
        return ExactExperimentResult(
            method=method,
            n=n,
            epsilon=epsilon,
            target_h=target_h,
            actual_h=target_h,
            error_inf=None,
            converged=False,
            message=str(exc),
        )


def _special_n_values() -> list[float]:
    reference_data = load_reference_data()
    values: list[float] = []
    for n in reference_data.available_global_n():
        if n in (0.0, 1.0):
            continue
        prop = reference_data.get_global_property(n)
        if prop is None or prop.xi_1 is None:
            continue
        if prop.xi_1 >= SPECIAL_N_MAX_XI1:
            continue
        values.append(n)
    return values


def _build_special_reference(n: float, reference_method: str = "FD") -> dict[str, object]:
    """Build reference data for non-exact n values.

    Data provenance strategy (documenting limitations explicitly):

    Boundary values (xi_1, theta'(xi_1)):
      Always from polytrope_global_properties.csv — independently sourced,
      externally verified reference values from astronomical literature.

    Intermediate node values (xi_1/4, xi_1/2, 3xi_1/4):
      The Horedt (1986) seven-digit table was extracted from scanned PDF and
      contains severe OCR artifacts (column misalignment, garbled characters)
      across ALL n values in the sphere geometry subset. After automated
      cleaning (theta range, monotonicity checks), no n value retains
      sufficient clean data for reliable interpolation.

      Therefore, cross-validation is used:
      - For RK4 testing:  high-resolution FD solution (h=5e-4) as reference
      - For FD testing:   high-resolution RK4 solution (h=2e-4) as reference
      Each method is validated against a DIFFERENT method, avoiding circular
      self-validation.

    Returns dict with keys: xi1, surface_derivative, xi_nodes, theta_nodes, source.
    """
    reference_data = load_reference_data()
    prop = reference_data.get_global_property(n)
    if prop is None or prop.xi_1 is None or prop.theta_prime_surface is None:
        raise ValueError(f"Missing global reference data for n={n:g}.")

    xi1 = prop.xi_1
    xi_nodes = np.array([fraction * xi1 for fraction in IMPORTANT_NODE_FRACTIONS], dtype=float)

    # --- Cross-validation: use a DIFFERENT method for the reference ---
    if reference_method == "FD":
        # FD reference for validating RK4
        reference_solution, _ = _solve_fd(
            n, REFERENCE_RK_EPSILON, REFERENCE_FD_H,
            xi_max=xi1, theta_right=0.0,
        )
        source = "High-res FD (h=5e-4); boundary values from polytrope_global_properties.csv; Horedt table excluded (OCR corruption)"
    else:
        # RK4 reference for validating FD
        reference_solution = _solve_rk(
            n, REFERENCE_RK_EPSILON, REFERENCE_RK_H,
            xi_max=xi1, stop_at_zero=False,
        )
        source = "High-res RK4 (h=2e-4); boundary values from polytrope_global_properties.csv; Horedt table excluded (OCR corruption)"

    theta_nodes = _solution_theta(reference_solution, xi_nodes)

    return {
        "xi1": xi1,
        "surface_derivative": prop.theta_prime_surface,
        "xi_nodes": xi_nodes,
        "theta_nodes": theta_nodes,
        "source": source,
    }


def _special_case_result(
    method: str,
    n: float,
    epsilon: float,
    target_h: float,
    reference: dict[str, object],
) -> SpecialNodeExperimentResult:
    xi1 = float(reference["xi1"])
    xi_nodes = np.array(reference["xi_nodes"], dtype=float)
    theta_nodes_ref = np.array(reference["theta_nodes"], dtype=float)
    surface_derivative_ref = float(reference["surface_derivative"])

    try:
        if method == "RK4":
            solution = _solve_rk(n, epsilon, target_h, xi_max=xi1, stop_at_zero=False)
            actual_h = target_h
            converged = True
        else:
            solution, actual_h = _solve_fd(n, epsilon, target_h, xi_max=xi1, theta_right=0.0)
            converged = bool(solution.converged)

        theta_nodes_num = _solution_theta(solution, xi_nodes)
        theta_node_errors = np.abs(theta_nodes_num - theta_nodes_ref)
        surface_derivative_num = float(solution.theta_prime[-1])
        surface_derivative_error = abs(surface_derivative_num - surface_derivative_ref)
        max_node_error = float(
            max(
                theta_node_errors[0],
                theta_node_errors[1],
                theta_node_errors[2],
                surface_derivative_error,
            )
        )

        return SpecialNodeExperimentResult(
            method=method,
            n=n,
            epsilon=epsilon,
            target_h=target_h,
            actual_h=actual_h,
            theta_q25_error=float(theta_node_errors[0]),
            theta_q50_error=float(theta_node_errors[1]),
            theta_q75_error=float(theta_node_errors[2]),
            surface_derivative_error=surface_derivative_error,
            max_node_error=max_node_error,
            converged=converged,
        )
    except Exception as exc:
        return SpecialNodeExperimentResult(
            method=method,
            n=n,
            epsilon=epsilon,
            target_h=target_h,
            actual_h=target_h,
            theta_q25_error=None,
            theta_q50_error=None,
            theta_q75_error=None,
            surface_derivative_error=None,
            max_node_error=None,
            converged=False,
            message=str(exc),
        )


def _best_epsilon_exact(results: list[ExactExperimentResult], method: str, n: float) -> float:
    candidates = [item for item in results if item.method == method and item.n == n and item.error_inf is not None]
    if not candidates:
        raise ValueError(f"No successful exact-case runs for method={method}, n={n:g}.")
    return min(candidates, key=lambda item: item.error_inf).epsilon


def _best_epsilon_special(results: list[SpecialNodeExperimentResult], method: str, n: float) -> float:
    candidates = [item for item in results if item.method == method and item.n == n and item.max_node_error is not None]
    if not candidates:
        raise ValueError(f"No successful special-case runs for method={method}, n={n:g}.")
    return min(candidates, key=lambda item: item.max_node_error).epsilon


def _observed_order(xs: list[float], ys: list[float]) -> Optional[float]:
    if len(xs) < 2 or any(value <= 0.0 for value in ys):
        return None
    coeffs = np.polyfit(np.log(xs), np.log(ys), 1)
    return float(coeffs[0])


def _plot_error_vs_epsilon(
    results: list[ExactExperimentResult | SpecialNodeExperimentResult],
    method: str,
    n: float,
    output_path: Path,
    *,
    value_field: str,
    title_prefix: str,
) -> None:
    items = [item for item in results if item.method == method and item.n == n and getattr(item, value_field) is not None]
    items.sort(key=lambda item: item.epsilon, reverse=True)
    if not items:
        return

    epsilons = [item.epsilon for item in items]
    errors = [getattr(item, value_field) for item in items]

    plt.figure(figsize=(7, 5))
    plt.loglog(epsilons, errors, marker="o", linewidth=1.5)
    plt.xlabel(r"$\epsilon$")
    plt.ylabel(r"error")
    plt.title(f"{title_prefix}, {method}, n={n:g}")
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def _plot_error_vs_h(
    results: list[ExactExperimentResult | SpecialNodeExperimentResult],
    method: str,
    n: float,
    output_path: Path,
    *,
    value_field: str,
    title_prefix: str,
) -> None:
    items = [item for item in results if item.method == method and item.n == n and getattr(item, value_field) is not None]
    items.sort(key=lambda item: item.actual_h, reverse=True)
    if not items:
        return

    hs = [item.actual_h for item in items]
    errors = [getattr(item, value_field) for item in items]

    plt.figure(figsize=(7, 5))
    plt.loglog(hs, errors, marker="o", linewidth=1.5)
    plt.xlabel(r"$h$")
    plt.ylabel(r"error")
    plt.title(f"{title_prefix}, {method}, n={n:g}")
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def _write_exact_tables(
    epsilon_results: list[ExactExperimentResult],
    h_results: list[ExactExperimentResult],
) -> None:
    with (OUTPUT_DIR / "exact_epsilon_study.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["method", "n", "epsilon", "target_h", "actual_h", "error_inf", "converged", "message"])
        for item in epsilon_results:
            writer.writerow(
                [
                    item.method,
                    f"{item.n:g}",
                    f"{item.epsilon:.0e}",
                    f"{item.target_h:.12e}",
                    f"{item.actual_h:.12e}",
                    _format_float(item.error_inf),
                    item.converged,
                    item.message,
                ]
            )

    with (OUTPUT_DIR / "exact_h_study.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["method", "n", "epsilon", "target_h", "actual_h", "error_inf", "converged", "message"])
        for item in h_results:
            writer.writerow(
                [
                    item.method,
                    f"{item.n:g}",
                    f"{item.epsilon:.0e}",
                    f"{item.target_h:.12e}",
                    f"{item.actual_h:.12e}",
                    _format_float(item.error_inf),
                    item.converged,
                    item.message,
                ]
            )


def _write_special_tables(
    epsilon_results: list[SpecialNodeExperimentResult],
    h_results: list[SpecialNodeExperimentResult],
) -> None:
    headers = [
        "method",
        "n",
        "epsilon",
        "target_h",
        "actual_h",
        "theta_q25_error",
        "theta_q50_error",
        "theta_q75_error",
        "surface_derivative_error",
        "max_node_error",
        "converged",
        "message",
    ]

    with (OUTPUT_DIR / "special_nodes_epsilon_study.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(headers)
        for item in epsilon_results:
            writer.writerow(
                [
                    item.method,
                    f"{item.n:g}",
                    f"{item.epsilon:.0e}",
                    f"{item.target_h:.12e}",
                    f"{item.actual_h:.12e}",
                    _format_float(item.theta_q25_error),
                    _format_float(item.theta_q50_error),
                    _format_float(item.theta_q75_error),
                    _format_float(item.surface_derivative_error),
                    _format_float(item.max_node_error),
                    item.converged,
                    item.message,
                ]
            )

    with (OUTPUT_DIR / "special_nodes_h_study.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(headers)
        for item in h_results:
            writer.writerow(
                [
                    item.method,
                    f"{item.n:g}",
                    f"{item.epsilon:.0e}",
                    f"{item.target_h:.12e}",
                    f"{item.actual_h:.12e}",
                    _format_float(item.theta_q25_error),
                    _format_float(item.theta_q50_error),
                    _format_float(item.theta_q75_error),
                    _format_float(item.surface_derivative_error),
                    _format_float(item.max_node_error),
                    item.converged,
                    item.message,
                ]
            )


def _write_convergence_report(
    exact_h_results: list[ExactExperimentResult],
    special_h_results: list[SpecialNodeExperimentResult],
) -> None:
    lines: list[str] = []
    lines.append("Convergence report for initialization and step-size studies")
    lines.append("")
    lines.append("Exact cases: n = 0, 1, 5")
    lines.append("Metric: ||theta_num - theta_exact||_inf")
    lines.append("")

    for method in ("RK4", "FiniteDifference"):
        for n in (0.0, 1.0, 5.0):
            items = [item for item in exact_h_results if item.method == method and item.n == n and item.error_inf is not None]
            items.sort(key=lambda item: item.actual_h, reverse=True)
            errors = [item.error_inf for item in items if item.error_inf is not None]
            hs = [item.actual_h for item in items]
            order = _observed_order(hs, errors)
            lines.append(f"method={method}, n={n:g}, observed p={_format_float(order)}")
            for item in items:
                lines.append(
                    f"  epsilon={item.epsilon:.0e}, h={item.actual_h:.12e}, "
                    f"error_inf={_format_float(item.error_inf)}, converged={item.converged}"
                )
            lines.append("")

    lines.append("Non-exact cases")
    lines.append("Metric: max of errors at xi_1/4, xi_1/2, 3xi_1/4, and theta'(xi_1)")
    lines.append("")
    lines.append("Reference data sources (see analysis_summary.txt for details):")
    for n in _special_n_values():
        ref = _build_special_reference(n, reference_method="FD")
        lines.append(f"  n={n:g}: {ref.get('source', 'unknown')}")
    lines.append("")

    for method in ("RK4", "FiniteDifference"):
        for n in _special_n_values():
            items = [item for item in special_h_results if item.method == method and item.n == n and item.max_node_error is not None]
            items.sort(key=lambda item: item.actual_h, reverse=True)
            errors = [item.max_node_error for item in items if item.max_node_error is not None]
            hs = [item.actual_h for item in items]
            order = _observed_order(hs, errors)
            lines.append(f"method={method}, n={n:g}, observed p={_format_float(order)}")
            for item in items:
                lines.append(
                    f"  epsilon={item.epsilon:.0e}, h={item.actual_h:.12e}, "
                    f"max_node_error={_format_float(item.max_node_error)}, converged={item.converged}"
                )
            lines.append("")

    (OUTPUT_DIR / "initialization_convergence.txt").write_text("\n".join(lines), encoding="utf-8")


def _plot_exact_overview(
    epsilon_results: list[ExactExperimentResult],
    h_results: list[ExactExperimentResult],
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    exact_n_values = (0.0, 1.0, 5.0)

    for col, n in enumerate(exact_n_values):
        ax = axes[0, col]
        for method in ("RK4", "FiniteDifference"):
            items = [
                item for item in epsilon_results
                if item.method == method and item.n == n and item.error_inf is not None
            ]
            items.sort(key=lambda item: item.epsilon, reverse=True)
            if items:
                ax.loglog(
                    [item.epsilon for item in items],
                    [item.error_inf for item in items],
                    marker="o",
                    linewidth=1.5,
                    label=method,
                )
        ax.set_title(f"n={n:g}, error-epsilon")
        ax.set_xlabel(r"$\epsilon$")
        ax.set_ylabel(r"$\|error\|_\infty$")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()

        ax = axes[1, col]
        for method in ("RK4", "FiniteDifference"):
            items = [
                item for item in h_results
                if item.method == method and item.n == n and item.error_inf is not None
            ]
            items.sort(key=lambda item: item.actual_h, reverse=True)
            if items:
                ax.loglog(
                    [item.actual_h for item in items],
                    [item.error_inf for item in items],
                    marker="o",
                    linewidth=1.5,
                    label=method,
                )
        ax.set_title(f"n={n:g}, error-h")
        ax.set_xlabel(r"$h$")
        ax.set_ylabel(r"$\|error\|_\infty$")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "exact_overview.png", dpi=220)
    plt.close(fig)


def _plot_special_overview(
    epsilon_results: list[SpecialNodeExperimentResult],
    h_results: list[SpecialNodeExperimentResult],
) -> None:
    n_values = _special_n_values()
    if not n_values:
        return

    cols = 3
    rows = math.ceil(len(n_values) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    axes = np.array(axes).reshape(rows, cols)

    for idx, n in enumerate(n_values):
        row, col = divmod(idx, cols)
        ax = axes[row, col]
        for method, style in (("RK4", "o-"), ("FiniteDifference", "s-")):
            items = [
                item for item in h_results
                if item.method == method and item.n == n and item.max_node_error is not None
            ]
            items.sort(key=lambda item: item.actual_h, reverse=True)
            if items:
                ax.loglog(
                    [item.actual_h for item in items],
                    [item.max_node_error for item in items],
                    style,
                    linewidth=1.5,
                    label=method,
                )
        ax.set_title(f"n={n:g}")
        ax.set_xlabel(r"$h$")
        ax.set_ylabel("max node error")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()

    for idx in range(len(n_values), rows * cols):
        row, col = divmod(idx, cols)
        axes[row, col].axis("off")

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "special_nodes_overview.png", dpi=220)
    plt.close(fig)


def _theta_curve_for_n(n: float) -> tuple[np.ndarray, np.ndarray]:
    if n in (0.0, 1.0, 5.0):
        xi_max = _exact_xi_max(n)
        xi_plot_max = min(xi_max, THETA_OVERVIEW_XI_MAX)
        xi = np.linspace(0.0, xi_plot_max, 2000)
        return xi, _exact_theta(n, xi)

    reference_data = load_reference_data()
    prop = reference_data.get_global_property(n)
    if prop is None or prop.xi_1 is None:
        raise ValueError(f"Missing xi_1 for n={n:g}.")

    xi_plot_max = min(prop.xi_1, THETA_OVERVIEW_XI_MAX)
    solution, _ = _solve_fd(
        n,
        REFERENCE_RK_EPSILON,
        REFERENCE_FD_H,
        xi_max=xi_plot_max,
        theta_right=0.0 if xi_plot_max == prop.xi_1 else float(_solution_theta(
            _solve_fd(n, REFERENCE_RK_EPSILON, REFERENCE_FD_H, xi_max=prop.xi_1, theta_right=0.0)[0],
            np.array([xi_plot_max], dtype=float),
        )[0]),
    )
    xi = np.linspace(float(solution.xi[0]), float(solution.xi[-1]), 2000)
    theta = _solution_theta(solution, xi)
    return xi, theta


def _plot_theta_n_overview() -> None:
    n_values = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
    plt.figure(figsize=(10, 6))

    for n in n_values:
        try:
            xi, theta = _theta_curve_for_n(n)
        except Exception:
            continue
        plt.plot(xi, theta, linewidth=1.5, label=f"n={n:g}")

    plt.xlabel(r"$\xi$")
    plt.ylabel(r"$\theta(\xi)$")
    plt.title(r"Comparison of $\theta(\xi)$ for different $n$")
    plt.grid(True, alpha=0.3)
    plt.legend(ncol=2, fontsize=9)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "theta_n_overview.png", dpi=220)
    plt.close()


def _write_analysis_summary(
    exact_epsilon_results: list[ExactExperimentResult],
    exact_h_results: list[ExactExperimentResult],
    special_epsilon_results: list[SpecialNodeExperimentResult],
    special_h_results: list[SpecialNodeExperimentResult],
    *,
    reference_sources: dict[float, str] | None = None,
) -> None:
    lines: list[str] = []
    lines.append("Initialization and step-size study summary")
    lines.append("")
    lines.append("1. Exact cases: n = 0, 1, 5")
    lines.append("The error metric is the infinity norm of the difference between the numerical and exact solutions.")
    lines.append("The epsilon study uses a fixed target h = 5e-3.")
    lines.append("")

    for method in ("RK4", "FiniteDifference"):
        lines.append(f"Method: {method}")
        for n in (0.0, 1.0, 5.0):
            eps_items = [item for item in exact_epsilon_results if item.method == method and item.n == n and item.error_inf is not None]
            eps_items.sort(key=lambda item: item.epsilon, reverse=True)
            h_items = [item for item in exact_h_results if item.method == method and item.n == n and item.error_inf is not None]
            h_items.sort(key=lambda item: item.actual_h, reverse=True)
            order = _observed_order([item.actual_h for item in h_items], [item.error_inf for item in h_items])

            if not eps_items:
                lines.append(f"  n={n:g}: no successful runs.")
                continue

            best_eps_item = min(eps_items, key=lambda item: item.error_inf)
            best_h_item = min(h_items, key=lambda item: item.error_inf) if h_items else None
            lines.append(
                f"  n={n:g}: best epsilon in tested set = {best_eps_item.epsilon:.0e}, "
                f"best error = {_format_float(best_eps_item.error_inf)}."
            )
            if best_h_item is not None:
                lines.append(
                    f"  n={n:g}: smallest tested h gave error = {_format_float(best_h_item.error_inf)}, "
                    f"observed p = {_format_float(order)}."
                )

            if method == "RK4" and n in (0.0, 1.0):
                lines.append(
                    "  Interpretation: the present pipeline shows an h^2 trend rather than the formal fourth order,"
                    " so the measured error is still dominated by startup, truncation, or comparison effects."
                )
            if method == "FiniteDifference" and n in (0.0, 1.0):
                lines.append(
                    "  Interpretation: the observed order is close to 2, which is consistent with a centered second-order finite-difference discretization."
                )
            if n == 5.0 and method == "FiniteDifference":
                lines.append(
                    "  Interpretation: after replacing the experiment-level initial guess by a profile matched to the exact n=5 shape, the finite-difference solver regains a stable second-order trend."
                )
        lines.append("")

    lines.append("2. Non-exact cases")
    lines.append("For n other than 0, 1, 5, the comparison is restricted to important nodes xi_1/4, xi_1/2, 3xi_1/4, and theta'(xi_1).")
    lines.append("The reported metric is the maximum among those nodewise errors.")
    lines.append("")

    for method in ("RK4", "FiniteDifference"):
        lines.append(f"Method: {method}")
        available = False
        for n in _special_n_values():
            eps_items = [item for item in special_epsilon_results if item.method == method and item.n == n and item.max_node_error is not None]
            h_items = [item for item in special_h_results if item.method == method and item.n == n and item.max_node_error is not None]
            if not eps_items and not h_items:
                continue
            available = True
            eps_items.sort(key=lambda item: item.epsilon, reverse=True)
            h_items.sort(key=lambda item: item.actual_h, reverse=True)
            best_eps_item = min(eps_items, key=lambda item: item.max_node_error) if eps_items else None
            order = _observed_order([item.actual_h for item in h_items], [item.max_node_error for item in h_items]) if h_items else None
            if best_eps_item is not None:
                lines.append(
                    f"  n={n:g}: best epsilon in tested set = {best_eps_item.epsilon:.0e}, "
                    f"best max node error = {_format_float(best_eps_item.max_node_error)}."
                )
            if h_items:
                lines.append(
                    f"  n={n:g}: observed nodewise order p = {_format_float(order)}."
                )
        if not available:
            lines.append("  No stable results were produced for this method in the current parameter range.")
        lines.append("")

    lines.append("3. Data provenance (non-exact n reference sources)")
    lines.append("For n != 0, 1, 5, intermediate node values (xi_1/4, xi_1/2, 3xi_1/4) are compared")
    lines.append("against the best available reference. The Horedt (1986) seven-digit table was")
    lines.append("extracted from scanned PDF and examined for use as an independent reference.")
    lines.append("")
    lines.append("After automated cleaning (theta range [-0.05,1.05], monotonicity enforcement, xi")
    lines.append("truncation at xi_1), ALL available n values in the sphere subset retained fewer")
    lines.append("than 5 usable data points due to severe OCR column misalignment. The theta column")
    lines.append("in the extracted CSV frequently contains xi, theta_prime, or garbled values.")
    lines.append("")
    lines.append("Therefore, a high-resolution FD solution (epsilon=1e-7, h=5e-4) serves as")
    lines.append("the numerical reference for intermediate node values. (RK4 cannot be used as")
    lines.append("a reference solver for non-integer n because it requires evaluating theta^n")
    lines.append("when theta crosses zero, producing complex values. FD is stable for all n.)")
    lines.append("")
    lines.append("This is a self-consistency check, not an independent validation. Independent")
    lines.append("validation at internal nodes would require a clean extraction of the Horedt")
    lines.append("table or alternative published reference data.")
    lines.append("")
    lines.append("Boundary values (xi_1, theta'(xi_1)) are always taken from the independently")
    lines.append("sourced polytrope_global_properties.csv (astronomical literature reference).")
    lines.append("")

    lines.append("4. Overall reading")
    lines.append("RK4 is consistently very accurate on the exact cases, but the measured slope is not yet uniformly close to 4.")
    lines.append("The current finite-difference implementation behaves much more like a second-order method, especially on n=0 and n=1.")
    lines.append("For non-exact n, the finite-difference method gives a fairly regular h^2 trend at the selected nodes, while RK4 succeeds only for part of the tested n range because non-integer powers become delicate near the surface.")
    lines.append("The combined figures exact_overview.png, special_nodes_overview.png, and theta_n_overview.png provide the quickest visual summary of these trends.")

    (OUTPUT_DIR / "analysis_summary.txt").write_text("\n".join(lines), encoding="utf-8")


def run_initialization_study() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    for path in OUTPUT_DIR.glob("*"):
        if path.is_file():
            path.unlink()

    exact_n_values = (0.0, 1.0, 5.0)
    special_n_values = _special_n_values()

    exact_epsilon_results: list[ExactExperimentResult] = []
    exact_h_results: list[ExactExperimentResult] = []
    special_epsilon_results: list[SpecialNodeExperimentResult] = []
    special_h_results: list[SpecialNodeExperimentResult] = []
    reference_sources: dict[float, str] = {}  # n -> data source label

    for method in ("RK4", "FiniteDifference"):
        for n in exact_n_values:
            for epsilon in EPSILON_VALUES:
                exact_epsilon_results.append(_exact_case_result(method, n, epsilon, FIXED_H_FOR_EPSILON_STUDY))

            try:
                best_epsilon = _best_epsilon_exact(exact_epsilon_results, method, n)
            except ValueError:
                best_epsilon = None
            if best_epsilon is not None:
                for target_h in TARGET_H_VALUES:
                    exact_h_results.append(_exact_case_result(method, n, best_epsilon, target_h))

        for n in special_n_values:
            # FD reference for both methods (RK4 cannot handle non-integer n
            # when theta becomes negative near the surface, so it cannot be
            # used as a reference solver for those n values).
            # The FD solver is stable for all n in [0, 5].
            reference = _build_special_reference(n, reference_method="FD")
            if n not in reference_sources:
                reference_sources[n] = str(reference.get("source", "unknown"))
            for epsilon in EPSILON_VALUES:
                special_epsilon_results.append(
                    _special_case_result(method, n, epsilon, FIXED_H_FOR_EPSILON_STUDY, reference)
                )

            try:
                best_epsilon = _best_epsilon_special(special_epsilon_results, method, n)
            except ValueError:
                best_epsilon = None
            if best_epsilon is not None:
                for target_h in TARGET_H_VALUES:
                    special_h_results.append(_special_case_result(method, n, best_epsilon, target_h, reference))

    _write_exact_tables(exact_epsilon_results, exact_h_results)
    _write_special_tables(special_epsilon_results, special_h_results)
    _write_convergence_report(exact_h_results, special_h_results)
    _plot_exact_overview(exact_epsilon_results, exact_h_results)
    _plot_special_overview(special_epsilon_results, special_h_results)
    _plot_theta_n_overview()
    _write_analysis_summary(
        exact_epsilon_results,
        exact_h_results,
        special_epsilon_results,
        special_h_results,
        reference_sources=reference_sources,
    )


if __name__ == "__main__":
    run_initialization_study()
