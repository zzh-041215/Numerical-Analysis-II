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

from data_input import load_reference_data


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output_initial"
RK_FILE = BASE_DIR / "Ronge-Kutta.py"
FD_FILE = BASE_DIR / "finite-difference.py"

MODULE_CACHE: dict[str, object] = {}

EPSILON_VALUES = (1e-3, 1e-4, 1e-5)
TARGET_H_VALUES = (2e-2, 1e-2, 5e-3, 2.5e-3)
FIXED_H_FOR_EPSILON_STUDY = 5e-3
REFERENCE_RK_EPSILON = 1e-7
REFERENCE_RK_H = 2e-4
REFERENCE_FD_H = 5e-4
IMPORTANT_NODE_FRACTIONS = (0.25, 0.5, 0.75)
SPECIAL_N_MAX_XI1 = 50.0


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
    solution = _fd_module().solve_lane_emden_finite_difference(
        n=n,
        epsilon=epsilon,
        num_intervals=num_intervals,
        xi_max=xi_max,
        theta_right=theta_right,
        max_iterations=100,
        residual_tolerance=1e-10,
        update_tolerance=1e-10,
    )
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


def _build_special_reference(n: float) -> dict[str, object]:
    reference_data = load_reference_data()
    prop = reference_data.get_global_property(n)
    if prop is None or prop.xi_1 is None or prop.theta_prime_surface is None:
        raise ValueError(f"Missing global reference data for n={n:g}.")

    xi1 = prop.xi_1
    xi_nodes = np.array([fraction * xi1 for fraction in IMPORTANT_NODE_FRACTIONS], dtype=float)
    reference_solution, _ = _solve_fd(
        n,
        REFERENCE_RK_EPSILON,
        REFERENCE_FD_H,
        xi_max=xi1,
        theta_right=0.0,
    )
    theta_nodes = _solution_theta(reference_solution, xi_nodes)

    return {
        "xi1": xi1,
        "surface_derivative": prop.theta_prime_surface,
        "xi_nodes": xi_nodes,
        "theta_nodes": theta_nodes,
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

            _plot_error_vs_epsilon(
                exact_epsilon_results,
                method,
                n,
                OUTPUT_DIR / f"exact_{method.lower()}_n_{n:g}_error_vs_epsilon.png",
                value_field="error_inf",
                title_prefix="Error vs epsilon",
            )
            _plot_error_vs_h(
                exact_h_results,
                method,
                n,
                OUTPUT_DIR / f"exact_{method.lower()}_n_{n:g}_error_vs_h.png",
                value_field="error_inf",
                title_prefix="Error vs h",
            )

        for n in special_n_values:
            reference = _build_special_reference(n)
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

            _plot_error_vs_epsilon(
                special_epsilon_results,
                method,
                n,
                OUTPUT_DIR / f"special_{method.lower()}_n_{n:g}_error_vs_epsilon.png",
                value_field="max_node_error",
                title_prefix="Max node error vs epsilon",
            )
            _plot_error_vs_h(
                special_h_results,
                method,
                n,
                OUTPUT_DIR / f"special_{method.lower()}_n_{n:g}_error_vs_h.png",
                value_field="max_node_error",
                title_prefix="Max node error vs h",
            )

    _write_exact_tables(exact_epsilon_results, exact_h_results)
    _write_special_tables(special_epsilon_results, special_h_results)
    _write_convergence_report(exact_h_results, special_h_results)


if __name__ == "__main__":
    run_initialization_study()
