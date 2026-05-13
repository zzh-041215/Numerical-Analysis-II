from __future__ import annotations

import csv
import importlib.util
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data_input import load_reference_data


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output_initial"
RK_FILE = BASE_DIR / "Ronge-Kutta.py"
FD_FILE = BASE_DIR / "finite-difference.py"


@dataclass(frozen=True)
class InitializationStabilityResult:
    method: str
    n: float
    epsilon: float
    converged: bool
    max_error_inf: Optional[float]
    iterations: Optional[int]
    residual_norm: Optional[float]
    update_norm: Optional[float]
    message: str = ""


def _load_module(module_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {file_path}.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _format_float(value: Optional[float]) -> str:
    if value is None:
        return ""
    return f"{value:.12e}"


def _interpolate_solution_theta(solution, xi_values: np.ndarray) -> np.ndarray:
    return np.interp(xi_values, solution.xi, solution.theta)


def _exact_theta(n: float, xi: np.ndarray) -> Optional[np.ndarray]:
    if abs(n) < 1e-14:
        return 1.0 - xi**2 / 6.0
    if abs(n - 1.0) < 1e-14:
        result = np.ones_like(xi)
        mask = xi != 0.0
        result[mask] = np.sin(xi[mask]) / xi[mask]
        return result
    return None


def _reference_points(n: float) -> tuple[np.ndarray, np.ndarray, str]:
    reference_data = load_reference_data()
    points = reference_data.get_sphere_table(n, until_first_zero=True)
    xi = np.array([point.xi for point in points], dtype=float)
    theta_table = np.array([point.theta for point in points], dtype=float)
    theta_exact = _exact_theta(n, xi)
    if theta_exact is not None:
        return xi, theta_exact, "Exact"
    return xi, theta_table, "ReferenceTable"


def _reference_curve(n: float) -> tuple[np.ndarray, np.ndarray, str]:
    xi_sparse, theta_sparse, label = _reference_points(n)
    if xi_sparse.size == 0:
        return xi_sparse, theta_sparse, label

    if label == "Exact":
        xi_dense = np.linspace(0.0, float(xi_sparse[-1]), 2000)
        return xi_dense, _exact_theta(n, xi_dense), label

    if xi_sparse.size == 1:
        return xi_sparse, theta_sparse, label

    xi_dense = np.linspace(float(xi_sparse[0]), float(xi_sparse[-1]), 2000)
    theta_dense = np.interp(xi_dense, xi_sparse, theta_sparse)
    return xi_dense, theta_dense, label


def _compute_max_error(solution, n: float) -> Optional[float]:
    xi_ref, theta_ref, _ = _reference_points(n)
    if xi_ref.size == 0:
        return None
    mask = (xi_ref >= solution.xi[0]) & (xi_ref <= solution.xi[-1])
    if not np.any(mask):
        return None
    theta_num = _interpolate_solution_theta(solution, xi_ref[mask])
    return float(np.max(np.abs(theta_num - theta_ref[mask])))


def test_rk_epsilon_stability(
    n: float,
    epsilon: float,
    *,
    h: float = 2e-4,
    max_steps: int = 1_000_000,
):
    try:
        rk_module = _load_module("ronge_kutta_solver", RK_FILE)
        solution = rk_module.solve_lane_emden_rk4(
            n=n,
            epsilon=epsilon,
            h=h,
            max_steps=max_steps,
        )
        max_error_inf = _compute_max_error(solution, n)
        result = InitializationStabilityResult(
            method="RK4",
            n=n,
            epsilon=epsilon,
            converged=solution.first_zero is not None and (max_error_inf is None or math.isfinite(max_error_inf)),
            max_error_inf=max_error_inf,
            iterations=None,
            residual_norm=None,
            update_norm=None,
        )
        return result, solution
    except Exception as exc:
        return (
            InitializationStabilityResult(
                method="RK4",
                n=n,
                epsilon=epsilon,
                converged=False,
                max_error_inf=None,
                iterations=None,
                residual_norm=None,
                update_norm=None,
                message=str(exc),
            ),
            None,
        )


def test_fd_epsilon_stability(
    n: float,
    epsilon: float,
    *,
    num_intervals: int = 2000,
    max_iterations: int = 80,
    residual_tolerance: float = 1e-10,
    update_tolerance: float = 1e-10,
):
    try:
        fd_module = _load_module("finite_difference_solver", FD_FILE)
        solution = fd_module.solve_lane_emden_finite_difference(
            n=n,
            epsilon=epsilon,
            num_intervals=num_intervals,
            max_iterations=max_iterations,
            residual_tolerance=residual_tolerance,
            update_tolerance=update_tolerance,
        )
        max_error_inf = _compute_max_error(solution, n)
        result = InitializationStabilityResult(
            method="FiniteDifference",
            n=n,
            epsilon=epsilon,
            converged=bool(solution.converged),
            max_error_inf=max_error_inf,
            iterations=solution.iterations,
            residual_norm=solution.residual_norm,
            update_norm=solution.update_norm,
        )
        return result, solution
    except Exception as exc:
        return (
            InitializationStabilityResult(
                method="FiniteDifference",
                n=n,
                epsilon=epsilon,
                converged=False,
                max_error_inf=None,
                iterations=None,
                residual_norm=None,
                update_norm=None,
                message=str(exc),
            ),
            None,
        )


def results_to_rows(results: Iterable[InitializationStabilityResult]) -> list[dict[str, object]]:
    return [
        {
            "method": result.method,
            "n": result.n,
            "epsilon": result.epsilon,
            "converged": result.converged,
            "max_error_inf": result.max_error_inf,
            "iterations": result.iterations,
            "residual_norm": result.residual_norm,
            "update_norm": result.update_norm,
            "message": result.message,
        }
        for result in results
    ]


def _write_error_table(results: Iterable[InitializationStabilityResult], output_path: Path) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "method",
                "n",
                "epsilon",
                "converged",
                "max_error_inf",
                "iterations",
                "residual_norm",
                "update_norm",
                "message",
            ],
        )
        writer.writeheader()
        for row in results_to_rows(results):
            writer.writerow(row)


def _write_summary_table(results: Iterable[InitializationStabilityResult], output_path: Path) -> None:
    grouped = sorted(results, key=lambda item: (item.n, item.method, item.epsilon))
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["n", "method", "epsilon", "converged", "max_error_inf"])
        for item in grouped:
            writer.writerow(
                [
                    f"{item.n:g}",
                    item.method,
                    f"{item.epsilon:.0e}",
                    item.converged,
                    _format_float(item.max_error_inf),
                ]
            )


def _compute_observed_orders(items: list[InitializationStabilityResult]) -> list[tuple[float, float, Optional[float]]]:
    ordered = sorted(items, key=lambda item: item.epsilon, reverse=True)
    output: list[tuple[float, float, Optional[float]]] = []
    for left, right in zip(ordered, ordered[1:]):
        if (
            left.max_error_inf is None
            or right.max_error_inf is None
            or left.max_error_inf <= 0.0
            or right.max_error_inf <= 0.0
        ):
            order = None
        else:
            order = math.log(left.max_error_inf / right.max_error_inf) / math.log(left.epsilon / right.epsilon)
        output.append((left.epsilon, right.epsilon, order))
    return output


def _write_convergence_report(results: list[InitializationStabilityResult], output_path: Path) -> None:
    grouped: dict[tuple[float, str], list[InitializationStabilityResult]] = {}
    for result in results:
        grouped.setdefault((result.n, result.method), []).append(result)

    lines: list[str] = []
    lines.append("Observed convergence order with respect to epsilon")
    lines.append("Error metric: infinity norm of theta error on reference coordinates")
    lines.append("")

    for (n, method) in sorted(grouped):
        lines.append(f"n = {n:g}, method = {method}")
        for item in sorted(grouped[(n, method)], key=lambda value: value.epsilon, reverse=True):
            lines.append(
                f"  epsilon = {item.epsilon:.0e}, converged = {item.converged}, "
                f"||error||_inf = {_format_float(item.max_error_inf) or 'N/A'}"
            )
        for eps_left, eps_right, order in _compute_observed_orders(grouped[(n, method)]):
            order_text = "N/A" if order is None else f"{order:.6f}"
            lines.append(f"  order from {eps_left:.0e} to {eps_right:.0e}: {order_text}")
        lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")


def _plot_method_comparison(
    n: float,
    epsilon_to_solution: dict[float, object],
    method_name: str,
    output_path: Path,
) -> None:
    xi_ref, theta_ref, label = _reference_curve(n)
    plt.figure(figsize=(8, 5))
    plt.plot(xi_ref, theta_ref, color="black", linewidth=2.0, label=label)

    for epsilon in sorted(epsilon_to_solution, reverse=True):
        solution = epsilon_to_solution[epsilon]
        xi_plot = np.linspace(float(solution.xi[0]), float(solution.xi[-1]), 2000)
        theta_plot = _interpolate_solution_theta(solution, xi_plot)
        plt.plot(xi_plot, theta_plot, linewidth=1.5, label=fr"$\epsilon={epsilon:.0e}$")

    plt.xlabel(r"$\xi$")
    plt.ylabel(r"$\theta(\xi)$")
    plt.title(f"{method_name} solution comparison, n={n:g}")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def _plot_error_curves(
    n: float,
    epsilon_to_solution: dict[float, object],
    method_name: str,
    output_path: Path,
) -> None:
    xi_ref, theta_ref, label = _reference_curve(n)
    plt.figure(figsize=(8, 5))

    for epsilon in sorted(epsilon_to_solution, reverse=True):
        solution = epsilon_to_solution[epsilon]
        mask = (xi_ref >= solution.xi[0]) & (xi_ref <= solution.xi[-1])
        xi_eval = xi_ref[mask]
        theta_num = _interpolate_solution_theta(solution, xi_eval)
        error = np.abs(theta_num - theta_ref[mask])
        plt.plot(xi_eval, error, linewidth=1.5, label=fr"$\epsilon={epsilon:.0e}$")

    plt.xlabel(r"$\xi$")
    plt.ylabel(r"$\|error\|_\infty$ pointwise view")
    plt.title(f"{method_name} absolute error, n={n:g} vs {label.lower()}")
    plt.yscale("log")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def _write_pointwise_reference_comparison(
    n: float,
    rk_solutions: dict[float, object],
    fd_solutions: dict[float, object],
    output_path: Path,
) -> None:
    xi_ref, theta_ref, label = _reference_points(n)
    if label == "Exact" or xi_ref.size == 0:
        return

    epsilons = sorted(set(rk_solutions) | set(fd_solutions), reverse=True)
    fieldnames = ["xi", "theta_reference"]
    for epsilon in epsilons:
        fieldnames.extend(
            [
                f"rk4_theta_eps_{epsilon:.0e}",
                f"rk4_abs_error_eps_{epsilon:.0e}",
                f"fd_theta_eps_{epsilon:.0e}",
                f"fd_abs_error_eps_{epsilon:.0e}",
            ]
        )

    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for index, xi in enumerate(xi_ref):
            row: dict[str, object] = {
                "xi": f"{xi:.12e}",
                "theta_reference": f"{theta_ref[index]:.12e}",
            }
            for epsilon in epsilons:
                if epsilon in rk_solutions:
                    rk_theta = float(_interpolate_solution_theta(rk_solutions[epsilon], np.array([xi]))[0])
                    row[f"rk4_theta_eps_{epsilon:.0e}"] = f"{rk_theta:.12e}"
                    row[f"rk4_abs_error_eps_{epsilon:.0e}"] = f"{abs(rk_theta - theta_ref[index]):.12e}"
                else:
                    row[f"rk4_theta_eps_{epsilon:.0e}"] = ""
                    row[f"rk4_abs_error_eps_{epsilon:.0e}"] = ""

                if epsilon in fd_solutions:
                    fd_theta = float(_interpolate_solution_theta(fd_solutions[epsilon], np.array([xi]))[0])
                    row[f"fd_theta_eps_{epsilon:.0e}"] = f"{fd_theta:.12e}"
                    row[f"fd_abs_error_eps_{epsilon:.0e}"] = f"{abs(fd_theta - theta_ref[index]):.12e}"
                else:
                    row[f"fd_theta_eps_{epsilon:.0e}"] = ""
                    row[f"fd_abs_error_eps_{epsilon:.0e}"] = ""

            writer.writerow(row)


def run_initialization_study(
    n_values: Optional[Iterable[float]] = None,
    epsilons: Iterable[float] = (1e-3, 1e-4, 1e-5),
    *,
    rk_step: float = 2e-4,
    fd_num_intervals: int = 2000,
    fd_max_iterations: int = 80,
) -> list[InitializationStabilityResult]:
    if n_values is None:
        n_values = load_reference_data().available_sphere_n()

    OUTPUT_DIR.mkdir(exist_ok=True)
    for path in OUTPUT_DIR.glob("*"):
        if path.is_file():
            path.unlink()

    results: list[InitializationStabilityResult] = []

    for n in n_values:
        rk_solutions: dict[float, object] = {}
        fd_solutions: dict[float, object] = {}

        for epsilon in epsilons:
            rk_result, rk_solution = test_rk_epsilon_stability(n, epsilon, h=rk_step)
            fd_result, fd_solution = test_fd_epsilon_stability(
                n,
                epsilon,
                num_intervals=fd_num_intervals,
                max_iterations=fd_max_iterations,
            )

            results.extend([rk_result, fd_result])

            if rk_solution is not None and rk_result.converged:
                rk_solutions[epsilon] = rk_solution
            if fd_solution is not None and fd_result.converged:
                fd_solutions[epsilon] = fd_solution

        if rk_solutions:
            _plot_method_comparison(n, rk_solutions, "RK4", OUTPUT_DIR / f"rk4_n_{n:g}_solutions.png")
            _plot_error_curves(n, rk_solutions, "RK4", OUTPUT_DIR / f"rk4_n_{n:g}_errors.png")

        if fd_solutions:
            _plot_method_comparison(
                n,
                fd_solutions,
                "FiniteDifference",
                OUTPUT_DIR / f"fd_n_{n:g}_solutions.png",
            )
            _plot_error_curves(
                n,
                fd_solutions,
                "FiniteDifference",
                OUTPUT_DIR / f"fd_n_{n:g}_errors.png",
            )

        _write_pointwise_reference_comparison(
            n,
            rk_solutions,
            fd_solutions,
            OUTPUT_DIR / f"reference_pointwise_n_{n:g}.csv",
        )

    _write_error_table(results, OUTPUT_DIR / "initialization_error_table.csv")
    _write_summary_table(results, OUTPUT_DIR / "initialization_error_summary.csv")
    _write_convergence_report(results, OUTPUT_DIR / "initialization_convergence.txt")
    return results


if __name__ == "__main__":
    run_initialization_study()
