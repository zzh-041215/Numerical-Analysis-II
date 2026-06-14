#!/usr/bin/env python
"""Unified experiment harness for the Lane-Emden numerical study project.

Runs all major experiments and generates a comprehensive output summary.
Usage: python run_all_experiments.py [--quick]

Options:
    --quick    Run reduced parameter sets for faster execution (default: full).
"""

from __future__ import annotations

import importlib.util
import math
import sys
import time
from pathlib import Path

import numpy as np

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)


# ============================================================================
# Utility: module loading
# ============================================================================

def _load_module(name: str, rel_path: str):
    """Load a Python module from a path relative to BASE_DIR."""
    file_path = BASE_DIR / rel_path
    spec = importlib.util.spec_from_file_location(name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {file_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# ============================================================================
# Experiment definitions
# ============================================================================

def experiment_1_convergence_basic(quick: bool = False) -> str:
    """Basic convergence study using exact solutions (n=0, 1, 5)."""
    from data_input import load_reference_data

    rk_mod = _load_module("rk", "Ronge-Kutta.py")
    fd_mod = _load_module("fd", "finite-difference.py")

    ref = load_reference_data()

    h_values = [2e-2, 1e-2, 7.5e-3, 5e-3, 3.75e-3, 2.5e-3, 1.25e-3]
    if quick:
        h_values = [1e-2, 5e-3, 2.5e-3]

    exact_cases = {
        0.0: (math.sqrt(6.0),
              lambda xi: 1.0 - xi**2 / 6.0),
        1.0: (math.pi,
              lambda xi: np.where(xi != 0, np.sin(xi) / xi, 1.0)),
        5.0: (10.0,
              lambda xi: 1.0 / np.sqrt(1.0 + xi**2 / 3.0)),
    }

    lines = ["=" * 70,
             "Experiment 1: Basic Convergence Study (Exact Cases)",
             "=" * 70, ""]

    for n_val, (xi_max, exact_func) in exact_cases.items():
        lines.append(f"--- n = {n_val:g} ---")
        lines.append(f"{'Method':>20s}  {'h':>10s}  {'Error_inf':>14s}  {'Order':>8s}")
        lines.append("-" * 60)

        prev_err_rk = None
        prev_h_rk = None
        prev_err_fd = None
        prev_h_fd = None

        for h_val in h_values:
            # RK4
            try:
                sol_rk = rk_mod.solve_lane_emden_rk4(
                    n=n_val, epsilon=1e-4, h=h_val, xi_max=xi_max,
                    stop_at_zero=(n_val != 5.0),
                )
                xi_eval = np.linspace(1e-4, min(xi_max, sol_rk.xi[-1]), 2000)
                err_rk = np.max(np.abs(
                    np.interp(xi_eval, sol_rk.xi, sol_rk.theta) - exact_func(xi_eval)
                ))
                order_str = ""
                if prev_err_rk is not None and prev_h_rk is not None and err_rk > 0:
                    p_obs = np.log(prev_err_rk / err_rk) / np.log(prev_h_rk / h_val)
                    order_str = f"{p_obs:.3f}"
                lines.append(f"{'RK4':>20s}  {h_val:10.2e}  {err_rk:14.6e}  {order_str:>8s}")
                prev_err_rk, prev_h_rk = err_rk, h_val
            except Exception as e:
                lines.append(f"{'RK4':>20s}  {h_val:10.2e}  {'FAILED':>14s}  {str(e)[:30]}")

            # Finite Difference
            try:
                n_int = max(50, int(np.ceil((xi_max - 1e-4) / h_val)))
                sol_fd = fd_mod.solve_lane_emden_finite_difference(
                    n=n_val, epsilon=1e-4, num_intervals=n_int,
                    xi_max=xi_max, theta_right=exact_func(np.array([xi_max]))[0]
                    if n_val != 5.0 else 0.0,
                )
                xi_eval_fd = np.linspace(1e-4, xi_max, 2000)
                err_fd = np.max(np.abs(
                    np.interp(xi_eval_fd, sol_fd.xi, sol_fd.theta) - exact_func(xi_eval_fd)
                ))
                order_str_fd = ""
                if prev_err_fd is not None and prev_h_fd is not None and err_fd > 0:
                    p_obs_fd = np.log(prev_err_fd / err_fd) / np.log(prev_h_fd / h_val)
                    order_str_fd = f"{p_obs_fd:.3f}"
                lines.append(f"{'FiniteDiff':>20s}  {h_val:10.2e}  {err_fd:14.6e}  {order_str_fd:>8s}")
                prev_err_fd, prev_h_fd = err_fd, h_val
            except Exception as e:
                lines.append(f"{'FiniteDiff':>20s}  {h_val:10.2e}  {'FAILED':>14s}  {str(e)[:30]}")

        lines.append("")

    return "\n".join(lines)


def experiment_2_physical_quantities(quick: bool = False) -> str:
    """Compute physical quantities across n values."""
    from data_input import load_reference_data

    pq_mod = _load_module("pq", "physical_quantities.py")

    ref = load_reference_data()
    n_values = ref.available_global_n()
    if quick:
        n_values = n_values[::2]  # every other n

    lines = ["=" * 70,
             "Experiment 2: Physical Quantities",
             "=" * 70, ""]
    lines.append(pq_mod.generate_physical_table(n_values=n_values))

    return "\n".join(lines)


def experiment_3_geometry_comparison(quick: bool = False) -> str:
    """Compare Slab, Cylinder, Sphere geometries."""
    geom_mod = _load_module("geom", "generalized_geometry.py")

    n_values = [0.5, 1.5, 3.0]
    if quick:
        n_values = [1.5]

    lines = ["=" * 70,
             "Experiment 3: Geometry Comparison (Slab/Cylinder/Sphere)",
             "=" * 70, ""]
    lines.append(f"{'n':>6s}  {'Slab xi_1':>12s}  {'Cyl xi_1':>12s}  {'Sph xi_1':>12s}")
    lines.append("-" * 52)

    for n_val in n_values:
        try:
            sols = geom_mod.compare_geometries(n_val, epsilon=1e-4, h=2e-3)
            slab_xi = f"{sols[0].first_zero:.6f}" if 0 in sols and sols[0].first_zero else "N/A"
            cyl_xi = f"{sols[1].first_zero:.6f}" if 1 in sols and sols[1].first_zero else "N/A"
            sph_xi = f"{sols[2].first_zero:.6f}" if 2 in sols and sols[2].first_zero else "N/A"
            lines.append(f"{n_val:6.2f}  {slab_xi:>12s}  {cyl_xi:>12s}  {sph_xi:>12s}")
        except Exception as e:
            lines.append(f"{n_val:6.2f}  FAILED: {str(e)[:40]}")

    return "\n".join(lines)


def experiment_4_manufactured_convergence(quick: bool = False) -> str:
    """MMS convergence verification."""
    mms_mod = _load_module("mms", "manufactured.py")

    n_values = [0.0, 1.5, 3.0]
    h_vals = [0.2, 0.1, 0.05, 0.025, 0.0125]
    if quick:
        h_vals = [0.2, 0.1, 0.05]

    lines = ["=" * 70,
             "Experiment 4: MMS Convergence Verification",
             "=" * 70, ""]
    lines.append(f"{'n':>6s}  {'Obs. Order':>12s}  {'R^2':>8s}  {'Expected':>10s}")
    lines.append("-" * 45)

    for n_val in n_values:
        ms = mms_mod.CosineBumpSolution(n_val, 5.0, amplitude=0.5)
        result = mms_mod.verify_convergence_order(ms, mms_mod.solve_manufactured_rk4, h_vals)
        if result["observed_order"] is not None:
            lines.append(f"{n_val:6.2f}  {result['observed_order']:12.3f}  "
                         f"{result['r_squared']:8.4f}  {'4.0':>10s}")
        else:
            lines.append(f"{n_val:6.2f}  {'N/A':>12s}")

    return "\n".join(lines)


def experiment_5_method_comparison(quick: bool = False) -> str:
    """Compare all methods on n=1 (has exact solution)."""
    import time as time_mod

    # Load all solver modules
    rk4_mod = _load_module("rk4", "Ronge-Kutta.py")
    shoot_mod = _load_module("shoot", "shooting.py")
    rich_mod = _load_module("rich", "richardson.py")

    n_test = 1.0
    xi_max = math.pi
    exact_func = lambda xi: np.where(xi != 0, np.sin(xi) / xi, 1.0)
    h_values = [2e-2, 1e-2, 5e-3, 2.5e-3]
    if quick:
        h_values = [1e-2, 5e-3]

    lines = ["=" * 70,
             "Experiment 5: Multi-Method Comparison (n=1)",
             "=" * 70, ""]
    lines.append(f"{'Method':>25s}  {'h':>10s}  {'Error':>14s}  {'Time':>10s}  {'xi_1':>12s}")
    lines.append("-" * 80)

    for h_val in h_values:
        # RK4
        t0 = time_mod.perf_counter()
        sol = rk4_mod.solve_lane_emden_rk4(n=n_test, epsilon=1e-4, h=h_val, xi_max=xi_max)
        t1 = time_mod.perf_counter()
        xi_eval = np.linspace(1e-4, xi_max, 2000)
        err = np.max(np.abs(np.interp(xi_eval, sol.xi, sol.theta) - exact_func(xi_eval)))
        lines.append(f"{'RK4':>25s}  {h_val:10.2e}  {err:14.6e}  {t1-t0:10.4f}  {sol.first_zero:.8f}")

        # Shooting
        t0 = time_mod.perf_counter()
        sol_s = shoot_mod.shoot_to_surface(n_test, epsilon=1e-4, h=h_val)
        t1 = time_mod.perf_counter()
        xi_eval = np.linspace(1e-4, xi_max, 2000)
        err_s = np.max(np.abs(np.interp(xi_eval, sol_s.xi, sol_s.theta) - exact_func(xi_eval)))
        lines.append(f"{'Shooting':>25s}  {h_val:10.2e}  {err_s:14.6e}  {t1-t0:10.4f}  {sol_s.xi_1:.8f}")

    return "\n".join(lines)


def experiment_6_isothermal(quick: bool = False) -> str:
    """Isothermal sphere solution."""
    iso_mod = _load_module("iso", "isothermal.py")

    lines = ["=" * 70,
             "Experiment 6: Isothermal Sphere (n -> infinity)",
             "=" * 70, ""]

    xi_max_values = [10.0, 30.0, 50.0]
    if quick:
        xi_max_values = [10.0, 30.0]

    lines.append(f"{'xi_max':>10s}  {'psi(xi_max)':>14s}  {'Asymptotic':>14s}")
    lines.append("-" * 45)
    for xm in xi_max_values:
        sol = iso_mod.solve_isothermal_rk4(epsilon=1e-4, h=2e-2, xi_max=xm)
        psi_asy = iso_mod.isothermal_asymptotic(np.array([xm]))[0]
        lines.append(f"{xm:10.1f}  {sol.psi[-1]:14.6f}  {psi_asy:14.6f}")

    return "\n".join(lines)


def experiment_7_tov(quick: bool = False) -> str:
    """TOV mass-radius relations."""
    tov_mod = _load_module("tov", "tov.py")

    n_values = [1.0, 1.5, 2.0]
    if quick:
        n_values = [1.5]

    lines = ["=" * 70,
             "Experiment 7: TOV Mass-Radius Relations",
             "=" * 70, ""]

    for n_val in n_values:
        lines.append(f"\n--- n = {n_val:g} ---")
        lines.append(f"{'sigma':>8s}  {'Radius':>12s}  {'Mass':>12s}")
        lines.append("-" * 38)

        sigmas = [0.0, 0.1, 0.2, 0.3]
        if quick:
            sigmas = [0.0, 0.2]
        try:
            sig, rad, mass = tov_mod.mass_radius_curve(n_val, sigmas)
            for s, r, m in zip(sig, rad, mass):
                lines.append(f"{s:8.2f}  {r:12.6f}  {m:12.6f}")
        except Exception as e:
            lines.append(f"  FAILED: {e}")

    return "\n".join(lines)


def experiment_8_continuation(quick: bool = False) -> str:
    """Parameter continuation across n values."""
    pc_mod = _load_module("pc", "parameter_continuation.py")

    lines = ["=" * 70,
             "Experiment 8: Parameter Continuation (Solution Family vs n)",
             "=" * 70, ""]

    delta = 0.5 if quick else 0.25
    curve = pc_mod.trace_parameter_curve(n_start=0.0, n_end=4.5, delta_n=delta)

    lines.append(f"{'n':>6s}  {'xi_1':>12s}")
    lines.append("-" * 22)
    for n_val, xi1 in zip(curve["n_values"], curve["xi_1_values"]):
        lines.append(f"{n_val:6.2f}  {xi1:12.6f}")

    return "\n".join(lines)


# ============================================================================
# Main
# ============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Run all Lane-Emden experiments.")
    parser.add_argument("--quick", action="store_true",
                        help="Use reduced parameter sets for faster execution.")
    args = parser.parse_args()

    quick = args.quick
    mode_str = "QUICK" if quick else "FULL"

    print(f"=" * 70)
    print(f"  Lane-Emden Numerical Study -- {mode_str} Experiment Suite")
    print(f"=" * 70)
    print(f"  Output directory: {OUTPUT_DIR}")
    print(f"  Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    experiments = [
        ("Convergence (Basic)", experiment_1_convergence_basic),
        ("Physical Quantities", experiment_2_physical_quantities),
        ("Geometry Comparison", experiment_3_geometry_comparison),
        ("MMS Verification", experiment_4_manufactured_convergence),
        ("Method Comparison", experiment_5_method_comparison),
        ("Isothermal Sphere", experiment_6_isothermal),
        ("TOV Equations", experiment_7_tov),
        ("Parameter Continuation", experiment_8_continuation),
    ]

    all_output = []
    all_output.append(f"Lane-Emden Numerical Study Report ({mode_str} Mode)")
    all_output.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    all_output.append("=" * 70)
    all_output.append("")

    for name, func in experiments:
        print(f"Running: {name}...")
        try:
            result = func(quick=quick)
            all_output.append(result)
            all_output.append("")
            print(f"  OK")
        except Exception as e:
            msg = f"  FAILED: {e}"
            print(msg)
            all_output.append(f"Experiment '{name}': FAILED - {e}")
            all_output.append("")

    # Write summary
    summary_path = OUTPUT_DIR / "experiment_summary.txt"
    summary_path.write_text("\n".join(all_output), encoding="utf-8")
    print(f"\nSummary written to: {summary_path}")

    # Quick stats
    n_total = len(experiments)
    n_ok = sum(1 for line in all_output if line.startswith("="))
    print(f"Experiments completed: {n_ok}/{n_total}")
    print(f"Done at: {time.strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
