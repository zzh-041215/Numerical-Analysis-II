from __future__ import annotations

from typing import Callable, Optional

import numpy as np

from data_input import load_reference_data


def estimate_physical_uncertainty(
    n: float,
    solver_func: Callable,
    h_values: list[float],
    epsilon: float = 1e-4,
    **solver_kwargs,
) -> dict:
    """Propagate numerical discretization error to physical quantities.

    Computes each physical quantity at several grid resolutions and estimates
    the uncertainty by Richardson extrapolation and grid convergence index.

    Parameters
    ----------
    n:
        Polytropic index.
    solver_func:
        Function with signature solver_func(n, h, **kwargs) -> solution.
        The solution must have attributes: xi, theta, theta_prime, first_zero.
    h_values:
        List of step sizes (finest to coarsest).
    epsilon:
        Starting point.
    solver_kwargs:
        Additional arguments passed to solver_func.

    Returns
    -------
    dict with keys for each physical quantity, containing:
        - value: best estimate (from finest grid or Richardson extrapolation)
        - uncertainty: estimated standard uncertainty
        - values_per_h: list of values at each resolution
        - gci: Grid Convergence Index (if >=3 grids available)
    """
    from physical_quantities import compute_physical_quantities

    h_values = sorted(h_values)  # finest to coarsest

    quantities_per_h = {
        "xi_1": [],
        "theta_prime_1": [],
        "mass_parameter": [],
        "central_condensation": [],
    }

    for h in h_values:
        try:
            sol = solver_func(n, epsilon=epsilon, h=h, **solver_kwargs)
            pq = compute_physical_quantities(
                n, sol.xi, sol.theta, sol.theta_prime, sol.first_zero,
            )

            quantities_per_h["xi_1"].append(pq.xi_1)
            quantities_per_h["theta_prime_1"].append(pq.theta_prime_1)
            quantities_per_h["mass_parameter"].append(pq.mass_parameter)
            quantities_per_h["central_condensation"].append(pq.central_condensation)
        except Exception:
            # Fill with None for failed resolutions
            for key in quantities_per_h:
                quantities_per_h[key].append(None)

    # Remove failed entries
    valid_indices = [i for i, v in enumerate(quantities_per_h["xi_1"]) if v is not None]
    if len(valid_indices) < 2:
        return {"error": "Insufficient valid solutions"}

    h_valid = [h_values[i] for i in valid_indices]

    results = {}
    for qty_name in quantities_per_h:
        values = [quantities_per_h[qty_name][i] for i in valid_indices]

        if len(values) < 2:
            results[qty_name] = {
                "value": values[0] if values else None,
                "uncertainty": None,
                "values_per_h": values,
                "gci": None,
            }
            continue

        # Best value: finest grid result
        best_value = values[0]

        # Richardson extrapolation estimate
        if len(values) >= 2:
            # Estimate order from last two grids
            if len(values) >= 3 and values[0] is not None and values[1] is not None and values[2] is not None:
                # Estimate observed order
                e32 = abs(values[2] - values[1])
                e21 = abs(values[1] - values[0])
                if e32 > 0 and e21 > 0:
                    p_est = np.log(e32 / e21) / np.log(2.0)
                    p_est = max(1.0, min(p_est, 6.0))  # clamp to reasonable range
                else:
                    p_est = 2.0
            else:
                p_est = 2.0  # default for FD

            # Richardson extrapolation
            if values[0] is not None and values[1] is not None:
                rich_value = (2.0 ** p_est * values[0] - values[1]) / (2.0 ** p_est - 1.0)
                # Uncertainty: difference between Richardson and finest grid
                uncertainty = abs(rich_value - values[0])
            else:
                rich_value = best_value
                uncertainty = abs(values[0] - values[1]) if values[1] is not None else None

            # Grid Convergence Index (GCI) for 3+ grids
            gci = None
            if len(values) >= 3 and values[0] is not None and values[1] is not None and values[2] is not None:
                r = 2.0  # grid refinement ratio
                e21_abs = abs(values[0] - values[1])
                Fs = 1.25  # safety factor for 3 grids
                gci = Fs * e21_abs / (r ** p_est - 1.0)
        else:
            rich_value = best_value
            uncertainty = None
            gci = None

        results[qty_name] = {
            "value": best_value,
            "richardson_value": rich_value,
            "uncertainty": uncertainty,
            "relative_uncertainty": uncertainty / abs(best_value) if best_value and uncertainty else None,
            "values_per_h": values,
            "gci": gci,
            "observed_order": p_est if len(values) >= 3 else None,
        }

    return results


def generate_uncertainty_report(
    n_values: list[float],
    solver_func: Callable,
    h_values: list[float] = None,
    **solver_kwargs,
) -> str:
    """Generate a formatted uncertainty report for multiple n values.

    Returns a formatted string table.
    """
    if h_values is None:
        h_values = [5e-3, 1e-2, 2e-2]

    lines = []
    lines.append("Uncertainty Propagation Report")
    lines.append("=" * 70)
    lines.append(f"Grid resolutions: {', '.join(f'{h:.0e}' for h in sorted(h_values))}")
    lines.append("")
    lines.append(f"{'n':>6s}  {'xi_1':>12s}  {'+/-':>10s}  "
                 f"{'Mass':>12s}  {'+/-':>10s}  {'GCI(xi_1)':>10s}")
    lines.append("-" * 75)

    for n in n_values:
        try:
            results = estimate_physical_uncertainty(n, solver_func, h_values,
                                                    **solver_kwargs)
            if "error" in results:
                lines.append(f"{n:6.2f}  {results['error']}")
                continue

            xi1 = results["xi_1"]
            mass = results["mass_parameter"]

            xi1_str = f"{xi1['value']:.6f}" if xi1['value'] else "N/A"
            xi1_unc = f"{xi1['uncertainty']:.2e}" if xi1['uncertainty'] else "N/A"
            mass_str = f"{mass['value']:.6f}" if mass['value'] else "N/A"
            mass_unc = f"{mass['uncertainty']:.2e}" if mass['uncertainty'] else "N/A"
            gci_str = f"{xi1['gci']:.2e}" if xi1['gci'] else "N/A"

            lines.append(f"{n:6.2f}  {xi1_str:>12s}  {xi1_unc:>10s}  "
                         f"{mass_str:>12s}  {mass_unc:>10s}  {gci_str:>10s}")
        except Exception as e:
            lines.append(f"{n:6.2f}  FAILED: {str(e)[:40]}")

    return "\n".join(lines)


if __name__ == "__main__":
    import importlib.util
    import sys
    from pathlib import Path

    print("=" * 70)
    print("Uncertainty Propagation Analysis")
    print("=" * 70)

    # Load FD solver for uncertainty analysis
    fd_path = Path(__file__).resolve().parent / "finite-difference.py"
    spec = importlib.util.spec_from_file_location("fd_unc", fd_path)
    fd_mod = importlib.util.module_from_spec(spec)
    sys.modules["fd_unc"] = fd_mod
    spec.loader.exec_module(fd_mod)

    def fd_solver(n, epsilon=1e-4, h=5e-3, **kw):
        ref = load_reference_data()
        prop = ref.get_global_property(n)
        xi_max = prop.xi_1 if prop and prop.xi_1 else 10.0
        num_intervals = max(50, int(np.ceil((xi_max - epsilon) / h)))
        return fd_mod.solve_lane_emden_finite_difference(
            n=n, epsilon=epsilon, num_intervals=num_intervals,
            xi_max=xi_max, theta_right=0.0,
        )

    h_vals = [2.5e-3, 5e-3, 1e-2]  # finest to coarsest

    report = generate_uncertainty_report(
        [0.0, 1.0, 1.5, 2.0, 3.0], fd_solver, h_vals,
    )
    print("\n" + report)
