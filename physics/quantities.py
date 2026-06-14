from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import sys
from pathlib import Path

import numpy as np

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from data_input import (
    GlobalProperty,
    LaneEmdenReferenceData,
    load_reference_data,
)


@dataclass(frozen=True)
class PhysicalQuantities:
    """Physical quantities derived from a Lane-Emden solution.

    These quantities characterize the global structure of a polytropic star
    and are computed from the dimensionless Lane-Emden solution.
    """
    n: float
    xi_1: Optional[float]           # dimensionless radius (first zero)
    theta_prime_1: Optional[float]  # surface derivative -xi_1^2 * theta'(xi_1)

    # Dimensionless quantities (from Lane-Emden solution alone)
    mass_parameter: Optional[float]       # -xi_1^2 * theta'(xi_1) = dimensionless mass
    central_condensation: Optional[float] # rho_c / <rho>
    potential_energy: Optional[float]     # W_n = dimensionless gravitational energy

    # Derived scaling relations (require physical constants for absolute values)
    mass_radius_exponent: Optional[float]  # d ln M / d ln R for fixed K

    @property
    def summary(self) -> str:
        """One-line summary of key quantities."""
        parts = [f"n={self.n:g}"]
        if self.xi_1 is not None:
            parts.append(f"xi_1={self.xi_1:.6f}")
        if self.mass_parameter is not None:
            parts.append(f"mass_param={self.mass_parameter:.6f}")
        if self.central_condensation is not None:
            parts.append(f"rho_c/<rho>={self.central_condensation:.3f}")
        return ", ".join(parts)


def compute_physical_quantities(
    n: float,
    xi: np.ndarray,
    theta: np.ndarray,
    theta_prime: np.ndarray,
    first_zero: Optional[float] = None,
) -> PhysicalQuantities:
    """Compute all physical quantities from a Lane-Emden numerical solution.

    Parameters
    ----------
    n:
        Polytropic index.
    xi, theta, theta_prime:
        Numerical solution arrays.
    first_zero:
        Pre-computed first zero. If None, it will be found from the data.

    Returns
    -------
    PhysicalQuantities with all computable derived values.
    """
    # Determine first zero
    xi_1 = first_zero
    if xi_1 is None:
        for i in range(1, len(theta)):
            if theta[i] <= 0.0:
                # Linear interpolation
                if theta[i - 1] > 0:
                    xi_1 = xi[i - 1] - theta[i - 1] * (xi[i] - xi[i - 1]) / (theta[i] - theta[i - 1])
                else:
                    xi_1 = xi[i]
                break
        if xi_1 is None and abs(theta[-1]) < 1e-10:
            xi_1 = xi[-1]

    # Surface derivative at xi_1
    theta_prime_1: Optional[float] = None
    if xi_1 is not None:
        theta_prime_1 = float(np.interp(xi_1, xi, theta_prime))

    # Dimensionless mass: -xi_1^2 * theta'(xi_1)
    mass_parameter: Optional[float] = None
    if xi_1 is not None and theta_prime_1 is not None:
        mass_parameter = -xi_1 ** 2 * theta_prime_1

    # Central condensation: rho_c / <rho> = xi_1^3 / (3 * mass_parameter)
    central_condensation: Optional[float] = None
    if xi_1 is not None and mass_parameter is not None and mass_parameter > 0:
        central_condensation = xi_1 ** 3 / (3.0 * mass_parameter)

    # Gravitational potential energy parameter W_n
    # W_n = -3 / (5 - n) * xi_1 * theta'(xi_1)^2 / [integral] but we use
    # a simpler formula for the dimensionless binding energy
    potential_energy: Optional[float] = None
    if xi_1 is not None and theta_prime_1 is not None and n < 5.0:
        # W_n = 3/(5-n) * (mass_parameter / xi_1^3) * (...)
        # Standard formula: W_n = 3 * integral_0^{xi_1} theta^n * xi^2 dxi / xi_1^3
        # We approximate the integral numerically
        if len(xi) > 2:
            mask = xi <= xi_1
            xi_masked = xi[mask]
            theta_masked = theta[mask]
            integrand = theta_masked ** n * xi_masked ** 2
            integral = float(np.trapz(integrand, xi_masked)) if len(xi_masked) >= 2 else 0.0
            if xi_1 > 0 and mass_parameter is not None and mass_parameter > 0:
                potential_energy = -3.0 / (5.0 - n) * (mass_parameter ** 2) / xi_1 ** 3

    # Mass-radius exponent for fixed K, n
    # M ∝ R^{(3-n)/(1-n)} for n != 1
    mass_radius_exponent: Optional[float] = None
    if n != 1.0:
        mass_radius_exponent = (3.0 - n) / (1.0 - n)

    return PhysicalQuantities(
        n=n,
        xi_1=xi_1,
        theta_prime_1=theta_prime_1,
        mass_parameter=mass_parameter,
        central_condensation=central_condensation,
        potential_energy=potential_energy,
        mass_radius_exponent=mass_radius_exponent,
    )


def compute_mass_radius_relation(
    n: float,
    K: float,       # polytropic constant in cgs
    rho_c: float,   # central density in g/cm^3
    G: float = 6.67430e-8,  # gravitational constant in cgs
) -> dict[str, float]:
    """Compute the physical mass and radius for given polytropic parameters.

    Uses the analytic scaling relations:
        R = alpha^{-1} * xi_1
        M = 4 * pi * rho_c * alpha^{-3} * (-xi_1^2 * theta'(xi_1))

    where alpha^2 = 4*pi*G / (K*(n+1)) * rho_c^{(1-n)/n}

    Parameters
    ----------
    n:
        Polytropic index. Must be < 5 for finite radius.
    K:
        Polytropic constant (cgs units).
    rho_c:
        Central density (g/cm^3).
    G:
        Gravitational constant (cgs).

    Returns
    -------
    dict with R (cm), R_solar, M (g), M_solar, and alpha.
    """
    import math

    if n >= 5.0:
        raise ValueError(f"n={n} >= 5: infinite radius, mass-radius relation not defined.")

    ref = load_reference_data()
    prop = ref.get_global_property(n)
    if prop is None or prop.xi_1 is None or prop.theta_prime_surface is None:
        raise ValueError(f"No reference data for n={n}.")

    xi_1 = prop.xi_1
    theta_prime_1 = prop.theta_prime_surface

    # alpha from definition
    alpha_sq = 4.0 * math.pi * G / (K * (n + 1.0)) * rho_c ** (1.0 - 1.0 / n)
    alpha = math.sqrt(alpha_sq)

    # Radius
    R = xi_1 / alpha
    R_solar = R / 6.957e10  # solar radius in cm

    # Mass
    M = 4.0 * math.pi * rho_c / alpha ** 3 * (-xi_1 ** 2 * theta_prime_1)
    M_solar = M / 1.989e33  # solar mass in g

    return {
        "R_cm": R,
        "R_solar": R_solar,
        "M_g": M,
        "M_solar": M_solar,
        "alpha": alpha,
        "xi_1": xi_1,
        "theta_prime_surface": theta_prime_1,
    }


def compare_with_reference(
    n: float,
    xi: np.ndarray,
    theta: np.ndarray,
    theta_prime: np.ndarray,
    first_zero: Optional[float] = None,
) -> dict:
    """Compare computed physical quantities with reference values.

    Returns a dict with computed values, reference values, and relative errors.
    """
    computed = compute_physical_quantities(n, xi, theta, theta_prime, first_zero)

    ref = load_reference_data()
    prop = ref.get_global_property(n)

    comparison = {
        "n": n,
        "xi_1_computed": computed.xi_1,
        "xi_1_reference": prop.xi_1 if prop else None,
        "theta_prime_1_computed": computed.theta_prime_1,
        "theta_prime_1_reference": prop.theta_prime_surface if prop else None,
        "mass_param_computed": computed.mass_parameter,
        "mass_param_reference": prop.mass if prop else None,
        "central_condensation_computed": computed.central_condensation,
        "central_condensation_reference": prop.rho_c_over_rho_avg if prop else None,
        "potential_energy_computed": computed.potential_energy,
        "potential_energy_reference": prop.W_n if prop else None,
    }

    # Relative errors
    for key in ("xi_1", "theta_prime_1", "mass_param", "central_condensation", "potential_energy"):
        comp_val = comparison[f"{key}_computed"]
        ref_val = comparison[f"{key}_reference"]
        if comp_val is not None and ref_val is not None and ref_val != 0:
            comparison[f"{key}_rel_error"] = abs(comp_val - ref_val) / abs(ref_val)
        else:
            comparison[f"{key}_rel_error"] = None

    return comparison


def generate_physical_table(
    n_values: Optional[list[float]] = None,
    solver_func=None,
    **solver_kwargs,
) -> str:
    """Generate a formatted table of physical quantities across n values.

    Parameters
    ----------
    n_values:
        List of n values. Defaults to all available reference n values.
    solver_func:
        Function with signature solver_func(n, **kwargs) -> solution.
        If None, uses finite-difference solver as default.
    solver_kwargs:
        Passed to solver_func.

    Returns
    -------
    Formatted string table.
    """
    if n_values is None:
        ref = load_reference_data()
        n_values = ref.available_global_n()

    if solver_func is None:
        # Default to finite-difference solver
        import importlib.util
        import sys
        from pathlib import Path

        fd_path = Path(__file__).resolve().parent.parent / "solvers" / "fd.py"
        spec = importlib.util.spec_from_file_location("fd_temp", fd_path)
        fd_mod = importlib.util.module_from_spec(spec)
        sys.modules["fd_temp"] = fd_mod
        spec.loader.exec_module(fd_mod)

        def solver_func(n, **kw):
            ref_data = load_reference_data()
            prop = ref_data.get_global_property(n)
            xi_max = prop.xi_1 if prop and prop.xi_1 else 10.0
            eps = kw.get("epsilon", 1e-6)
            n_int = kw.get("num_intervals", 2000)
            return fd_mod.solve_lane_emden_finite_difference(
                n=n, epsilon=eps, num_intervals=n_int,
                xi_max=xi_max, theta_right=0.0,
            )

    lines = []
    header = (
        f"{'n':>6s}  {'xi_1':>10s}  {'-xi_1^2 θ\'(ξ_1)':>16s}  "
        f"{'ρ_c/<ρ>':>10s}  {'W_n':>10s}  {'M-R exp':>10s}"
    )
    lines.append(header)
    lines.append("-" * len(header))

    for n in n_values:
        try:
            sol = solver_func(n, **solver_kwargs)
            pq = compute_physical_quantities(
                n, sol.xi, sol.theta, sol.theta_prime, sol.first_zero,
            )
            lines.append(
                f"{pq.n:6.2f}  "
                f"{pq.xi_1 or float('nan'):10.6f}  "
                f"{pq.mass_parameter or float('nan'):16.8f}  "
                f"{pq.central_condensation or float('nan'):10.3f}  "
                f"{pq.potential_energy or float('nan'):10.6f}  "
                f"{pq.mass_radius_exponent or float('nan'):10.4f}"
            )
        except Exception as e:
            lines.append(f"{n:6.2f}  {'FAILED':>10s}: {str(e)[:50]}")

    return "\n".join(lines)


if __name__ == "__main__":
    print("=" * 70)
    print("Physical Quantities from Lane-Emden Solutions")
    print("=" * 70)

    table = generate_physical_table()
    print("\n" + table)

    print("\n" + "=" * 70)
    print("Comparison with Reference Data")
    print("=" * 70)

    ref = load_reference_data()
    for n in ref.available_global_n():
        prop = ref.get_global_property(n)
        if prop is None:
            continue
        print(f"\nn = {n:g}:")
        print(f"  Reference:  xi_1={prop.xi_1}, mass={prop.mass}, "
              f"rho_c/<rho>={prop.rho_c_over_rho_avg}, W_n={prop.W_n}")

    print("\n" + "=" * 70)
    print("Example: Mass-Radius Relation for n=3 polytrope")
    print("=" * 70)
    # n=3 corresponds to relativistic degenerate electron gas (Chandrasekhar limit)
    try:
        mr = compute_mass_radius_relation(
            n=3.0, K=3.841e14, rho_c=1.0e6,  # typical WD values
        )
        print(f"  R = {mr['R_cm']:.3e} cm = {mr['R_solar']:.4f} R_sun")
        print(f"  M = {mr['M_g']:.3e} g = {mr['M_solar']:.4f} M_sun")
        print(f"  alpha = {mr['alpha']:.6e}")
    except Exception as e:
        print(f"  Failed: {e}")

    # Physical quantities evolution plot
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from pathlib import Path

    OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output_initial"
    OUTPUT_DIR.mkdir(exist_ok=True)

    # Collect computed and reference values
    ref = load_reference_data()
    n_vals = ref.available_global_n()
    xi1_comp = []
    xi1_ref = []
    mass_comp = []
    mass_ref = []
    cond_comp = []
    cond_ref = []

    for n in n_vals:
        prop = ref.get_global_property(n)
        if prop is None:
            continue
        try:
            # Quick FD solve
            import importlib.util, sys
            fd_p = Path(__file__).resolve().parent.parent / "solvers" / "fd.py"
            spec = importlib.util.spec_from_file_location("fd_pq", fd_p)
            fd_m = importlib.util.module_from_spec(spec)
            sys.modules["fd_pq"] = fd_m
            spec.loader.exec_module(fd_m)
            n_int = max(100, int(np.ceil((prop.xi_1 - 1e-4) / 2e-3)))
            sol = fd_m.solve_lane_emden_finite_difference(
                n=n, epsilon=1e-4, num_intervals=n_int,
                xi_max=prop.xi_1, theta_right=0.0,
            )
            pq = compute_physical_quantities(n, sol.xi, sol.theta, sol.theta_prime, sol.first_zero)
            xi1_comp.append(pq.xi_1)
            xi1_ref.append(prop.xi_1)
            mass_comp.append(pq.mass_parameter)
            mass_ref.append(prop.mass)
            cond_comp.append(pq.central_condensation)
            cond_ref.append(prop.rho_c_over_rho_avg)
        except Exception:
            continue

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # xi_1 vs n
    ax = axes[0]
    ax.plot(n_vals[:len(xi1_comp)], xi1_comp, "o-", linewidth=1.5, label="Computed (FD)")
    ax.plot(n_vals[:len(xi1_ref)], xi1_ref, "s--", linewidth=1.5, alpha=0.7, label="Reference")
    ax.set_xlabel("n")
    ax.set_ylabel(r"$\xi_1$")
    ax.set_title("First Zero")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axvline(5.0, color="red", linestyle=":", alpha=0.5)

    # Mass parameter
    ax = axes[1]
    ax.plot(n_vals[:len(mass_comp)], mass_comp, "o-", linewidth=1.5, label="Computed (FD)")
    ax.plot(n_vals[:len(mass_ref)], mass_ref, "s--", linewidth=1.5, alpha=0.7, label="Reference")
    ax.set_xlabel("n")
    ax.set_ylabel(r"$-\xi_1^2 \theta'(\xi_1)$")
    ax.set_title("Dimensionless Mass")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Central condensation (log scale)
    ax = axes[2]
    ax.semilogy(n_vals[:len(cond_comp)], cond_comp, "o-", linewidth=1.5, label="Computed (FD)")
    ax.semilogy(n_vals[:len(cond_ref)], cond_ref, "s--", linewidth=1.5, alpha=0.7, label="Reference")
    ax.set_xlabel("n")
    ax.set_ylabel(r"$\rho_c / \langle\rho\rangle$")
    ax.set_title("Central Condensation")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.suptitle("Physical Quantities from Lane-Emden Solutions", fontsize=13)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "physical_quantities.png", dpi=200)
    plt.close()
    print(f"\nPlot saved to {OUTPUT_DIR / 'physical_quantities.png'}")
