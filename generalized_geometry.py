from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from data_input import LaneEmdenPoint, load_reference_data


@dataclass(frozen=True)
class GeneralizedGeometrySolution:
    """Solution of the generalized Lane-Emden equation for any geometry.

    (1/xi^k) d/dxi (xi^k dtheta/dxi) + theta^n = 0

    where k = 0 (Slab), 1 (Cylinder), 2 (Sphere).
    """
    n: float
    k: int            # geometry parameter
    xi: np.ndarray
    theta: np.ndarray
    theta_prime: np.ndarray
    first_zero: Optional[float]
    num_steps: int


def taylor_initial_conditions_general(n: float, k: int, epsilon: float) -> tuple[float, float]:
    """Taylor-expanded initial conditions for generalized geometry.

    The generalized equation expands to:
        theta'' + (k/xi) * theta' + theta^n = 0

    Taylor expansion around xi=0 (using evenness for k=2, and appropriate
    behavior for other k):
        theta(epsilon) = 1 - epsilon^2/(2*(k+1)) + n*epsilon^4/(8*(k+1)*(k+3)) + ...
        theta'(epsilon) = -epsilon/(k+1) + n*epsilon^3/(2*(k+1)*(k+3)) + ...
    """
    if epsilon <= 0:
        raise ValueError("epsilon must be positive.")

    c1 = 1.0 / (2.0 * (k + 1))
    c2 = n / (8.0 * (k + 1) * (k + 3))

    theta = 1.0 - c1 * epsilon**2 + c2 * epsilon**4
    theta_prime = -2.0 * c1 * epsilon + 4.0 * c2 * epsilon**3

    return theta, theta_prime


def generalized_rhs(xi: float, state: np.ndarray, n: float, k: int) -> np.ndarray:
    """RHS of the first-order generalized Lane-Emden system.

    y1' = y2
    y2' = -(k/xi) * y2 - y1^n
    """
    theta, theta_prime = state
    if theta < 0.0 and not float(n).is_integer():
        theta_power = 0.0
    else:
        theta_power = theta ** n

    return np.array(
        [theta_prime, -float(k) * theta_prime / xi - theta_power],
        dtype=float,
    )


def rk4_step_general(xi: float, state: np.ndarray, h: float, n: float, k: int) -> np.ndarray:
    """One classical RK4 step for the generalized equation."""
    k1 = generalized_rhs(xi, state, n, k)
    k2 = generalized_rhs(xi + h / 2.0, state + h * k1 / 2.0, n, k)
    k3 = generalized_rhs(xi + h / 2.0, state + h * k2 / 2.0, n, k)
    k4 = generalized_rhs(xi + h, state + h * k3, n, k)
    return state + h * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0


def solve_generalized_rk4(
    n: float,
    k: int,
    epsilon: float = 1e-6,
    h: float = 1e-3,
    xi_max: Optional[float] = None,
    stop_at_zero: bool = True,
    max_steps: int = 1_000_000,
) -> GeneralizedGeometrySolution:
    """Solve the generalized Lane-Emden equation using RK4.

    Parameters
    ----------
    n:
        Polytropic index.
    k:
        Geometry parameter: 0=Slab, 1=Cylinder, 2=Sphere.
    epsilon:
        Starting point.
    h:
        Fixed step size.
    xi_max:
        Maximum xi. If None, tries to use reference data.
    stop_at_zero:
        Whether to stop at the first zero crossing.
    max_steps:
        Maximum number of steps.

    Returns
    -------
    GeneralizedGeometrySolution.
    """
    if k not in (0, 1, 2):
        raise ValueError("k must be 0 (Slab), 1 (Cylinder), or 2 (Sphere).")

    if xi_max is None:
        # Try to get reference first zero
        ref = load_reference_data()
        # Use the sphere table for xi_1 as a rough guide
        prop = ref.get_global_property(n)
        if prop is not None and prop.xi_1 is not None:
            # Cylinder and slab have different first zeros
            # Use a generous estimate
            xi_max = prop.xi_1 * (1.5 if k < 2 else 1.0) + h
        else:
            xi_max = 15.0

    theta0, theta_prime0 = taylor_initial_conditions_general(n, k, epsilon)

    xi_list = [epsilon]
    theta_list = [theta0]
    theta_prime_list = [theta_prime0]

    xi = epsilon
    state = np.array([theta0, theta_prime0], dtype=float)
    first_zero: Optional[float] = None
    step_count = 0

    while xi < xi_max and step_count < max_steps:
        step = min(h, xi_max - xi)
        new_state = rk4_step_general(xi, state, step, n, k)
        new_xi = xi + step
        step_count += 1

        # Check for zero crossing
        if stop_at_zero and state[0] > 0.0 and new_state[0] <= 0.0:
            weight = abs(state[0]) / (abs(state[0]) + abs(new_state[0]))
            first_zero = float(xi + weight * (new_xi - xi))
            xi_list.append(first_zero)
            theta_list.append(0.0)
            theta_prime_list.append(float(new_state[1]))
            break

        if stop_at_zero and state[0] <= 0.0:
            first_zero = float(xi)
            break

        xi_list.append(new_xi)
        theta_list.append(float(new_state[0]))
        theta_prime_list.append(float(new_state[1]))

        xi = new_xi
        state = new_state

    return GeneralizedGeometrySolution(
        n=n,
        k=k,
        xi=np.array(xi_list, dtype=float),
        theta=np.array(theta_list, dtype=float),
        theta_prime=np.array(theta_prime_list, dtype=float),
        first_zero=first_zero,
        num_steps=step_count,
    )


def compare_geometries(
    n: float,
    epsilon: float = 1e-4,
    h: float = 1e-3,
    xi_max: Optional[float] = None,
) -> dict[int, GeneralizedGeometrySolution]:
    """Solve the Lane-Emden equation for all three geometries at fixed n.

    Returns a dict mapping k -> solution.
    """
    results = {}
    geometry_names = {0: "Slab", 1: "Cylinder", 2: "Sphere"}

    for k in (0, 1, 2):
        try:
            sol = solve_generalized_rk4(n=n, k=k, epsilon=epsilon, h=h, xi_max=xi_max)
            results[k] = sol
        except Exception as e:
            print(f"  {geometry_names[k]} (k={k}): FAILED - {e}")

    return results


def compare_with_horedt(
    n: float,
    k: int,
    solution: GeneralizedGeometrySolution,
) -> dict:
    """Compare a generalized solution with Horedt reference data.

    Returns comparison dict with errors at key points.
    """
    geometry_map = {0: "Slab", 1: "Cylinder", 2: "Sphere"}
    geom_name = geometry_map[k]

    ref = load_reference_data()
    points = ref.sphere_tables.get(n, [])  # Horedt data for this n

    if not points:
        return {"n": n, "k": k, "geometry": geom_name, "error": "No reference data"}

    # Filter by geometry
    geom_points = [p for p in points if p.geometry == geom_name]
    if not geom_points:
        return {"n": n, "k": k, "geometry": geom_name, "error": f"No {geom_name} data"}

    # Compare at reference points
    errors = []
    for p in geom_points:
        if p.xi > solution.xi[-1]:
            break
        if p.theta is None:
            continue
        theta_num = float(np.interp(p.xi, solution.xi, solution.theta))
        errors.append(abs(theta_num - p.theta))

    result = {
        "n": n,
        "k": k,
        "geometry": geom_name,
        "num_comparison_points": len(errors),
    }

    if errors:
        result["max_error"] = max(errors)
        result["mean_error"] = np.mean(errors)

    return result


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from pathlib import Path

    print("=" * 70)
    print("Generalized Geometry Lane-Emden Solver")
    print("=" * 70)
    print("k=0: Slab (plane-parallel), k=1: Cylinder, k=2: Sphere")

    # Compare geometries for n=1.5
    n_test = 1.5
    print(f"\n--- Comparing geometries for n = {n_test} ---")
    solutions = compare_geometries(n_test, epsilon=1e-4, h=2e-3)

    geometry_names = {0: "Slab (k=0)", 1: "Cylinder (k=1)", 2: "Sphere (k=2)"}
    for k, sol in solutions.items():
        print(f"  {geometry_names[k]}:")
        print(f"    First zero xi_1 = {sol.first_zero:.6f}" if sol.first_zero else "    No finite zero")
        print(f"    Steps: {sol.num_steps}")

    # Plot
    output_dir = Path(__file__).resolve().parent / "output_initial"
    output_dir.mkdir(exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # All three geometries at n=1.5
    ax = axes[0]
    colors = {0: "blue", 1: "green", 2: "red"}
    for k, sol in solutions.items():
        ax.plot(sol.xi, sol.theta, color=colors[k], linewidth=1.5,
                label=f"{geometry_names[k]}")
    ax.set_xlabel(r"$\xi$")
    ax.set_ylabel(r"$\theta(\xi)$")
    ax.set_title(f"Geometry comparison, n = {n_test}")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Sphere solutions for different n
    ax = axes[1]
    for n_val in (0.0, 1.0, 2.0, 3.0, 4.0):
        sol = solve_generalized_rk4(n=n_val, k=2, epsilon=1e-4, h=2e-3)
        ax.plot(sol.xi, sol.theta, linewidth=1.5, label=f"n={n_val:g}")
    ax.set_xlabel(r"$\xi$")
    ax.set_ylabel(r"$\theta(\xi)$")
    ax.set_title("Sphere (k=2): various n")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "geometry_overview.png", dpi=200)
    plt.close()
    print(f"\nPlot saved to {output_dir / 'geometry_overview.png'}")

    # Compare with Horedt reference data for sphere
    print("\n--- Comparison with Horedt reference (Sphere) ---")
    from data_input import LaneEmdenReferenceData
    ref_data = LaneEmdenReferenceData.from_csv()
    for n_val in ref_data.available_sphere_n()[:5]:  # first 5 available n
        sol = solve_generalized_rk4(n=n_val, k=2, epsilon=1e-4, h=5e-3)
        comp = compare_with_horedt(n_val, 2, sol)
        if "max_error" in comp:
            print(f"  n={n_val:g}: max_error={comp['max_error']:.2e}, "
                  f"mean_error={comp['mean_error']:.2e}, "
                  f"points={comp['num_comparison_points']}")
