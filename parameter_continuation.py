from __future__ import annotations

from typing import Optional

import numpy as np

from data_input import load_reference_data


def trace_parameter_curve(
    n_start: float = 0.0,
    n_end: float = 4.9,
    delta_n: float = 0.1,
    epsilon: float = 1e-4,
    h: float = 2e-3,
) -> dict:
    """Trace the solution family as a function of polytropic index n.

    Uses numerical continuation: starts from n_start (where an exact solution
    is available) and increments n by delta_n. At each step, the previous
    solution serves as the initial guess for the finite-difference solver.

    Returns
    -------
    dict with keys: n_values, xi_1_values, theta_prime_1_values, mass_values,
    central_condensation_values.
    """
    import importlib.util
    import sys
    from pathlib import Path

    # Load FD solver
    fd_path = Path(__file__).resolve().parent / "finite-difference.py"
    spec = importlib.util.spec_from_file_location("fd_cont", fd_path)
    fd_mod = importlib.util.module_from_spec(spec)
    sys.modules["fd_cont"] = fd_mod
    spec.loader.exec_module(fd_mod)

    ref = load_reference_data()

    n_values = []
    xi_1_values = []
    tp_1_values = []
    mass_values = []

    prev_solution = None

    for n in np.arange(n_start, n_end + delta_n / 2, delta_n):
        n = round(float(n), 10)  # avoid floating point drift

        prop = ref.get_global_property(n)
        if prop is None or prop.xi_1 is None:
            continue

        xi_max = prop.xi_1

        try:
            # If we have a previous solution, use it as initial guess
            if prev_solution is not None:
                # Interpolate previous solution onto current grid
                num_intervals = max(100, int(np.ceil((xi_max - epsilon) / h)))
                xi_new = np.linspace(epsilon, xi_max, num_intervals + 1)
                theta_guess = np.interp(xi_new, prev_solution.xi, prev_solution.theta)

                # Use this as initial guess by monkey-patching
                original_guess = fd_mod.initial_guess

                def continuation_guess(xi_arr, th_left, th_right):
                    guess = np.interp(xi_arr, xi_new, theta_guess)
                    guess[0] = th_left
                    guess[-1] = th_right
                    return np.maximum(guess, 1e-6)

                fd_mod.initial_guess = continuation_guess
                fd_mod.solve_lane_emden_finite_difference.__globals__[
                    "initial_guess"] = continuation_guess

            sol = fd_mod.solve_lane_emden_finite_difference(
                n=n, epsilon=epsilon,
                num_intervals=max(100, int(np.ceil((xi_max - epsilon) / h))),
                xi_max=xi_max, theta_right=0.0,
            )

            # Restore original guess function
            if prev_solution is not None:
                fd_mod.initial_guess = original_guess
                fd_mod.solve_lane_emden_finite_difference.__globals__[
                    "initial_guess"] = original_guess

            prev_solution = sol

            n_values.append(n)
            xi_1_values.append(sol.first_zero or xi_max)
            tp_1_values.append(float(np.interp(sol.first_zero or xi_max,
                                                sol.xi, sol.theta_prime)))

            # Mass parameter
            if sol.first_zero:
                mass = -sol.first_zero ** 2 * tp_1_values[-1]
                mass_values.append(mass)
            else:
                mass_values.append(np.nan)

        except Exception as e:
            print(f"  n={n:g}: continuation failed - {e}")
            prev_solution = None
            continue

    return {
        "n_values": n_values,
        "xi_1_values": xi_1_values,
    }


def trace_critical_behavior(
    n_approach: list[float],
    epsilon: float = 1e-4,
    h: float = 2e-3,
) -> dict:
    """Study the approach to n=5 critical behavior.

    As n -> 5, xi_1 -> infinity and the mass parameter approaches a constant.
    """
    import importlib.util
    import sys
    from pathlib import Path

    fd_path = Path(__file__).resolve().parent / "finite-difference.py"
    spec = importlib.util.spec_from_file_location("fd_crit", fd_path)
    fd_mod = importlib.util.module_from_spec(spec)
    sys.modules["fd_crit"] = fd_mod
    spec.loader.exec_module(fd_mod)

    results = {"n": [], "xi_1": [], "theta_prime_1": [], "mass": []}

    for n in n_approach:
        if n >= 5.0:
            continue

        ref = load_reference_data()
        prop = ref.get_global_property(n)
        if prop is None:
            continue
        xi_max = min(prop.xi_1 or 50.0, 50.0)  # cap at 50

        try:
            sol = fd_mod.solve_lane_emden_finite_difference(
                n=n, epsilon=epsilon,
                num_intervals=max(200, int(np.ceil((xi_max - epsilon) / h))),
                xi_max=xi_max, theta_right=0.0,
            )

            xi_1 = sol.first_zero or xi_max
            tp_1 = float(np.interp(xi_1, sol.xi, sol.theta_prime))
            mass = -xi_1 ** 2 * tp_1

            results["n"].append(n)
            results["xi_1"].append(xi_1)
            results["theta_prime_1"].append(tp_1)
            results["mass"].append(mass)

        except Exception as e:
            print(f"  n={n:g}: {e}")

    return results


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from pathlib import Path

    print("=" * 70)
    print("Parameter Continuation: Solution Family vs n")
    print("=" * 70)

    # Trace the solution curve
    print("\nTracing solution family from n=0 to n=4.5...")
    curve = trace_parameter_curve(n_start=0.0, n_end=4.5, delta_n=0.5,
                                   epsilon=1e-4, h=2e-3)

    print(f"\n{'n':>6s}  {'xi_1':>12s}")
    print("-" * 25)
    for n_val, xi1 in zip(curve["n_values"], curve["xi_1_values"]):
        print(f"{n_val:6.2f}  {xi1:12.6f}")

    # Plot
    output_dir = Path(__file__).resolve().parent / "output_initial"
    output_dir.mkdir(exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # xi_1 vs n
    ax1.plot(curve["n_values"], curve["xi_1_values"], "o-", linewidth=1.5)
    ax1.set_xlabel(r"$n$")
    ax1.set_ylabel(r"$\xi_1$ (first zero)")
    ax1.set_title("First Zero vs Polytropic Index")
    ax1.grid(True, alpha=0.3)
    # Add reference line at n=5
    ax1.axvline(5.0, color="red", linestyle="--", alpha=0.5, label="n=5 (critical)")

    # Reference values for comparison
    ref = load_reference_data()
    ref_n = ref.available_global_n()
    ref_xi1 = [ref.get_global_property(n).xi_1 for n in ref_n]
    ax1.scatter(ref_n, ref_xi1, marker="x", color="red", s=40,
                zorder=5, label="Reference")
    ax1.legend()

    # Critical behavior near n=5
    print("\nApproaching n=5 critical point...")
    n_fine = np.arange(4.0, 4.95, 0.05)
    crit = trace_critical_behavior(list(n_fine), epsilon=1e-4, h=2e-3)

    if crit["n"]:
        ax2.semilogy(crit["n"], crit["xi_1"], "o-", linewidth=1.5)
        ax2.set_xlabel(r"$n$")
        ax2.set_ylabel(r"$\xi_1$ (log scale)")
        ax2.set_title("Critical Divergence: xi_1 as n -> 5")
        ax2.grid(True, alpha=0.3)
        ax2.axvline(5.0, color="red", linestyle="--", alpha=0.5)

        # Fit to check scaling: xi_1 ~ (5-n)^{-1/2}
        if len(crit["n"]) >= 4:
            log_dn = np.log(5.0 - np.array(crit["n"]))
            log_xi1 = np.log(crit["xi_1"])
            slope = np.polyfit(log_dn, log_xi1, 1)[0]
            print(f"  Scaling: xi_1 ~ (5-n)^{slope:.3f} (expected -0.5)")

    plt.tight_layout()
    plt.savefig(output_dir / "parameter_continuation.png", dpi=200)
    plt.close()
    print(f"\nPlot saved to {output_dir / 'parameter_continuation.png'}")
