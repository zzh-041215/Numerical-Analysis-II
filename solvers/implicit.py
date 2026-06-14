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
class ImplicitRKSolution:
    """Solution of Lane-Emden using implicit Runge-Kutta (Radau IIA)."""
    n: float
    xi: np.ndarray
    theta: np.ndarray
    theta_prime: np.ndarray
    first_zero: Optional[float]
    num_steps: int
    num_newton_iters: int
    newton_per_step: Optional[list] = None


# 3-stage Radau IIA Butcher tableau (order 5, L-stable)
# This method is particularly well-suited for stiff ODEs.
RADAU_C = np.array([(4.0 - np.sqrt(6.0)) / 10.0,
                     (4.0 + np.sqrt(6.0)) / 10.0,
                     1.0])
RADAU_A = np.array([
    [(88.0 - 7.0 * np.sqrt(6.0)) / 360.0,
     (296.0 - 169.0 * np.sqrt(6.0)) / 1800.0,
     (-2.0 + 3.0 * np.sqrt(6.0)) / 225.0],
    [(296.0 + 169.0 * np.sqrt(6.0)) / 1800.0,
     (88.0 + 7.0 * np.sqrt(6.0)) / 360.0,
     (-2.0 - 3.0 * np.sqrt(6.0)) / 225.0],
    [(16.0 - np.sqrt(6.0)) / 36.0,
     (16.0 + np.sqrt(6.0)) / 36.0,
     1.0 / 9.0],
])
RADAU_B = np.array([(16.0 - np.sqrt(6.0)) / 36.0,
                    (16.0 + np.sqrt(6.0)) / 36.0,
                    1.0 / 9.0])


def taylor_initial_conditions(n: float, epsilon: float) -> tuple[float, float]:
    """Taylor expansion near xi=0."""
    theta = 1.0 - epsilon**2 / 6.0 + n * epsilon**4 / 120.0
    theta_prime = -epsilon / 3.0 + n * epsilon**3 / 30.0
    return theta, theta_prime


def _radau_step(state: np.ndarray, xi: float, h: float, n: float,
                max_newton: int = 10, newton_tol: float = 1e-12) -> tuple[np.ndarray, int]:
    """One Radau IIA step: solve for stage values via simplified Newton.

    Returns (new_state, newton_iterations).
    """
    s = 3  # stages
    dim = 2  # (theta, theta_prime)

    # Initial guess for stages: use explicit Euler
    Z = np.zeros((s, dim))
    for i in range(s):
        xi_s = xi + RADAU_C[i] * h
        def rhs_simple(t, tp):
            if t < 0 and not float(n).is_integer():
                return np.array([tp, -2.0 * tp / xi_s])
            return np.array([tp, -2.0 * tp / xi_s - t ** n])

        Z[i] = state + RADAU_C[i] * h * rhs_simple(state[0], state[1])

    # Simplified Newton iteration
    total_iters = 0
    for _ in range(max_newton):
        max_correction = 0.0
        for i in range(s):
            xi_s = xi + RADAU_C[i] * h

            def rhs(t, tp):
                if t < 0 and not float(n).is_integer():
                    t_pow = 0.0
                else:
                    t_pow = t ** n
                return np.array([tp, -2.0 * tp / xi_s - t_pow])

            # Stage equation residual
            stage_res = Z[i] - state
            for j in range(s):
                stage_res -= h * RADAU_A[i, j] * rhs(Z[j][0], Z[j][1])

            # Jacobian approximation (use diagonal dominance)
            J_diag = np.eye(dim) - h * RADAU_A[i, i] * np.array([
                [0, 1],
                [-n * Z[i][0] ** (n - 1) if Z[i][0] > 0 else 0, -2.0 / xi_s],
            ])

            try:
                correction = np.linalg.solve(J_diag, -stage_res)
            except np.linalg.LinAlgError:
                correction = -stage_res / (1.0 + h)

            Z[i] += correction
            max_correction = max(max_correction, np.max(np.abs(correction)))

        total_iters += 1
        if max_correction < newton_tol:
            break

    # Combine stages
    new_state = state.copy()
    for i in range(s):
        xi_s = xi + RADAU_C[i] * h
        def rhs(t, tp):
            if t < 0 and not float(n).is_integer():
                t_pow = 0.0
            else:
                t_pow = t ** n
            return np.array([tp, -2.0 * tp / xi_s - t_pow])
        new_state += h * RADAU_B[i] * rhs(Z[i][0], Z[i][1])

    return new_state, total_iters


def solve_lane_emden_radau(
    n: float,
    epsilon: float = 1e-6,
    h: float = 1e-2,
    xi_max: Optional[float] = None,
    stop_at_zero: bool = True,
    max_steps: int = 100_000,
) -> ImplicitRKSolution:
    """Solve Lane-Emden using 3-stage Radau IIA (implicit RK, order 5, L-stable).

    Radau IIA is L-stable (perfect damping of stiff modes), making it
    suitable for n close to 5 where the problem becomes moderately stiff.
    """
    if xi_max is None:
        ref = load_reference_data()
        prop = ref.get_global_property(n)
        xi_max = (prop.xi_1 or 10.0) + h if prop else 10.0

    theta0, theta_prime0 = taylor_initial_conditions(n, epsilon)

    xi_list = [epsilon]
    theta_list = [theta0]
    tp_list = [theta_prime0]

    xi = epsilon
    state = np.array([theta0, theta_prime0], dtype=float)
    first_zero = None
    step_count = 0
    total_newton = 0
    newton_per_step = []

    while xi < xi_max and step_count < max_steps:
        step = min(h, xi_max - xi)

        # Check for near-zero
        if stop_at_zero and state[0] <= 1e-10 and state[1] < 0:
            first_zero = float(xi - state[0] / state[1])
            xi_list.append(first_zero)
            theta_list.append(0.0)
            tp_list.append(float(state[1]))
            break

        new_state, n_iters = _radau_step(state, xi, step, n)
        new_xi = xi + step
        step_count += 1
        total_newton += n_iters
        newton_per_step.append(n_iters)

        if stop_at_zero and state[0] > 0 and new_state[0] <= 0:
            w = abs(state[0]) / (abs(state[0]) + abs(new_state[0]))
            first_zero = float(xi + w * step)
            xi_list.append(first_zero)
            theta_list.append(0.0)
            tp_list.append(float(new_state[1]))
            break

        xi_list.append(new_xi)
        theta_list.append(float(new_state[0]))
        tp_list.append(float(new_state[1]))

        xi = new_xi
        state = new_state

    return ImplicitRKSolution(
        n=n, xi=np.array(xi_list), theta=np.array(theta_list),
        theta_prime=np.array(tp_list), first_zero=first_zero,
        num_steps=step_count, num_newton_iters=total_newton,
        newton_per_step=newton_per_step,
    )


if __name__ == "__main__":
    import math
    import time
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from pathlib import Path
    import importlib.util, sys

    # Import RK4 for comparison
    rk_path = Path(__file__).resolve().parent / "rk4.py"
    spec = importlib.util.spec_from_file_location("rk_temp2", rk_path)
    rk_mod = importlib.util.module_from_spec(spec)
    sys.modules["rk_temp2"] = rk_mod
    spec.loader.exec_module(rk_mod)

    OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output_initial"
    OUTPUT_DIR.mkdir(exist_ok=True)

    print("=" * 70)
    print("Implicit RK (Radau IIA) vs Explicit RK4 — Experimental Study")
    print("=" * 70)

    # --- Collect data for all (n, h, method) combinations ---
    h_values = [5e-2, 2e-2, 1e-2, 5e-3, 2e-3]
    n_values = [0.0, 3.0, 4.5]
    results = {}  # results[(n, h, method)] = dict with keys

    for n_val in n_values:
        if abs(n_val) < 1e-14:
            xi_max_val = math.sqrt(6.0)
            xi_ref = math.sqrt(6.0)
        elif abs(n_val - 1.0) < 1e-14:
            xi_max_val = math.pi
            xi_ref = math.pi
        else:
            ref = load_reference_data()
            xi_max_val = ref.get_first_zero(n_val) or 10.0
            xi_ref = xi_max_val

        print(f"\n--- n = {n_val:g} (ref xi_1 = {xi_ref:.8f}) ---")

        # Radau IIA
        print("  Radau IIA:")
        for h_test in h_values:
            try:
                t0 = time.perf_counter()
                sol = solve_lane_emden_radau(n=n_val, epsilon=1e-4, h=h_test,
                                             xi_max=xi_max_val)
                elapsed = time.perf_counter() - t0
                xi1_err = abs(sol.first_zero - xi_ref) if sol.first_zero else float('inf')
                results[(n_val, h_test, 'radau')] = {
                    'steps': sol.num_steps, 'newton': sol.num_newton_iters,
                    'xi_1': sol.first_zero, 'error': xi1_err, 'time': elapsed,
                    'xi': sol.xi, 'theta': sol.theta,
                    'newton_per_step': sol.newton_per_step,
                }
                xi1_str = f"{sol.first_zero:.8f}" if sol.first_zero else "None"
                print(f"    h={h_test:.0e}: {sol.num_steps} steps, "
                      f"{sol.num_newton_iters} Newton, "
                      f"{elapsed:.4f}s, xi_1={xi1_str}, err={xi1_err:.2e}")
            except Exception as e:
                print(f"    h={h_test:.0e}: FAILED - {e}")
                results[(n_val, h_test, 'radau')] = None

        # RK4
        print("  RK4:")
        for h_test in h_values:
            try:
                t0 = time.perf_counter()
                sol = rk_mod.solve_lane_emden_rk4(n=n_val, epsilon=1e-4, h=h_test,
                                                   xi_max=xi_max_val)
                elapsed = time.perf_counter() - t0
                xi1_err = abs(sol.first_zero - xi_ref) if sol.first_zero else float('inf')
                results[(n_val, h_test, 'rk4')] = {
                    'steps': len(sol.xi) - 1, 'xi_1': sol.first_zero,
                    'error': xi1_err, 'time': elapsed,
                    'xi': sol.xi, 'theta': sol.theta,
                }
                xi1_str = f"{sol.first_zero:.8f}" if sol.first_zero else "None"
                print(f"    h={h_test:.0e}: {len(sol.xi) - 1} steps, "
                      f"{elapsed:.4f}s, xi_1={xi1_str}, err={xi1_err:.2e}")
            except Exception as e:
                print(f"    h={h_test:.0e}: FAILED - {e}")
                results[(n_val, h_test, 'rk4')] = None

    # ============================================================
    # Generate 2×2 overview figure
    # ============================================================
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # --- Panel (0,0): theta(xi) for n=4.5, Radau vs RK4 at multiple h ---
    ax = axes[0, 0]
    n_plot = 4.5
    h_plot_list = [5e-2, 2e-2, 1e-2]
    colors = {5e-2: '#2196F3', 2e-2: '#4CAF50', 1e-2: '#FF9800'}
    for h_test in h_plot_list:
        # Radau
        r = results.get((n_plot, h_test, 'radau'))
        if r is not None:
            ax.plot(r['xi'], r['theta'], color=colors[h_test], linewidth=1.5,
                    linestyle='-', label=f'Radau h={h_test:.0e}')
        # RK4
        r_rk = results.get((n_plot, h_test, 'rk4'))
        if r_rk is not None:
            ax.plot(r_rk['xi'], r_rk['theta'], color=colors[h_test], linewidth=1.0,
                    linestyle='--', alpha=0.7, label=f'RK4 h={h_test:.0e}')
    ax.set_xlabel(r'$\xi$', fontsize=11)
    ax.set_ylabel(r'$\theta(\xi)$', fontsize=11)
    ax.set_title(f'Density Profiles: Radau IIA vs RK4 (n={n_plot})', fontsize=12, fontweight='bold')
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)
    ax.annotate('Solid: Radau IIA (L-stable)\nDashed: RK4 (explicit)\n'
                'n=4.5 near-critical → stiff regime',
                xy=(0.97, 0.97), xycoords='axes fraction', fontsize=8,
                ha='right', va='top',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    # --- Panel (0,1): Error in xi_1 vs step size h (log-log) ---
    ax = axes[0, 1]
    markers = {0.0: 'o', 3.0: 's', 4.5: '^'}
    for n_val in n_values:
        for method, m_label, m_style in [('radau', 'Radau', '-'), ('rk4', 'RK4', '--')]:
            h_err = []
            err_vals = []
            for h_test in h_values:
                r = results.get((n_val, h_test, method))
                if r is not None and r['error'] < 1e10 and r['error'] > 0:
                    h_err.append(h_test)
                    err_vals.append(r['error'])
            if h_err:
                ax.loglog(h_err, err_vals, linestyle=m_style, marker=markers[n_val],
                          markersize=6, linewidth=1.2,
                          label=f'{m_label}, n={n_val:g}')
    # Reference O(h^4) and O(h^2) lines
    h_ref = np.array([1e-3, 1e-1])
    ax.loglog(h_ref, 1e-4 * (h_ref / 1e-2) ** 4, 'k:', linewidth=1, alpha=0.5, label=r'$O(h^4)$')
    ax.loglog(h_ref, 1e-2 * (h_ref / 1e-2) ** 2, 'k-.', linewidth=1, alpha=0.5, label=r'$O(h^2)$')
    ax.set_xlabel('Step size h', fontsize=11)
    ax.set_ylabel(r'Error in $\xi_1$', fontsize=11)
    ax.set_title(r'Error Convergence: Radau IIA vs RK4', fontsize=12, fontweight='bold')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3, which='both')

    # --- Panel (1,0): Newton iterations per step vs xi for n=4.5, h=2e-2 ---
    ax = axes[1, 0]
    r_n = results.get((4.5, 2e-2, 'radau'))
    if r_n is not None and r_n['newton_per_step'] is not None:
        nps = r_n['newton_per_step']
        xi_vals = r_n['xi'][1:len(nps)+1]  # xi after each step
        # Bar chart of iterations per step
        ax.bar(range(len(nps)), nps, width=0.8, color='steelblue', alpha=0.7, edgecolor='navy', linewidth=0.3)
        ax.set_xlabel('Step index', fontsize=11)
        ax.set_ylabel('Newton iterations', fontsize=11)
        ax.set_title(f'Newton Iterations per Step (n=4.5, h=2e-2)', fontsize=12, fontweight='bold')
        avg_iters = np.mean(nps)
        ax.axhline(y=avg_iters, color='red', linestyle='--', linewidth=1,
                   label=f'Average: {avg_iters:.1f} iters/step')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim(bottom=0)
        ax.annotate(f'Total steps: {len(nps)}\nTotal Newton iters: {sum(nps)}\n'
                    f'Max iters/step: {max(nps)}',
                    xy=(0.97, 0.97), xycoords='axes fraction', fontsize=8,
                    ha='right', va='top',
                    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    else:
        ax.text(0.5, 0.5, 'No per-step Newton data available', transform=ax.transAxes,
                ha='center', va='center', fontsize=12)
        ax.set_title('Newton Iterations per Step', fontsize=12, fontweight='bold')

    # --- Panel (1,1): Summary table ---
    ax = axes[1, 1]
    ax.axis('off')
    table_data = []
    for n_val in n_values:
        for h_test in [5e-2, 2e-2, 1e-2]:
            row = [f'n={n_val:g}, h={h_test:.0e}']
            for method, m_label in [('radau', 'Radau'), ('rk4', 'RK4')]:
                r = results.get((n_val, h_test, method))
                if r is not None:
                    row.append(f"{r['steps']}")
                    row.append(f"{r['xi_1']:.6f}" if r['xi_1'] else 'N/A')
                    row.append(f"{r['error']:.2e}")
                    row.append(f"{r['time']:.4f}s")
                else:
                    row.extend(['—', '—', '—', '—'])
            table_data.append(row)

    col_labels = ['Case', 'Steps\n(Radau)', 'ξ₁ (Radau)', 'Err (Radau)', 'Time',
                       'Steps\n(RK4)', 'ξ₁ (RK4)', 'Err (RK4)', 'Time']
    # Reshape data: 2 rows for Radau, 2 rows for RK4 per case
    flat_data = []
    for row in table_data:
        flat_data.append([row[0]] + row[1:5] + row[5:9])

    table = ax.table(cellText=flat_data, colLabels=col_labels,
                     cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(7)
    table.scale(0.95, 1.5)
    ax.set_title('Key Metrics: Radau IIA vs RK4 Comparison', fontsize=12, fontweight='bold',
                 pad=20)

    fig.suptitle('Implicit Radau IIA Method: Experimental Validation', fontsize=15, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(OUTPUT_DIR / 'radau_overview.png', dpi=200)
    plt.close()
    print(f"\nPlot saved to {OUTPUT_DIR / 'radau_overview.png'}")
