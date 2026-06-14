from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from data_input import load_reference_data


@dataclass(frozen=True)
class SpectralSolution:
    """Solution of Lane-Emden equation using Chebyshev spectral collocation."""
    n: float
    xi: np.ndarray           # physical grid points
    theta: np.ndarray        # solution at grid points
    theta_prime: np.ndarray  # derivative at grid points
    first_zero: Optional[float]
    converged: bool
    iterations: int
    residual_norm: float


def chebyshev_grid(N: int) -> np.ndarray:
    """Chebyshev-Gauss-Lobatto nodes on [-1, 1]."""
    return -np.cos(np.pi * np.arange(N + 1) / N)


def chebyshev_differentiation_matrix(x: np.ndarray) -> np.ndarray:
    """Compute the Chebyshev differentiation matrix.

    Returns D such that f' = D @ f at the collocation points.
    """
    N = len(x) - 1
    if N == 0:
        return np.zeros((1, 1))

    c = np.ones(N + 1)
    c[0] = 2.0
    c[-1] = 2.0

    D = np.zeros((N + 1, N + 1))
    for i in range(N + 1):
        for j in range(N + 1):
            if i != j:
                D[i, j] = (c[i] / c[j]) * (-1.0) ** (i + j) / (x[i] - x[j])

    # Diagonal entries
    for i in range(N + 1):
        D[i, i] = -np.sum(D[i, :])

    return D


def map_to_physical(x_cheb: np.ndarray, a: float, b: float) -> np.ndarray:
    """Map Chebyshev nodes from [-1, 1] to [a, b]."""
    return a + (b - a) * (x_cheb + 1.0) / 2.0


def solve_spectral_lane_emden(
    n: float,
    N: int = 40,
    epsilon: float = 0.0,
    xi_max: Optional[float] = None,
    max_iterations: int = 30,
    tolerance: float = 1e-12,
    damping_initial: float = 0.5,
) -> SpectralSolution:
    """Solve Lane-Emden using Chebyshev spectral collocation + Newton.

    Uses an even extension trick: since theta(xi) is even about xi=0,
    we solve on [-xi_max, xi_max] with even symmetry. This avoids the
    singularity at xi=0 cleanly.

    Parameters
    ----------
    n:
        Polytropic index.
    N:
        Number of Chebyshev nodes (polynomial degree).
    epsilon:
        Not used (spectral method naturally handles xi=0).
    xi_max:
        Domain half-width. If None, uses reference xi_1.
    max_iterations:
        Maximum Newton iterations.
    tolerance:
        Convergence tolerance.

    Returns
    -------
    SpectralSolution.
    """
    if xi_max is None:
        ref = load_reference_data()
        prop = ref.get_global_property(n)
        if prop is not None and prop.xi_1 is not None:
            xi_max = prop.xi_1
        else:
            xi_max = 5.0

    # Chebyshev grid on [-1, 1] mapped to [-xi_max, xi_max]
    x_cheb = chebyshev_grid(N)
    xi_full = map_to_physical(x_cheb, -xi_max, xi_max)

    # Initial guess: n=5 shape (smooth, positive)
    theta_guess = 1.0 / np.sqrt(1.0 + xi_full ** 2 / 3.0)

    # Enforce BC: theta(0) = 1 (closest point to 0)
    idx_zero = np.argmin(np.abs(xi_full))
    theta_guess[idx_zero] = 1.0

    # Newton iteration
    D = chebyshev_differentiation_matrix(x_cheb) * 2.0 / (2.0 * xi_max)
    D2 = D @ D

    converged = False
    iterations = 0
    theta = theta_guess.copy()

    for iterations in range(1, max_iterations + 1):
        # Build residual F(theta) = theta'' + (2/xi)*theta' + theta^n = 0
        th_p = D @ theta
        th_pp = D2 @ theta

        # Handle xi=0 singularity: use l'Hopital: (2/xi)*theta' -> 2*theta''
        coeff = np.full(N + 1, 2.0 / xi_full)
        coeff[idx_zero] = 0.0  # at xi=0, the term 2*theta'/xi -> 0 (by evenness)

        residual = th_pp + coeff * th_p + theta ** n

        # BC at xi=0: theta(0) = 1
        residual[idx_zero] = theta[idx_zero] - 1.0

        res_norm = float(np.linalg.norm(residual, ord=np.inf))
        if res_norm < tolerance:
            converged = True
            break

        # Jacobian
        J = D2.copy()
        for i in range(N + 1):
            J[i, :] = 0.0
            J[i, i] = D2[i, i] + coeff[i] * D[i, i] + n * theta[i] ** (n - 1.0)
            for j in range(N + 1):
                J[i, j] = D2[i, j] + coeff[i] * D[i, j]
            J[i, i] += n * theta[i] ** (n - 1.0)

        # BC row
        J[idx_zero, :] = 0.0
        J[idx_zero, idx_zero] = 1.0
        residual[idx_zero] = theta[idx_zero] - 1.0

        # Solve
        try:
            delta = np.linalg.solve(J, -residual)
        except np.linalg.LinAlgError:
            delta = -residual / (np.diag(J) + 1e-10)

        # Damped Newton
        damping = damping_initial
        for _ in range(20):
            candidate = theta + damping * delta
            # Check positivity
            if np.all(candidate > 0):
                break
            damping *= 0.5

        theta = candidate

    # Extract positive half
    mask = xi_full >= -1e-12
    xi_phys = xi_full[mask]
    theta_phys = theta[mask]
    theta_prime_phys = (D @ theta)[mask]

    # Find first zero
    first_zero = None
    for i in range(1, len(theta_phys)):
        if theta_phys[i] <= 0:
            if theta_phys[i - 1] > 0:
                w = abs(theta_phys[i - 1]) / (abs(theta_phys[i - 1]) + abs(theta_phys[i]))
                first_zero = float(xi_phys[i - 1] + w * (xi_phys[i] - xi_phys[i - 1]))
            break

    return SpectralSolution(
        n=n, xi=xi_phys, theta=theta_phys, theta_prime=theta_prime_phys,
        first_zero=first_zero, converged=converged,
        iterations=iterations, residual_norm=res_norm,
    )


if __name__ == "__main__":
    import math

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from pathlib import Path

    print("=" * 70)
    print("Chebyshev Spectral Method for Lane-Emden Equation")
    print("=" * 70)

    OUTPUT_DIR = Path(__file__).resolve().parent / "output_initial"
    OUTPUT_DIR.mkdir(exist_ok=True)

    # Collect data for plot
    N_values = [10, 15, 20, 25, 30, 35, 40]
    plot_data = {}

    for n_val in (0.0, 1.0, 3.0):
        print(f"\n--- n = {n_val:g} ---")

        if abs(n_val) < 1e-14:
            xi_max_val = math.sqrt(6.0)
            exact = lambda xi: 1.0 - xi**2 / 6.0
        elif abs(n_val - 1.0) < 1e-14:
            xi_max_val = math.pi
            def exact(xi):
                r = np.ones_like(xi)
                m = xi != 0
                r[m] = np.sin(xi[m]) / xi[m]
                return r
        else:
            xi_max_val = load_reference_data().get_first_zero(n_val) or 7.0
            exact = None

        errors = []
        Ns_ok = []
        for N in N_values:
            try:
                sol = solve_spectral_lane_emden(n=n_val, N=N, xi_max=xi_max_val)
                if exact is not None:
                    err = np.max(np.abs(sol.theta - exact(sol.xi)))
                    print(f"  N={N:3d}: error={err:.2e}, xi_1={sol.first_zero}, "
                          f"iter={sol.iterations}, conv={sol.converged}")
                    errors.append(err)
                    Ns_ok.append(N)
                else:
                    print(f"  N={N:3d}: xi_1={sol.first_zero:.8f}, "
                          f"iter={sol.iterations}, conv={sol.converged}")
            except Exception as e:
                print(f"  N={N:3d}: FAILED - {str(e)[:60]}")
        plot_data[n_val] = (Ns_ok, errors)

    # Spectral convergence plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    for n_val, (Ns, errors) in plot_data.items():
        if errors:
            ax1.semilogy(Ns, errors, "o-", linewidth=1.5, label=f"n={n_val:g}")

    # Add reference O(N^(-N)) line for comparison
    N_ref = np.array([10, 40])
    ax1.semilogy(N_ref, np.exp(-0.3 * N_ref), "k:", linewidth=1, alpha=0.5, label=r"$\sim e^{-cN}$")
    ax1.set_xlabel("N (Chebyshev nodes)")
    ax1.set_ylabel(r"$\|\theta - \theta_{exact}\|_\infty$")
    ax1.set_title("Spectral Convergence (exact cases)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Solution profiles
    for n_val in (0.0, 1.5, 3.0):
        sol = solve_spectral_lane_emden(n=n_val, N=35,
                                        xi_max=load_reference_data().get_first_zero(n_val) or 7.0)
        ax2.plot(sol.xi, sol.theta, linewidth=1.5, label=f"n={n_val:g}")
    ax2.set_xlabel(r"$\xi$")
    ax2.set_ylabel(r"$\theta(\xi)$")
    ax2.set_title("Spectral Solutions")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.suptitle("Chebyshev Spectral Method", fontsize=13)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "spectral_convergence.png", dpi=200)
    plt.close()
    print(f"\nPlot saved to {OUTPUT_DIR / 'spectral_convergence.png'}")
