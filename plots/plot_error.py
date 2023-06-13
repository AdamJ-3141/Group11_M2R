"""
Plots the exact lorenz63 solution vs the approximated solution
using the propagator.
"""

from dynamical_systems import linear_system, lorenz_63, find_approximation, propagate_from_u0
import matplotlib.pyplot as plt
import numpy as np
import math


def factor_int(n):
    val = math.ceil(math.sqrt(n))
    val2 = int(n/val)
    while val2 * val != float(n):
        val -= 1
        val2 = int(n/val)
    return val, val2


def plot_error(system: callable, T, N, D_r, noisy=False):
    fig = plt.figure()
    U_exact, U_obs, W_LR, W_in, b_in, dt = find_approximation(system, T, N=N, D_r=D_r, noisy=noisy)
    U_hat = propagate_from_u0(U_obs, W_LR, W_in, b_in)
    if U_obs.shape[0] == 3:
        ax3d = fig.add_subplot(1, 2, 1, projection='3d')
    else:
        ax3d = fig.add_subplot(1, 2, 1)
    ax3d.set_aspect("equal")
    ax2d = fig.add_subplot(1, 2, 2)
    ax3d.plot(*U_exact, "--", label="$U$", linewidth=0.4)
    ax3d.plot(*U_obs, label="$U_{obs}$", linewidth=0.6)
    ax3d.plot(*U_hat, label=r"$\hat{U}$")
    ax3d.legend()
    ax2d.semilogy(np.linspace(0, T, N+1), np.apply_along_axis(np.linalg.norm, 0, (U_hat - U_obs)))
    ax2d.set_xlabel("t")
    ax2d.set_ylabel("Log error")
    ax3d.set_title(("Noisy " if noisy else "")+r"Solution $U_{obs}$ and approximation $\hat{U}$")
    fig.tight_layout(pad=1.0)
    plt.show()


def range_Dr(system: callable, T, N, D_r: list[int], noisy=False):
    columns, rows = factor_int(len(D_r))
    fig, axs = plt.subplots(nrows=rows, ncols=columns, layout="constrained", subplot_kw={"projection": "3d"})
    for d, ax in zip(D_r, axs.flat):
        ax.set_aspect("equal")
        U, W_LR, W_in, b_in, dt = find_approximation(system, T, N=N, D_r=d, noisy=noisy)
        U_hat = propagate_from_u0(U, W_LR, W_in, b_in)
        ax.plot(*U, label="$U$", linewidth=0.3)
        ax.plot(*U_hat, label=r"$\hat{U}$", linewidth=0.5)
        ax.plot(-1, 1, "ro", label="$u_0$", markersize=3)
        plt.legend(bbox_to_anchor=(1, 0), loc="lower right",
                   bbox_transform=fig.transFigure, ncol=1)
        ax.set_title(f"$D_r = {d}$")
        ax.tick_params(axis="both", labelbottom=False, labelright=False, labelleft=False)
    plt.show()


if __name__ == "__main__":
    plot_error(lorenz_63, 15, 2000, 200, noisy=True)
