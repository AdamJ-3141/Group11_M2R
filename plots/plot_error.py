"""
Plots the exact lorenz63 solution vs the approximated solution
using the propagator.
"""

from dynamical_systems import linear_system, lorenz_63, find_approximation, propagate_from_u0
import matplotlib.pyplot as plt
import numpy as np


def plot_error(system: callable, T, N, D_r):
    fig = plt.figure()
    U, W_LR, W_in, b_in, dt = find_approximation(system, T, N=N, D_r=D_r, w=0.5, b=4, noisy=False)
    U_hat = propagate_from_u0(U, W_LR, W_in, b_in)
    if U.shape[0] == 3:
        ax3d = fig.add_subplot(1, 2, 1, projection='3d')
    else:
        ax3d = fig.add_subplot(1, 2, 1)
    ax3d.set_aspect("equal")
    ax2d = fig.add_subplot(1, 2, 2)
    ax3d.plot(*U, label="$U$", linewidth=0.5)
    ax3d.plot(*U_hat, label=r"$\hat{U}$")
    ax3d.legend()
    ax2d.semilogy(np.linspace(0, T, N+1), np.apply_along_axis(np.linalg.norm, 0, (U_hat - U)))
    ax2d.set_xlabel("t")
    ax2d.set_ylabel("Log error")
    ax3d.set_title(r"Solution $U$ and approximation $\hat{U}$")
    fig.tight_layout(pad=1.0)
    plt.show()


if __name__ == "__main__":
    plot_error(lorenz_63, 20, 2000, 200)
