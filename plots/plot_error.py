"""
Plots the exact lorenz63 solution vs the approximated solution
using the propagator.
"""

from dynamical_systems import linear_system, lorenz_63, find_approximation, propagate_from_u0
import matplotlib.pyplot as plt
import numpy as np


def lorenz_63_plot():
    fig = plt.figure()
    ax3d = fig.add_subplot(1, 2, 1, projection='3d')
    ax2d = fig.add_subplot(1, 2, 2)
    U, W_LR, W_in, b_in, dt = find_approximation(lorenz_63, 30, N=5000, D_r=5000)
    U_hat = propagate_from_u0(U, W_LR, W_in, b_in)
    ax3d.plot(*U_hat, label=r"$\hat{U}$")
    ax3d.plot(*U, label="$U$")
    ax3d.legend()
    ax2d.semilogy(np.linspace(0, 30, 5001), np.apply_along_axis(np.linalg.norm, 0, (U_hat - U)))
    ax2d.set_xlabel("t")
    ax2d.set_ylabel("Log error")
    ax3d.set_title("Lorenz-63 System")
    fig.tight_layout(pad=3.0)
    plt.show()


if __name__ == "__main__":
    lorenz_63_plot()