"""
Plots the exact lorenz63 solution vs the approximated solution
using the propagator.
"""

from dynamical_systems import linear_system, lorenz_63, \
    find_approximation, propagate_from_u0, propagate_from_point
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


def plot_error(system: callable, T, N, D_r, noisy=False, eps=0.2):
    U_exact, U_obs, W_LR, W_in, b_in, dt = find_approximation(system, T, N=N, D_r=D_r, noisy=noisy, eps=eps)
    U_hat = propagate_from_u0(U_obs, W_LR, W_in, b_in)
    fig = plt.figure()
    if U_obs.shape[0] == 3:
        ax3d = fig.add_subplot(1, 2, 1, projection='3d')
    else:
        ax3d = fig.add_subplot(1, 2, 1)
    ax3d.set_aspect("equal")
    ax2d = fig.add_subplot(1, 2, 2)
    ax3d.plot(*U_exact, "--", label="$U$", linewidth=0.6)
    ax3d.plot(*U_obs, label="$U_{obs}$", linewidth=0.4)
    ax3d.plot(*U_hat, label=r"$\hat{U}$")
    ax3d.legend()
    ax2d.plot(np.linspace(0, T, N+1), np.apply_along_axis(np.linalg.norm, 0, (U_hat - U_exact)))
    ax2d.set_xlabel("t")
    ax2d.set_ylabel("Absolute error")
    ax3d.set_title(("Noisy " if noisy else "")+r"Solution $U_{obs}$ and approximation $\hat{U}$", fontsize=10)
    fig.tight_layout(pad=1.0)
    plt.show()


def plot_future(system: callable, T, N, D_r, noisy=False, eps=0.2):
    U_exact, U_obs, W_LR, W_in, b_in, dt = find_approximation(system, T, N=N, D_r=D_r, noisy=noisy, eps=eps)
    u_0 = U_obs[:, 0]
    u_n = U_obs[:, -1]
    U_hat_from_start = propagate_from_point(u_0, 2*N, W_LR, W_in, b_in)
    U_hat_from_end = np.concatenate((U_obs[:, :-1], propagate_from_point(u_n, N, W_LR, W_in, b_in)), axis=1)
    U_exact_full = system(np.linspace(0, 2*T, 2*N + 1), noisy=noisy, eps=eps)[0]
    fig = plt.figure()
    plt.rcParams['figure.constrained_layout.use'] = True
    if U_obs.shape[0] == 3:
        ax_system = fig.add_subplot(1, 2, 1, projection='3d')
    else:
        ax_system = fig.add_subplot(1, 2, 1)
    ax_error = fig.add_subplot(1, 2, 2)
    ax_system.set_aspect("equal")
    ax_system.plot(*U_hat_from_start, label=r"$\hat{U}_0$")
    ax_system.plot(*U_hat_from_end, label=r"$\hat{U}_n$")
    ax_system.plot(*U_exact_full, "--", label=r"$U$", linewidth=0.5)
    ax_system.legend()
    ax_error.set_xlabel("t")
    ax_error.set_ylabel("Absolute Error")
    start_error = np.apply_along_axis(np.linalg.norm, 0, (U_hat_from_start - U_exact_full))
    end_error = np.apply_along_axis(np.linalg.norm, 0, (U_hat_from_end - U_exact_full))
    ax_error.plot(np.linspace(0, 2*T, 2*N+1), start_error, label="From $u_0$")
    ax_error.plot(np.linspace(0, 2 * T, 2 * N + 1), end_error, label="From $u_n$")
    ax_error.vlines(T, ymin=0, ymax=1.05*max(max(start_error), max(end_error)),
                    linestyles="dashed", colors="r", label=r"$t=N \Delta t$")
    ax_error.legend()
    fig.tight_layout(pad=2.0)
    fig.suptitle(f"Predicting $N={N}$ further data points with $D_r$={D_r}")
    plt.show()
    pass


def range_Dr(system: callable, T, N, D_r: list[int], noisy=False):
    columns, rows = factor_int(len(D_r))
    fig, axs = plt.subplots(nrows=rows, ncols=columns, layout="constrained")  # , subplot_kw={"projection": "3d"}
    for d, ax in zip(D_r, axs.flat):
        ax.set_aspect("equal")
        U_exact, U, W_LR, W_in, b_in, dt = find_approximation(system, T, N=N, D_r=d, noisy=noisy, w=0.1)
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
    # plot_error(linear_system, np.pi, 50, 15, noisy=False, eps=0.1)
    plot_future(lorenz_63, 15, 2000, 500, noisy=True, eps=0.2)
