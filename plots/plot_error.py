from dynamical_systems import linear_system, lorenz, find_approximation
import matplotlib.pyplot as plt
import numpy as np


def lorenz_63_plot():
    fig = plt.figure()
    ax3d = fig.add_subplot(1, 2, 1, projection='3d')
    ax2d = fig.add_subplot(1, 2, 2)
    U, U_hat = find_approximation(lorenz, 0, 30, N=3000, D_r=5000)
    ax3d.plot(*U_hat, label=r"$\hat{U}$")
    ax3d.plot(*U, label="z$U$")
    ax3d.legend()
    ax2d.semilogy(np.linspace(0, 30, 3001), np.apply_along_axis(np.linalg.norm, 0, (U_hat - U)))
    ax2d.set_xlabel("t")
    ax2d.set_ylabel("Log error")
    ax3d.set_title("Lorenz-63 System")
    fig.tight_layout(pad=3.0)


D_r_vals = [1, 10, 100, 1000, 10000]
N = 300


for dr_ind, D_r in enumerate(D_r_vals):
    U, U_hat = find_approximation(linear_system, 0, 2*np.pi, N=N, D_r=D_r)
    ax.plot(np.log10(np.apply_along_axis(np.linalg.norm, 0, (U_hat - U))), label=f"D_r = {D_r}")

ax.legend()