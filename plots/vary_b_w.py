"""
Colour plot, varying b and w for different dynamical systems.
"""

from dynamical_systems import lorenz_63, linear_system, find_approximation, propagate_until_diverge
import numpy as np
import matplotlib.pyplot as plt
import time
import datetime


if __name__ == "__main__":
    b_size = 10
    b_max = 4
    w_size = 10
    w_max = 0.5
    T = 20
    N = 2000
    D_r = 1500
    realisations = 1
    T_f_matrix = np.empty((b_size+1, w_size+1))
    i = 0
    start = time.time()
    for b_ in range(b_size+1):
        b = b_ * (b_max/(b_size+1))
        for w_ in range(w_size+1):
            tau_f_list = []
            i += 1
            elapsed = time.time() - start
            p = i/((b_size+1) * (w_size+1))
            secs = round(elapsed/p - elapsed)
            print(f"{round(100*p, 2)}%  -  Estimated Time:"
                  f" {datetime.timedelta(seconds=secs)}", end="\r")
            w = w_ * (w_max/(w_size+1))
            for _ in range(realisations):
                U, W_LR, W_in, b_in, dt = find_approximation(lorenz_63, T, N=N, D_r=D_r, b=b, w=w)
                tau_f = propagate_until_diverge(U, T, W_LR, W_in, b_in)
                tau_f_list.append(tau_f)
            T_f_matrix[b_, w_] = sum(tau_f_list)/realisations
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title(r"A plot of forecast time $\tau_f$ for different $w$ and $b$.")
    im = ax.imshow(T_f_matrix, origin="lower", extent=[0, w_max, 0, b_max], aspect=w_max/b_max, cmap="inferno")
    cbar = plt.colorbar(im, ax=ax, cmap="inferno")
    cbar.set_label(r"$\tau_f$")
    ax.set_xlabel("$w$")
    ax.set_ylabel("$b$")
    plt.show()
    plt.savefig(f"plot_images/w-b_{w_size}x{b_size}_N{N}Dr{D_r}.png")
