"""
Colour plot, varying b and w for different dynamical systems.
"""

from dynamical_systems import lorenz_63, linear_system, find_approximation, propagate_until_diverge
import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":
    b_size = 100
    b_max = 5
    w_size = 100
    w_max = 0.4
    T = 10
    N = 100
    D_r = 50
    T_f_matrix = np.empty((b_size+1, w_size+1))
    i = 0
    for b_ in range(b_size+1):
        b = b_ * (b_max/(b_size+1))
        for w_ in range(w_size+1):
            print(f"{i}/{(b_size+1) * (w_size+1)}", end="\r")
            i += 1
            w = w_ * (w_max/(w_size+1))
            U, W_LR, W_in, b_in, dt = find_approximation(lorenz_63, T, N=N, D_r=D_r, b=b, w=w)
            tau_f = propagate_until_diverge(U, T, W_LR, W_in, b_in)
            T_f_matrix[b_, w_] = tau_f
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    im = ax.imshow(T_f_matrix, origin="lower", extent=[0, w_max, 0, b_max], aspect=w_max/b_max, cmap="hot")
    cbar = plt.colorbar(im, ax=ax, cmap="hot")
    cbar.set_label("Colorbar")
    ax.set_xlabel("w")
    ax.set_ylabel("b")
    plt.show()
