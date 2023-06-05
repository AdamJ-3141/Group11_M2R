"""
Colour plot, varying b and w for different dynamical systems.
"""

from dynamical_systems import lorenz_63, find_approximation, propagate_until_diverge
import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":
    T_f_matrix = np.empty((30, 30))
    for b_ in range(0, 30):
        b = b_ / 6
        for w_ in range(0, 30):
            w = w_ / 3000
            U, W_LR, W_in, b_in, dt = find_approximation(lorenz_63, 20, N=400, D_r=200, b=b, w=w)
            tau_f = propagate_until_diverge(U, 20, W_LR, W_in, b_in)
            T_f_matrix[b_, w_] = tau_f
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(np.log(T_f_matrix), origin="lower", extent=[0, 29/3000, 0, 29/6], aspect=0.001)
    ax.set_xlabel("w")
    ax.set_ylabel("b")
    plt.show()
