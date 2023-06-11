"""
Colour plot, varying b and w for different dynamical systems.
"""

from dynamical_systems import lorenz_63, linear_system, find_approximation, propagate_until_diverge
import numpy as np
import matplotlib.pyplot as plt
import time
import datetime


if __name__ == "__main__":
    b_size = 300
    b_max = 4
    w_size = 300
    w_max = 0.4
    T = 400
    N = 20000
    D_r = 400
    noisy = False
    realisations = 10
    T_f_mean = np.empty((b_size+1, w_size+1))
    T_f_sd = np.empty((b_size + 1, w_size + 1))
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
                U, W_LR, W_in, b_in, dt = find_approximation(lorenz_63, T, N=N, D_r=D_r, b=b, w=w, noisy=noisy)
                tau_f = propagate_until_diverge(U, T, W_LR, W_in, b_in)
                tau_f_list.append(tau_f)
            T_f_mean[b_, w_] = sum(tau_f_list)/realisations
            T_f_sd[b_, w_] = np.std(tau_f_list)
    fig = plt.figure()
    ax_mean = fig.add_subplot(1, 2, 1)
    ax_std = fig.add_subplot(1, 2, 2)
    ax_mean.set_title(r"Mean of $\tau_f$.")
    im_mean = ax_mean.imshow(T_f_mean, origin="lower", extent=[0, w_max, 0, b_max], aspect=w_max/b_max, cmap="inferno")
    cbar = plt.colorbar(im_mean, ax=ax_mean, cmap="inferno", shrink=0.5)
    cbar.set_label(r"$\mu_{\tau_f}$")
    ax_mean.set_xlabel("$w$")
    ax_mean.set_ylabel("$b$")
    ax_std.set_title(r"Standard deviation of $\tau_f$.")
    im_std = ax_std.imshow(T_f_sd, origin="lower", extent=[0, w_max, 0, b_max], aspect=w_max / b_max, cmap="inferno")
    cbar = plt.colorbar(im_std, ax=ax_std, cmap="inferno", shrink=0.5)
    cbar.set_label(r"$\sigma_{\tau_f}$")
    ax_mean.set_xlabel("$w$")
    ax_mean.set_ylabel("$b$")
    fig.tight_layout(pad=2.0)
    plt.show()
    print(f"Elapsed Time: {datetime.timedelta(seconds=round(time.time()-start))}")
