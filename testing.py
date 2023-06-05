import numpy as np
from dynamical_systems import lorenz_63, find_approximation, propagate_until_diverge, propagate_from_u0
import matplotlib.pyplot as plt

if __name__ == "__main__":
    b = 4
    w = 0.005
    N = 1000
    D_r = 4000
    t1 = 15
    U, W_LR, W_in, b_in, dt = find_approximation(lorenz_63, t1, N=N, D_r=D_r, b=b, w=w)
    U_hat = propagate_from_u0(U, W_LR, W_in, b_in)
    tau_f = propagate_until_diverge(U, t1, W_LR, W_in, b_in)
    fig = plt.figure()
    ax3d = fig.add_subplot(1, 2, 1, projection='3d')
    ax2d = fig.add_subplot(1, 2, 2)
    ax3d.plot(*U_hat, label=r"$\hat{U}$")
    ax3d.plot(*U, label="$U$")
    ax3d.legend()
    ax2d.semilogy(np.linspace(0, t1, N+1),
                  np.apply_along_axis(np.linalg.norm, 0, (U_hat - U))/np.apply_along_axis(np.linalg.norm, 0, U))
    ax2d.set_xlabel("t")
    ax2d.set_ylabel("Log error")
    ax3d.set_title("Lorenz-63 System")
    fig.tight_layout(pad=3.0)
    ax2d.hlines(0.05, 0, t1, "r")
    plt.show()
    print(tau_f)