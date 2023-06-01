"""
Algorithm
Input: Time series u_n, n=1,2,...,N
Parameters:
    Regularization parameter b
    Reservoir Dimension D_r
    Internal parameters W_in, B_in

Construct observation matrix U = [u_1, u_2, u_3, ..., u_n]
Construct random features Phi_n = tanh(W_in @ U_nm1 + b_in)
Construct feature matrix Phi = [Phi_1, ..., Phi_N]

Output:
    W_LR = U @ Phi.t @ (Phi @ Phi.t + bI)^-1

"""

import numpy as np
import matplotlib.pyplot as plt
from numpy import sin, cos
from numpy.linalg import inv
from scipy.integrate import odeint
np.random.seed(8008)


def get_lorenz_vals():
    def lorenz(xyz, *, s=10, r=28, b=2.667):
        """
        Parameters
        ----------
        xyz : array-like, shape (3,)
           Point of interest in three-dimensional space.
        s, r, b : float
           Parameters defining the Lorenz attractor.

        Returns
        -------
        xyz_dot : array, shape (3,)
           Values of the Lorenz attractor's partial derivatives at *xyz*.
        """
        x, y, z = xyz
        x_dot = s*(y - x)
        y_dot = r*x - y - x*z
        z_dot = x*y - b*z
        return np.array([x_dot, y_dot, z_dot])

    dt = 0.01
    num_steps = 1000

    xyzs = np.empty((num_steps + 1, 3))
    xyzs[0] = (0., 1., 1.05)
    for i in range(num_steps):
        xyzs[i + 1] = xyzs[i] + lorenz(xyzs[i]) * dt

    # ax = plt.figure().add_subplot(projection='3d')
    # ax.plot(*xyzs.T, lw=0.5)
    # ax.set_xlabel("X Axis")
    # ax.set_ylabel("Y Axis")
    # ax.set_zlabel("Z Axis")
    # ax.set_title("Lorenz Attractor")
    # plt.show()
    return xyzs.T


def lorenz_63(u, t):
    r = 28
    s = 10
    b = 8/3
    x, y, z = u
    x_dot = s * (y - x)
    y_dot = r * x - y - x * z
    z_dot = x * y - b * z
    return np.array([x_dot, y_dot, z_dot])


def lorenz(t):
    u0 = np.array([0., 1., 1.05])
    sol: np.ndarray = odeint(lorenz_63, u0, t)
    return sol[:, :3].transpose()


def linear_system(t, c1=1, c2=1):

    """
    Return values for the system

    dx/dt = (0   1) (x)
    dy/dt   (-1  0) (y)
    """
    x = c1 * sin(t) - c2 * cos(t)
    y = c1 * cos(t) + c2 * sin(t)
    return np.array([x, y])


def find_approximation(system: callable, t0: float, t1: float,
                       N=100, D_r=50, w=0.005, b=4, beta=1E-5):
    U: np.ndarray = system(np.linspace(t0, t1, N+1))
    D = U.shape[0]
    U_o = U[:, 1:]
    U_i = U[:, :-1]
    W_in = w * (2 * np.random.random((D_r, D)) - 1)
    b_in = b * (2 * np.random.random((D_r, 1)) - 1)
    Phi = np.tanh(W_in @ U_i + b_in)
    W_LR = (U_o @ Phi.T @ inv(Phi @ Phi.T + beta * np.identity(D_r)))

    U_hat = np.atleast_2d(U[:, 0]).T
    for _ in range(N):
        u_n = U_hat[:, -1]
        phi = np.tanh(np.atleast_2d(W_in @ u_n).T + b_in)
        u_np1 = W_LR @ phi
        U_hat = np.concatenate((U_hat, u_np1), axis=1)
    return U, U_hat


# D_r_vals = [1, 10, 100, 1000, 10000]
# N = 300
fig = plt.figure()
ax3d = fig.add_subplot(1, 2, 1, projection='3d')
ax2d = fig.add_subplot(1, 2, 2)
U, U_hat = find_approximation(lorenz, 0, 5, N=1000, D_r=5000)
ax3d.plot(*U_hat)
ax3d.plot(*U)
ax2d.plot(np.log10(np.apply_along_axis(np.linalg.norm, 0, (U_hat - U))))
#
# for dr_ind, D_r in enumerate(D_r_vals):
#     U, U_hat = find_approximation(linear_system, 0, 2*np.pi, N=N, D_r=D_r)
#     ax.plot(np.log10(np.apply_along_axis(np.linalg.norm, 0, (U_hat - U))), label=f"D_r = {D_r}")
#
# ax.legend()
# for D_r in range(1, 1000, 20):
#     U, U_hat = find_approximation(get_lorenz_vals, 0, 2*np.pi, N=300, D_r=D_r)
#     if D_r == 1: ax.plot(*U)
#     ax.plot(*U_hat)

# axs[0].plot(*U)
# axs[0].plot(*U_hat)
# axs[0].set_box_aspect(1)
#
# norm_error = np.apply_along_axis(np.linalg.norm, 0, (U_hat - U))
# print(norm_error)
#
# axs[1].plot(norm_error)
plt.show()

