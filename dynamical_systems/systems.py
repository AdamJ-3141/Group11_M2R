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
from numpy import sin, cos
from scipy.integrate import odeint


def lorenz_63_system(u, t):
    r = 28
    s = 10
    b = 8/3
    x, y, z = u
    x_dot = s * (y - x)
    y_dot = r * x - y - x * z
    z_dot = x * y - b * z
    return np.array([x_dot, y_dot, z_dot])


def lorenz_63(t):
    u0 = np.array([0., 1., 1.05])
    sol: np.ndarray = odeint(lorenz_63_system, u0, t)
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
