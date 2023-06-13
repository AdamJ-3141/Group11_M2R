"""
Functions for the lorenz system and a linear system.
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


def lorenz_63(t, noisy, eps):
    u0 = np.array([1., 1., 1.])
    sol: np.ndarray = odeint(lorenz_63_system, u0, t)
    U = sol[:, :3].transpose()
    if noisy:
        noise_matrix = eps*np.random.normal(size=U.shape)
    else:
        noise_matrix = np.zeros(U.shape)
    U_obs = U.copy()
    U_obs += noise_matrix
    return U, U_obs


def linear_system(t, noisy, eps, c1=1, c2=1):
    """
    Return values for the system

    dx/dt = (0   1) (x)
    dy/dt   (-1  0) (y)
    """
    x = c1 * sin(t) - c2 * cos(t)
    y = c1 * cos(t) + c2 * sin(t)
    U = np.array([x, y])
    if noisy:
        noise_matrix = eps*np.random.normal(size=U.shape)
    else:
        noise_matrix = np.zeros(U.shape)
    U_obs = U.copy()
    U_obs += noise_matrix
    return U, U_obs
