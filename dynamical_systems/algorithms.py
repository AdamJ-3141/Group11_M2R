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

from numpy.linalg import inv
import numpy as np
np.random.seed(42493874)


def find_approximation(system: callable, t1: float,
                       N=1000, D_r=400, w=0.1, b=4, beta=1E-5,
                       noisy=False, eps=0.2):
    dt = t1/(N+1)
    U: np.ndarray = system(np.arange(0, t1, dt), noisy, eps)
    D = U.shape[0]
    U_o = U[:, 1:]
    U_i = U[:, :-1]
    W_in = w * (2 * np.random.random((D_r, D)) - 1)
    b_in = b * (2 * np.random.random((D_r, 1)) - 1)
    Phi = (1/np.sqrt(D_r))*np.tanh(W_in @ U_i + b_in)
    W_LR = (U_o @ Phi.T @ inv(Phi @ Phi.T + beta * np.identity(D_r)))

    return U, W_LR, W_in, b_in, dt


def propagate_from_u0(U, W_LR, W_in, b_in):
    N = U.shape[1] - 1
    D_r = b_in.shape[0]
    U_hat = np.atleast_2d(U[:, 0]).T
    for n in range(N):
        u_hat_n = U_hat[:, -1]
        phi = (1/np.sqrt(D_r))*np.tanh(np.atleast_2d(W_in @ u_hat_n).T + b_in)
        u_np1 = W_LR @ phi
        U_hat = np.concatenate((U_hat, u_np1), axis=1)
    return U_hat


def propagate_until_diverge(U, t1, W_LR, W_in, b_in):
    N = U.shape[1] - 1
    dt = t1 / (N + 1)
    D_r = b_in.shape[0]
    U_hat = np.atleast_2d(U[:, 0]).T
    tau_f = t1
    for n in range(N):
        u_hat_n = U_hat[:, -1]
        u_n = U[:, n]
        phi = (1/np.sqrt(D_r))*np.tanh(np.atleast_2d(W_in @ u_hat_n).T + b_in)
        u_np1 = W_LR @ phi
        rel_error = np.linalg.norm(u_hat_n - u_n)/np.linalg.norm(u_n)
        if rel_error > 0.05:
            tau_f = n * dt
            return tau_f
        U_hat = np.concatenate((U_hat, u_np1), axis=1)
    return tau_f


def estimate_next(U, U_hat, W_LR, W_in, b_in):
    u_n = U[:, -1]
    u_hat_n = U_hat[:, -1]
    u_np1_from_exact = W_LR @ np.tanh(np.atleast_2d(W_in @ u_n).T + b_in)
    u_np1_from_prop = W_LR @ np.tanh(np.atleast_2d(W_in @ u_hat_n).T + b_in)
    return
