from numpy.linalg import inv
import numpy as np
np.random.seed(42493874)


def find_approximation(system: callable, t1: float,
                       N=100, D_r=50, w=0.005, b=4, beta=1E-5):
    dt = t1/(N+1)
    U: np.ndarray = system(np.arange(0, t1, dt))
    D = U.shape[0]
    U_o = U[:, 1:]
    U_i = U[:, :-1]
    W_in = w * (2 * np.random.random((D_r, D)) - 1)
    b_in = b * (2 * np.random.random((D_r, 1)) - 1)
    Phi = np.tanh(W_in @ U_i + b_in)
    W_LR = (U_o @ Phi.T @ inv(Phi @ Phi.T + beta * np.identity(D_r)))

    return U, W_LR, W_in, b_in, dt


def propagate_from_u0(U, W_LR, W_in, b_in):
    N = U.shape[1] - 1
    U_hat = np.atleast_2d(U[:, 0]).T
    for n in range(N):
        u_hat_n = U_hat[:, -1]
        phi = np.tanh(np.atleast_2d(W_in @ u_hat_n).T + b_in)
        u_np1 = W_LR @ phi
        U_hat = np.concatenate((U_hat, u_np1), axis=1)
    return U_hat


def propagate_until_diverge(U, t1, W_LR, W_in, b_in):
    N = U.shape[1] - 1
    dt = t1 / (N + 1)
    U_hat = np.atleast_2d(U[:, 0]).T
    tau_f = np.inf
    for n in range(N):
        u_hat_n = U_hat[:, -1]
        u_n = U[:, n]
        phi = np.tanh(np.atleast_2d(W_in @ u_hat_n).T + b_in)
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
