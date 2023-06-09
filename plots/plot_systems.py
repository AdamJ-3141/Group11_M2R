from dynamical_systems import linear_system, lorenz_63
import numpy as np
import matplotlib.pyplot as plt


def linear_plot():
    t = np.linspace(0, np.pi, 100)
    U = linear_system(t, 1, 1)

    plt.plot(*U)
    plt.plot(-1, 1, "ro")
    plt.annotate("$u(0)$", xy=(-1, 1), xytext=(-1, 0.85))
    plt.gca().set_aspect("equal")
    plt.gca().set_xlabel("$x$")
    plt.gca().set_ylabel("$y$")
    plt.title("2D Linear System: Half-Circle")
    plt.show()


def lorenz_63_plot():
    t = np.linspace(0, 40, 5000)
    U = lorenz_63(t, False, 0)
    fig = plt.figure()
    ax3d = fig.add_subplot(1, 1, 1, projection='3d')
    ax3d.plot(*U)
    ax3d.plot(1, 1, 1, "ro")
    ax3d.text(1, -6, 1, "$u_0$")
    ax3d.set_aspect("equal")
    plt.title("3D Nonlinear System: Lorenz-63")
    plt.show()


lorenz_63_plot()
