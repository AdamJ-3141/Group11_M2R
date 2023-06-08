import numpy as np
import matplotlib.pyplot as plt

filename = 'data/tau.npy'

with open(filename, 'rb') as f:
    Tau = np.load(f)


M = np.apply_along_axis(np.mean, 2, Tau)
V = np.apply_along_axis(np.var, 2, Tau)

# plotting
fig, ax = plt.subplots(1, 2)
im = ax[0].imshow(M, origin='lower',
                  extent=[0, 0.02, 0, 10000], aspect=0.02/10000)
plt.colorbar(im, ax=ax[0])
im = ax[1].imshow(V, origin='lower',
                  extent=[0, 0.02, 0, 10000], aspect=0.02/10000)
plt.colorbar(im, ax=ax[1])
fig.tight_layout(pad=3.0)

plt.show()
