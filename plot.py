import numpy as np
import matplotlib.pyplot as plt

filename = ''

with open(filename, 'rb') as f:
    Tau = np.load(f)

M = np.apply_along_axis(np.mean, 2, Tau)
V = np.apply_along_axis(np.var, 2, Tau)

# plotting
fig, ax = plt.subplots(1, 2)
im = ax[0].imshow(M, origin='lower',
                  extent=[0, 0.02, 0, 10000], aspect=10000/0.02)
plt.colorbar(im, ax=ax[0])
im = ax[1].imshow(V, origin='lower',
                  extent=[0, 0.02, 0, 10000], aspect=10000/0.02)
plt.colorbar(im, ax=ax[1])

plt.show()
