import numpy as np
import h5py
import matplotlib.pyplot as plt

plt.ion()

file_name = "xmission.h5"

with h5py.File(file_name, 'r') as f:
    t_a = f['dir/t_axis']
    dt = t_a.attrs['dt']
    t_a = t_a[()]
    xmitt = f['dir/pulse'][()]

f_a = np.arange(xmitt.size // 2 + 1) / (xmitt.size * dt)
xmitt_dB = 20 * np.log10(np.abs(np.fft.rfft(xmitt)))
xmitt_dB -= xmitt_dB.max()

fig, axes = plt.subplots(2, 1)
ax = axes[0]
ax.plot(t_a, xmitt)

ax = axes[1]
ax.plot(f_a, xmitt_dB)
ax.set_ylim(-100, 3)
