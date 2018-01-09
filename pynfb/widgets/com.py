import numpy as np
from scipy.signal import welch
import pylab as plt
import h5py
from pynfb.postprocessing.utils import get_info
from mne.viz import plot_topomap
from pynfb.inlets.montage import Montage
from pynfb.signal_processing.filters import ButterFilter

file = r'C:\Users\Nikolai\Desktop\bci_nfb_bci\bci_nfb_bci\bci_mu_ica_S4_d3_08-25_15-02-43\experiment_data.h5'
with h5py.File(file) as f:
    fs, channels, protocol_names = get_info(f, [])
    x = np.concatenate([f['protocol{}/raw_data'.format(k + 1)][:] for k in range(12)])

montage = Montage(channels)

ica = -np.load('spat-ica.npy')
ssd = np.load('spat-ssd.npy')
csp = -np.load('spat-csp.npy')


b_filter = ButterFilter((10, 16), fs, len(channels))
ica_data = b_filter.apply(x)
ssd_t = np.dot(np.dot(ica_data.T, ica_data), ssd)

b_filter = ButterFilter((3, 45), fs, len(channels))
ica_data = b_filter.apply(x)
csp_t = np.dot(np.dot(ica_data.T, ica_data), csp)
ica_t = np.dot(np.dot(ica_data.T, ica_data), ica)

fig, axes = plt.subplots(3, 4)
for j, spat in enumerate([ica, ssd, csp]):
    plot_topomap(spat, montage.get_pos(), axes=axes[j, 3], show=False, contours=None)

for j, spat in enumerate([ica_t, ssd_t, csp_t]):
    plot_topomap(spat, montage.get_pos(), axes=axes[j, 2], show=False, contours=None)

for j, spat in enumerate([ica, ssd, csp]):
    t = np.arange(3*fs, 5*fs)
    axes[j, 0].plot(t/fs, np.dot(x[t, :], spat))
    axes[j, 0].set_ylim(-0.00001, 0.00001)
for j, spat in enumerate([ica, ssd, csp]):
    axes[j, 1].plot(*welch(np.dot(x, spat), fs, nperseg=fs*2))
    axes[j, 1].set_xlim(0, 30)

axes[2, 0].set_xlabel('Time, s')
axes[2, 1].set_xlabel('Freq., Hz')
axes[0, 0].set_title('Time ser.')
axes[0, 1].set_title('Spectrum')
axes[0, 2].set_title('Topography')
axes[0, 3].set_title('Spat. filt.')
axes[0, 0].set_ylabel('ICA')
axes[1, 0].set_ylabel('SSD')
axes[2, 0].set_ylabel('CSP')

plt.show()