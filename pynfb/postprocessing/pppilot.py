import h5py
from pynfb.postprocessing.utils import get_info
from pynfb.protocols.ssd.topomap_selector_ica import ICADialog
from PyQt4.QtGui import QApplication
import numpy as np
from scipy import signal
import pylab as plt
a = QApplication([])

experiment_file = r'C:\Users\Nikolai\PycharmProjects\nfb\pynfb\results\experiment_10-20_16-50-30\experiment_data.h5'
with h5py.File(experiment_file) as f:
    fs, channels, p_names = get_info(f, ['Pz', 'Fz', 'A1', 'A2', 'AUX'])
    data = [f['protocol{}/raw_data'.format(2*j+1)][:] for j in range(7)]
    x = np.concatenate([f['protocol{}/raw_data'.format(2*j+1)][:] for j in range(7)])
    (_, spatial, _, _, _, _) = ICADialog.get_rejection(x, channels, fs)


    plt.figure()
    leg = []
    for k in range(7):
        leg.append(p_names[2*k] + str(2*k+1))
        f, Pxx_spec = signal.welch(np.dot(data[k], spatial), fs, 'flattop', 1024, scaling='spectrum')
        plt.plot(f, Pxx_spec)
    plt.xlabel('frequency [Hz]')
    plt.ylabel('Linear spectrum [V RMS]')
    plt.legend(leg)
    plt.show()

print(fs, channels, p_names)