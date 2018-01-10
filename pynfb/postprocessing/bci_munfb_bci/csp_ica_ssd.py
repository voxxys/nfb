from PyQt4 import QtGui

from pynfb.inlets.montage import Montage
from pynfb.widgets.update_signals_dialog import SignalsSSDManager
import h5py
from pynfb.postprocessing.utils import get_info
if __name__ == '__main__':
    import numpy as np
    from scipy.io import loadmat
    #mat_file = loadmat(r'C:\Users\nsmetanin\Downloads\nfb_bci\wetransfer-07cfaf\Subj1_data.mat')
    #x = np.concatenate(  [mat_file['EEGdata'][:, :, j] for j in range(mat_file['EEGdata'].shape[2])], axis=1).T
    #trial_marks = (mat_file['EEGtimes'] == 0.).astype(int)
    #marks = np.concatenate([trial_marks[0] for k in range(mat_file['EEGdata'].shape[2]) ])
    #import pylab as plt
    #plt.plot(data)
    #plt.show()
    #channels = [b[0] for b in mat_file['EEGchanslabels'][0]]

    #print(np.shape(channels))
    #print(np.shape(x))
    #print(np.shape(marks))
    file = r'C:\Users\Nikolai\Desktop\bci_nfb_bci\bci_nfb_bci\bci_mu_ica_S5_d2_08-24_16-25-25\experiment_data.h5'
    with h5py.File(file) as f:
        fs, channels, protocol_names = get_info(f, [])
        x = [f['protocol{}/raw_data'.format(k+1)][:] for k in range(12)]

    n_ch = len(channels)
    from pynfb.signals import CompositeSignal, DerivedSignal

    signals = [DerivedSignal(ind = 0, source_freq=500, name='Signal',  n_channels=n_ch)]

    app = QtGui.QApplication([])

    montage = Montage(channels)

    w = SignalsSSDManager(signals, x, montage, None, None, [], protocol_seq=protocol_names[:12],  sampling_freq=fs)
    w.exec()

    import pylab as plt
    from scipy.signal import welch

    np.save('spat-ssd.npy', signals[0].spatial_filter)
    plt.plot(*welch(np.dot(np.concatenate(x), signals[0].spatial_filter), fs*4))
    plt.show()
    #plt.plot(np.arange(50000) / 258, marks * np.max(np.dot(x, signals[0].spatial_filter)))
    #plt.show()
    app.exec_()