import numpy as np
from collections import OrderedDict
from pynfb.postprocessing.utils import get_info, add_data
import h5py
from pynfb.signals.bci import BCISignal


with h5py.File(r'C:\Users\Nikolai\PycharmProjects\nfb\pynfb\results\bci-wrestling_10-11_13-17-44\experiment_data.h5') as f:
    fs, channels, p_names = get_info(f, ['A1', 'A2', 'Pz', 'AUX'])
    raw = OrderedDict()
    for j, name in enumerate(p_names):
        if name in ['Left', 'Open', 'Right']:
            raw = add_data(raw, name, f['protocol{}/raw_data'.format(j + 1)][:], j)

    # print('bci before:', list(raw.keys())[:9])
    bci = BCISignal(fs, channels, 'bci', 0)

    def get_Xy(protocols):
        print(protocols)
        X = [raw[prot] for prot in protocols]
        def get_state(name):
            if 'Open' in name:
                return 0
            elif 'Left' in name:
                return 1
            elif 'Right' in name:
                return 2
            else:
                raise TypeError('Bad state', name)
        y = [np.ones(len(raw[prot])) * get_state(prot) for prot in protocols]
        X = np.vstack(X)
        y = np.concatenate(y, 0)
        return X, y

    X_train, y_train = get_Xy(list(raw.keys())[:9])
    X_test, y_test = get_Xy(list(raw.keys())[9:12])
    print(list(raw.keys()))
    for k in range(1):
        bci.reset_model()
        a_train = bci.fit_model(X_train, y_train)
        print(bci.model.get_accuracies(X_train, y_train))
        print(bci.model.get_accuracies(X_test, y_test))

