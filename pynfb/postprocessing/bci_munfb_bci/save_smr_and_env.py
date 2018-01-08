from pynfb.postprocessing.utils import get_info, fft_filter
import pandas as pd
import pylab as plt
import json
import h5py
from mne.viz import plot_topomap
from pynfb.inlets.montage import Montage
from scipy.signal import *
import seaborn as sns
import numpy as np
from pynfb.signal_processing.filters import ButterFilter
cm = sns.color_palette()


def get_outliers_mask(data_raw: np.ndarray, iter_numb=2, std=7):
    data_pwr = data_raw
    indexes = np.arange(data_pwr.shape[0])
    for i in range(iter_numb):
        mask = np.abs(data_pwr - data_pwr.mean()) < std * data_pwr.std()
        #plt.plot(~mask)
        mask = ~(pd.Series(~mask).rolling(3*fs, center=True).mean() > 0)
        #plt.plot(~mask)
        #plt.show()
        indexes = indexes[mask]
        data_pwr = data_pwr[mask]
    print('Dropped {} outliers'.format(data_raw.shape[0] - len(indexes)))
    outliers_mask = np.ones(shape=(data_raw.shape[0], ))
    outliers_mask[indexes] = 0
    return outliers_mask.astype(bool)

work_dir = r'C:\Users\Nikolai\Desktop\bci_nfb_bci\bci_nfb_bci'
desc_file = 'info.json'


with open('{}/{}'.format(work_dir, desc_file)) as f:
    desc = json.loads(f.read())


#fig, axes = plt.subplots(len(desc['subjects']), 3)
#sns.despine(fig, left=True)

for subj, days in enumerate(desc['subjects'][-1:]):
    for day, exp_name in enumerate(days[2:3]):
        exp_data_path = '{}\{}\{}'.format(work_dir, exp_name, 'experiment_data.h5')
        print(exp_data_path)

        # read filter
        with h5py.File(exp_data_path) as f:
            fs, channels, p_names = get_info(f, ['A1', 'A2'])
            means = [f['protocol{}/signals_stats/left/mean'.format(k + 1)].value for k in range(len(p_names))]
            print([(np.isnan(mean), p) for mean, p in zip(means, p_names)])
            stds = [f['protocol{}/signals_stats/left/std'.format(k + 1)].value for k in range(len(p_names))]
            data = [f['protocol{}/signals_data'.format(k + 1)][:, 0] * (1 if np.isnan(stds[max(0, k-1)]) else stds[max(0, k-1)]) + (0 if np.isnan(means[max(0, k-1)]) else means[max(0, k-1)]) for k in range(len(p_names))]
            df = pd.DataFrame(np.concatenate(data), columns=['fb'])
            df['t'] = np.arange(len(df)) / fs

            data = [f['protocol{}/raw_data'.format(k + 1)][:] for k in range(len(p_names))]
            eeg = np.concatenate(data)
            spatial = f['protocol15/signals_stats/left/spatial_filter'][:]
            df['SMR'] = np.dot(eeg, spatial)

            plt.plot(df['t'], df['SMR'])
            from scipy import stats
            mask = ~get_outliers_mask(df['SMR'])

            plt.plot(df['t'][mask], df['SMR'][mask])

            plt.plot(df['t'], df['fb']*5)
            plt.show()

