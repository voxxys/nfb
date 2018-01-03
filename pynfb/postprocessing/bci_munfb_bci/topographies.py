from pynfb.postprocessing.utils import get_info, fft_filter
import pandas as pd
import pylab as plt
import json
import h5py
from mne.viz import plot_topomap
from pynfb.inlets.montage import Montage
from scipy.signal import *
import seaborn as sns

from pynfb.signal_processing.filters import ButterFilter
from pynfb.signal_processing.helpers import get_outliers_mask
cm = sns.color_palette()

work_dir = r'C:\Users\Nikolai\Desktop\bci_nfb_bci\bci_nfb_bci'
desc_file = 'info.json'
import numpy as np

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

with open('{}/{}'.format(work_dir, desc_file)) as f:
    desc = json.loads(f.read())


fig, axes = plt.subplots(len(desc['subjects']), 9)
sns.despine(fig, left=True)

topos = []
for subj, days in enumerate(desc['subjects'][:]):
    topos.append([])
    for day, exp_name in enumerate(days):

        exp_data_path = '{}\{}\{}'.format(work_dir, exp_name, 'experiment_data.h5')
        print(exp_data_path)

        # read filter
        with h5py.File(exp_data_path) as f:
            fs, channels, p_names = get_info(f, ['A1', 'A2'])
            data = [f['protocol{}/raw_data'.format(k + 1)][:] for k in range(len(p_names))]

            df = pd.DataFrame(np.concatenate(data), columns=channels)
            df['block_name'] = np.concatenate([[p] * len(d) for p, d in zip(p_names, data)])
            df['block_number'] = np.concatenate([[j + 1] * len(d) for j, d in enumerate(data)])
            del data
            montage = Montage(channels)
            spatial = f['protocol15/signals_stats/left/spatial_filter'][:]
            bandpass = f['protocol15/signals_stats/left/bandpass'][:]
            df['SMR'] = np.dot(df[channels], spatial)
            df = df.iloc[~get_outliers_mask(df['SMR'])]

            # restore ica topographies
            # df[channels] = fft_filter(df[channels], fs, [0.05, 45])
            b_filter = ButterFilter((3, 45), fs, len(channels))
            ica_data = b_filter.apply(df[channels][df['block_number'] < 13])
            topography = np.dot(np.dot(ica_data.T, ica_data), spatial)
            topos[subj].append(topography)
        plot_topomap(spatial, montage.get_pos(), axes=axes[subj, day*3], show=False, contours=None)
        plot_topomap(topography, montage.get_pos(), axes=axes[subj, day * 3 + 1], show=False, contours=None)


        states = ['Open', 'Left', 'Right']
        #axes[subj, 3 * day + 2].plot(df['SMR'][~df['block_name'].isin(states)], c=cm[3])
        lines_to_plot = [None]*6
        for s, state in enumerate(states):
            for before in [True, False]:
                smr = df['SMR'][(df['block_name'] == state) &
                         ((df['block_number'] < 15) if before else (df['block_number'] > 15))]
                freq, pxx = welch(smr.dropna(), fs, nperseg=fs)
                freq_slice = (freq > 5) & (freq <30)

                lines_to_plot[s + int(before)*3] = axes[subj, 3 * day + 2].plot(freq[freq_slice], pxx[freq_slice],
                                                                             ':' if before else '-', c=cm[s])[0]
                if before:
                    axes[subj, 3 * day + 2].plot(freq[freq_slice], pxx[freq_slice], alpha=0.3)

                #axes[subj, 3 * day + 2].plot(smr, alpha=0.5 if before else 1, c=cm[s])
                axes[subj, 3 * day + 2].set_yticks([])

                # plot band
                axes[subj, 3 * day + 2].vlines(bandpass, *axes[subj, 3 * day + 2].get_ylim())
                axes[subj, 3 * day + 2].set_xlabel(' ')
[axes[0, k*3 + 1].set_title('Day' + str(k+1)) for k in range(3)]
[axes[-1, k*3 + 2].set_xlabel('Freq., Hz') for k in range(3)]
[axes[-1, k*3 + 1].set_xlabel('Topography') for k in range(3)]
[axes[-1, k*3 + 0].set_xlabel('Spat. filt.') for k in range(3)]
[axes[k, 0].set_ylabel('S' + str(k+1)) for k in range(len(axes))]
plt.figlegend(lines_to_plot, [ 'Open AFTER', 'Left AFTER', 'Right AFTER', 'Open BEFORE', 'Left BEFORE', 'Right BEFORE'])

plt.show()

import pickle
with open('topos.pickle', 'wb') as f:
    pickle.dump(topos, f)