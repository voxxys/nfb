from pynfb.postprocessing.utils import get_info, fft_filter
import pandas as pd
import pylab as plt
import json
import h5py
from mne.viz import plot_topomap
from pynfb.inlets.montage import Montage
from scipy.signal import *
import seaborn as sns

from pynfb.signal_processing.helpers import get_outliers_mask
cm = sns.color_palette()

work_dir = r'C:\Users\Nikolai\Desktop\bci_nfb_bci\bci_nfb_bci'
desc_file = 'info.json'
import numpy as np

with open('{}/{}'.format(work_dir, desc_file)) as f:
    desc = json.loads(f.read())


fig, axes = plt.subplots(len(desc['subjects']), 9)
sns.despine(fig, left=True)

for subj, days in enumerate(desc['subjects'][:]):
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
            montage = Montage(channels)
            spatial = f['protocol15/signals_stats/left/spatial_filter'][:]
            bandpass = f['protocol15/signals_stats/left/bandpass'][:]
            df['SMR'] = np.dot(df[channels], spatial)
            df['SMR'] = df['SMR'].iloc[~get_outliers_mask(df[['SMR']], std=3, iter_numb=5)]

        plot_topomap(spatial, montage.get_pos(), axes=axes[subj, day*3], show=False, contours=None)


        states = ['Open', 'Left', 'Right']
        axes[subj, 3 * day + 2].plot(df['SMR'][~df['block_name'].isin(states)], c=cm[3])
        for s, state in enumerate(states):
            for before in [True, False]:
                smr = df['SMR'][(df['block_name'] == state) &
                         ((df['block_number'] < 15) if before else (df['block_number'] > 15))]
                freq, pxx = welch(smr.dropna(), fs, nperseg=fs)
                freq_slice = (freq > 5) & (freq <30)

                axes[subj, 3 * day + 1].plot(freq[freq_slice], pxx[freq_slice], alpha=0.5 if before else 1, c=cm[s])

                axes[subj, 3 * day + 2].plot(smr, alpha=0.5 if before else 1, c=cm[s])
                axes[subj, 3 * day + 1].set_yticks([])
                axes[subj, 3 * day + 2].set_yticks([])

                # plot band
                axes[subj, 3 * day + 1].vlines(bandpass, *axes[subj, 3 * day + 1].get_ylim())
[axes[0, k*3].set_title('Day' + str(k+1)) for k in range(3)]
[axes[k, 0].set_ylabel('S' + str(k+1)) for k in range(len(axes))]
plt.show()

