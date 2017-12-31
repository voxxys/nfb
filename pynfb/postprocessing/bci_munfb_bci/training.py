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
desc_file = 'info_mock.json'
import numpy as np

with open('{}/{}'.format(work_dir, desc_file)) as f:
    desc = json.loads(f.read())


fig, axes = plt.subplots(len(desc['subjects']), 3)
sns.despine(fig, left=True)

for subj, days in enumerate(desc['subjects'][:]):
    for day, exp_name in enumerate(days):
        exp_data_path = '{}\{}\{}'.format(work_dir, exp_name, 'experiment_data.h5')
        print(exp_data_path)

        # read filter
        with h5py.File(exp_data_path) as f:
            fs, channels, p_names = get_info(f, ['A1', 'A2'])
            means = [f['protocol{}/signals_stats/left/mean'.format(k + 1)].value for k in range(len(p_names))]
            print([(np.isnan(mean), p) for mean, p in zip(means, p_names)])
            stds = [f['protocol{}/signals_stats/left/std'.format(k + 1)].value for k in range(len(p_names))]
            data = [f['protocol{}/signals_data'.format(k + 1)][:, 0] * (1 if np.isnan(stds[k-1]) else stds[k-1]) + (0 if np.isnan(means[k-1]) else means[k-1]) for k in range(len(p_names))]

            df = pd.DataFrame(np.concatenate(data), columns=['fb'])
            df['t'] = np.arange(len(df)) / fs
            df['fb'] = df['fb'][df['fb'] < 3*df['fb'].std()]
            df['fb_roll'] = df['fb'].rolling(20*fs, center=True, min_periods=1).median()
            df['fb_roll_25'] = df['fb'].rolling(20 * fs, center=True, min_periods=1).quantile(0.25)
            df['fb_roll_75'] = df['fb'].rolling(20 * fs, center=True, min_periods=1).quantile(0.75)
            df['block_name'] = np.concatenate([[p] * len(d) for p, d in zip(p_names, data)])
            df['block_number'] = np.concatenate([[j + 1] * len(d) for j, d in enumerate(data)])
            del data

        blocks = ['FB', 'Rest', 'Baseline']
        #axes[subj, 3 * day + 2].plot(df['SMR'][~df['block_name'].isin(states)], c=cm[3])
        lines_to_plot = [None]*3
        for b in df['block_number'].unique():
            b_name = df['block_name'][df['block_number'] == b].iloc[0]

            if b_name in blocks:
                color = cm[blocks.index(b_name)]
                df_b = df[df['block_number'] == b]
                #axes[subj, day].plot(df_b['t'], df_b['fb'], '-', markersize=1, c=color, alpha=0.5, linewidth=1)
                lines_to_plot[blocks.index(b_name)] = axes[subj, day].plot(df_b['t'], df_b['fb_roll'], '-', markersize=1, c=color)[0]
                axes[subj, day].fill_between(df_b['t'], df_b['fb_roll_25'], df_b['fb_roll_75'], color=color, alpha=0.5 )

        axes[subj, day].set_yticks([])
[axes[0, k].set_title('Day' + str(k+1)) for k in range(3)]
[axes[-1, k + 0].set_xlabel('Time, s') for k in range(3)]
[axes[k, 0].set_ylabel('S' + str(k+1)) for k in range(len(axes))]
#plt.figlegend(lines_to_plot, [ 'Open AFTER', 'Left AFTER', 'Right AFTER', 'Open BEFORE', 'Left BEFORE', 'Right BEFORE'])
plt.figlegend(lines_to_plot, blocks)
plt.plot()
plt.show()

