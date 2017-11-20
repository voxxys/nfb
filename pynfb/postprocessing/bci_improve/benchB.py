import h5py
import numpy as np
from scipy import signal
import pandas as pd
import pylab as plt

from pynfb.postprocessing.utils import get_info, fft_filter
from pynfb.protocols import SelectSSDFilterWidget
from pynfb.protocols.ssd.topomap_selector_ica import ICADialog
from PyQt4.QtGui import QApplication

from pynfb.signal_processing.filters import ExponentialSmoother, ButterBandEnvelopeDetector
from pynfb.signals import DerivedSignal
from pynfb.signals.bci import BCISignal
import seaborn as sns

a = QApplication([])


# load data
def load_data_lists(file_path, add_after=False):
    with h5py.File(file_path) as f:
        fs, channels, p_names = get_info(f, ['AUX', 'A1', 'A2'])
        channels = [ch.split('-')[0] for ch in channels]
        if not add_after:
            p_names = p_names[:p_names.index('Pause')]
            data = np.array([f['protocol{}/raw_data'.format(k + 1)][:] for k in range(len(p_names))])
            labels = np.array(p_names)
        else:
            p_name_index = p_names.index('Pause')
            data = np.array([f['protocol{}/raw_data'.format(k + 1)][:] for k in range(p_name_index + 1, len(p_names))])
            labels = p_names[p_name_index + 1:]

            # else:
            #    data = np.array([f['protocol{}/raw_data'.format(k + 1)][:] for k in range(len(p_names)) if p_names!='Pause'])
            #    p_names.remove('Pause')
            #    labels = np.array(p_names)

    return fs, channels, p_names, data, labels


# load data
def load_data(file_path):
    with h5py.File(file_path) as f:
        fs, channels, p_names = get_info(f, ['AUX', 'A1', 'A2'])
        channels = [ch.split('-')[0] for ch in channels]
        p_names = p_names[:p_names.index('Pause')]
        data = [f['protocol{}/raw_data'.format(k + 1)][:] for k in range(len(p_names))]
        labels = np.concatenate([[p] * len(d) for p, d in zip(p_names, data)])
        data = np.concatenate(data)
    return fs, channels, p_names, data, labels


# select states
def select_states(data, labels):
    states = {'Rest': 0, 'Left': 1, 'Right': 2}
    indexes = [j for j in range(len(data)) if labels[j] in states.keys()]
    x = data[indexes]
    y = np.array(list(map(lambda x: states[x], labels[indexes])))
    return x, y


subj = 'ad3'
files = [r'C:\Users\Nikolai\Desktop\neurotlon_data\D3\motors_leftlegs_11-20_13-21-13\experiment_data.h5']

x = []
y = []
for ff in files:
    fs, channels, p_names, data, labels = load_data_lists(ff, add_after=False)
    x.append(data)
    y.append(labels)
# x = np.concatenate(x)
# y = np.concatenate(y)
x = x[0]
y = y[0]

if False:
    # fit bci
    bci = BCISignal(fs, channels, 'foo', 42)
    bci.fit_model(x_train, y_train)
    print(bci.model.get_accuracies(x_train, y_train))
    print(bci.model.get_accuracies(x, y))

try:
    filt = np.load('filt_{}.npy'.format(subj))
except FileNotFoundError:
    (rej, filt, topo, _unmix, _bandpass, _) = ICADialog.get_rejection(np.concatenate(x), channels, fs, mode='ica')
    # (_rej, filt, topo, _unmix, _bandpass, _) = ICADialog.get_rejection(np.concatenate(list(x[y=='Right']) + list(x[y=='Tongue'])), channels, fs, mode='csp')
    # filt, topography, bandpass, rejections = SelectSSDFilterWidget.select_filter_and_bandpass(np.concatenate(x), ch_names_to_2d_pos(channels), channels, sampling_freq=fs)
    np.save('filt_{}.npy'.format(subj), filt)

print(dict(zip(channels, filt)))

#filt *= 0
#filt[channels.index('Cz')] = 1


df = pd.DataFrame()
for j, (xx, yy) in enumerate(zip(x, y)):
    if yy in ['Right', 'Left', 'Tongue', 'Rest', 'Legs', 'Middle']:
        f, p = signal.welch(np.dot(xx, filt), fs, nperseg=int(fs) * 2)
        df = df.append(pd.DataFrame({'freq': f, 'power': p, 'state': yy, 'trial': j}))

signal = DerivedSignal(ind=1, bandpass_high=20, bandpass_low=17, name='legs', n_channels=len(channels), spatial_filter=filt,
              disable_spectrum_evaluation=False, n_samples=500, smoothing_factor=0.99, source_freq=fs,
              estimator_type='envdetector', temporal_filter_type='butter', smoother_type='exp', filter_order=2)

signal2 = DerivedSignal(ind=1, bandpass_high=20, bandpass_low=17, name='legs', n_channels=len(channels), spatial_filter=filt,
              disable_spectrum_evaluation=False, n_samples=500, smoothing_factor=0.99299, source_freq=fs,
              estimator_type='envdetector', temporal_filter_type='butter', smoother_type='exp', filter_order=2)

#import seaborn as sns
cm = sns.color_palette()
states = ['Legs', 'Left', 'Middle']
#plt.figure()
t = 0
df_signal = pd.DataFrame()

for j, (xx, yy) in enumerate(zip(x, y)):
    signal.update(xx)
    signal2.update(xx)
    df_signal = df_signal.append(pd.DataFrame({'times': np.arange(t, t + len(xx)) / fs, 'signal': signal.current_chunk, 'trial': j, 'state': yy}))
    plt.plot(np.arange(t, t + len(xx)) / fs, signal.current_chunk, c=cm[states.index(yy)])
    plt.plot(np.arange(t, t + len(xx)) / fs, signal2.current_chunk, '--', c=cm[states.index(yy)])
    t += len(xx)

plt.show()
print(df_signal)
sns.tsplot(df_signal, time='times', unit='trial', condition='state', value='signal', ci='sd', estimator=np.mean,
           color={'Legs': cm[0], 'Left': cm[1], 'Middle': cm[2]})
#plt.tight_layout()
plt.show()


# print([)
cm = sns.color_palette()
print(df['state'].unique())
print(df)
tongue_is = 'Legs'
df['state'].replace('Tongue', tongue_is, inplace=True)
sns.tsplot(df, time='freq', unit='trial', condition='state', value='power', ci='sd', estimator=np.mean,
           color={'Legs': cm[0], 'Left': cm[1], 'Middle': cm[2]})
plt.xlim(0, 25)
plt.show()
