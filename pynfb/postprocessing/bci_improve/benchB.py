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
import statsmodels.nonparametric.api as smnp


a = QApplication([])


def get_kde_data(data):
    kde = smnp.KDEUnivariate(data)
    kde.fit()
    return kde.support, kde.density


def find_border(data1, data2):
    grid1, kde1 = get_kde_data(data1)
    grid2, kde2 = get_kde_data(data2)
    grid = np.sort(np.concatenate((grid1, grid2)))
    kde1 = np.interp(grid, grid1, kde1)
    kde2 = np.interp(grid, grid2, kde2)

    #plt.figure()
    #plt.plot(grid, kde1)
    #plt.plot(grid, kde2)
    #plt.show()

    inds = np.arange(len(grid))
    return grid[np.argmax(kde1) + np.argmin(np.abs(kde1 - kde2)[(inds > np.argmax(kde1)) & (inds < np.argmax(kde2))])]


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
files = [r'C:\Users\Nikolai\Desktop\neurotlon_data\D4\motors_leftlegs_11-22_13-19-36\experiment_data.h5']

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
print(y)
print(len(x))

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
    #(_rej, filt, topo, _unmix, _bandpass, _) = ICADialog.get_rejection(np.concatenate(list(x[y=='Left']) + list(x[y=='Legs'])), channels, fs, mode='csp')
    # filt, topography, bandpass, rejections = SelectSSDFilterWidget.select_filter_and_bandpass(np.concatenate(x), ch_names_to_2d_pos(channels), channels, sampling_freq=fs)
    np.save('filt_{}.npy'.format(subj), filt)



#filt *= 0
#filt[channels.index('Cz')] = 1
#filt = np.concatenate([filt, [0, 0, 0, 0]])
#print(dict(zip(channels, filt)))

df = pd.DataFrame()
for j, (xx, yy) in enumerate(zip(x, y)):
    if yy in ['Right', 'Left', 'Tongue', 'Rest', 'Legs', 'Middle']:
        f, p = signal.welch(np.dot(xx, filt), fs, nperseg=int(fs) * 2)
        df = df.append(pd.DataFrame({'freq': f, 'power': p, 'state': yy, 'trial': j}))

signal = DerivedSignal(ind=1, bandpass_high=20, bandpass_low=17, name='legs', n_channels=len(channels), spatial_filter=filt,
              disable_spectrum_evaluation=False, n_samples=500, smoothing_factor=0.99, source_freq=fs,
              estimator_type='envdetector', temporal_filter_type='butter', smoother_type='exp', filter_order=2)

signal2 = DerivedSignal(ind=1, bandpass_high=20, bandpass_low=17, name='legs', n_channels=len(channels), spatial_filter=filt,
              disable_spectrum_evaluation=False, n_samples=500, smoothing_factor=0.9925, source_freq=fs,
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


df_signal['signal'] = (df_signal['signal'] - df_signal['signal'].mean()) / df_signal['signal'].std()

df_signal = df_signal[(df_signal['times']>2)]
high_legs = np.percentile(df_signal[df_signal['state'] == 'Legs']['signal'], 90)
low_left = np.percentile(df_signal[df_signal['state'] == 'Left']['signal'], 10)
print(high_legs)
print(low_left)
#print(sum((df_signal['state'] == 'Middle') & (df_signal['signal'] < low_left) & (df_signal['signal'] > high_legs)) / sum(df_signal['state'] == 'Middle'))


for state in states[:2]:
    sss = df_signal[(df_signal['state'] == state) & (df_signal['times']>3)]['signal']
    plt.vlines([np.percentile(sss, 80), np.percentile(sss, 20)], 0, 1, {'Legs': cm[0], 'Left': cm[1], 'Middle': cm[2]}[state])
    sns.distplot(sss, norm_hist=True)

plt.vlines(find_border(
    df_signal[(df_signal['state'] == 'Legs') & (df_signal['times']>3)]['signal'],
    df_signal[(df_signal['state'] == 'Left') & (df_signal['times']>3)]['signal']), 0, 1)
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
