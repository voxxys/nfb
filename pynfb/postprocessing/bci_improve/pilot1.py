import h5py
import numpy as np
from scipy import signal
import pandas as pd
import pylab as plt

from pynfb.postprocessing.utils import get_info, fft_filter
from pynfb.protocols import SelectSSDFilterWidget
from pynfb.protocols.ssd.topomap_selector_ica import ICADialog
from PyQt4.QtGui import QApplication
from pynfb.signals.bci import BCISignal

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
            data = np.array([f['protocol{}/raw_data'.format(k + 1)][:] for k in range(p_name_index+1, len(p_names))])
            labels = p_names[p_name_index+1:]

        #else:
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

subj = 'ad2'
if subj == 'a':
    files = {'imag':[r'C:\Users\Nikolai\Desktop\neurotlon_data\S3\motors_imag_11-14_16-55-17\experiment_data.h5',
                     r'C:\Users\Nikolai\Desktop\neurotlon_data\S3\motors_imag_11-14_17-21-22\experiment_data.h5'],
             'real': [r'C:\Users\Nikolai\Desktop\neurotlon_data\S3\motors_real_imag_11-14_16-20-06\experiment_data.h5']}['imag']
elif subj == 'n':
    files = [r'C:\Users\Nikolai\PycharmProjects\nfb\pynfb\results\motors_11-09_20-29-35\experiment_data.h5']
elif subj == 'ad2':
    files = [r'C:\Users\Nikolai\Desktop\neurotlon_data\D2\S3\motors_tongue_is_legs__rest_is_cog_11-17_20-46-16\experiment_data.h5']
else:
    files = []

x = []
y = []
for ff in files:
    fs, channels, p_names, data, labels = load_data_lists(ff, add_after=False)
    x.append(data)
    y.append(labels)
#x = np.concatenate(x)
#y = np.concatenate(y)
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
    #(_rej, filt, topo, _unmix, _bandpass, _) = ICADialog.get_rejection(np.concatenate(list(x[y=='Right']) + list(x[y=='Tongue'])), channels, fs, mode='csp')
    # filt, topography, bandpass, rejections = SelectSSDFilterWidget.select_filter_and_bandpass(np.concatenate(x), ch_names_to_2d_pos(channels), channels, sampling_freq=fs)
    np.save('filt_{}.npy'.format(subj), filt)

print(dict(zip(channels, filt)))

#filt *= 0
#filt[channels.index('Cz')] = 1


df = pd.DataFrame()
for j, (xx, yy) in enumerate(zip(x, y)):
    if yy in ['Right', 'Left', 'Tongue', 'Rest']:
        f, p = signal.welch(np.dot(xx, filt), fs, nperseg=int(fs)*2)
        df = df.append(pd.DataFrame({'freq': f, 'power': p, 'state': yy, 'trial': j}))


f, axs = plt.subplots(3, sharex=True)

x1 = [xx for xx, yy in zip(x, y) if yy in ['Left', 'Tongue']]
y1 = [yy for yy in y if yy in ['Left', 'Tongue']]

from pynfb.signal_processing.filters import FFTBandEnvelopeDetector, ExponentialSmoother, ButterBandEnvelopeDetector, ComplexDemodulationBandEnvelopeDetector, SGSmoother
smoother = ExponentialSmoother(0.99)
#smoother = SGSmoother(151, 4)
env = ButterBandEnvelopeDetector((17, 20), fs, smoother, 2)
#env = ComplexDemodulationBandEnvelopeDetector((8, 11), fs, smoother)
ica = np.dot(np.concatenate(x1), filt)
ica_full = np.dot(np.concatenate(x), filt)

yyy = np.concatenate([np.array([yy]*len(xx)) for xx, yy in zip(x1, y1)])
axs[0].plot(env.apply(ica)[1000:])

from sklearn.linear_model import LogisticRegressionCV
cf = LogisticRegressionCV()
envel = ButterBandEnvelopeDetector((17, 20), fs, smoother, 4).apply(ica)[:, None]**0.5
envel = (envel - envel.mean())/envel.std()
print(envel.shape)
cf.fit(envel, yyy)
axs[2].plot(cf.predict_proba(envel)[:, 0][1000:])

axs[1].plot(yyy[1000:])
axs[1].plot(cf.predict(envel)[1000:])


plt.show()


import seaborn as sns
#print([)
cm = sns.color_palette()
print(df['state'].unique())
print(df)
tongue_is = 'Legs'
df['state'].replace('Tongue', tongue_is, inplace=True)
sns.tsplot(df, time='freq', unit='trial', condition='state', value='power', ci='sd', estimator=np.mean, color={'Rest': cm[0], 'Left': cm[1], 'Right': cm[2], tongue_is: cm[3]})
plt.xlim(0, 25)
plt.show()



