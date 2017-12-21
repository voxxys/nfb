import h5py
from scipy.signal import butter, filtfilt, hilbert, welch
import pandas as pd
import pylab as plt
import numpy as np

f = h5py.File(r'C:\Users\Nikolai\PycharmProjects\nfb\pynfb\results\discrete_bar_test_12-20_20-52-27\experiment_data.h5')
fs = 500
df_all = pd.DataFrame()
get_all_data = lambda name: np.concatenate([f['protocol{}/{}'.format(k+1, name)] for k in range(3)])
df_all['time'] = get_all_data('timestamp_data')
df_all['raw'] = get_all_data('raw_data')[:, 0]
df_all['photo_events'] = get_all_data('raw_data')[:, 1]
df_all['nfb_events'] = get_all_data('mark_data')
df_all['signal'] = get_all_data('signals_data')[:, 0]
df_all['block'] = np.concatenate([[k+1]*len(f['protocol{}/timestamp_data'.format(k+1)]) for k in range(3)])
block_names = [f['protocol{}'.format(k+1)].attrs['name'] for k in range(3)]

b, a = butter(4, [9/250, 11/250], 'band')
df_all['filtfilt'] = filtfilt(b, a, df_all['raw'])
df_all['hilbert_env'] = np.abs(hilbert(df_all['filtfilt']))
df_all['rolling_env'] = df_all['hilbert_env'].rolling(30*fs, center=True).median()



# raw power
for block in range(1, 4):
    f, pxx = welch(df_all[df_all['block']==block]['raw'], fs=fs, nperseg=fs*4, scaling='spectrum')
    plt.plot(f, pxx)
plt.xlabel('freq, Hz')
plt.ylabel('power, $V^2$')
plt.legend(block_names)
plt.show()

# dynamics
for block in range(1, 4):
    plt.plot(df_all[df_all['block']==block]['time'], df_all[df_all['block']==block]['rolling_env'])
plt.xlabel('time, s')
plt.ylabel('voltage, V')
plt.legend(block_names)
plt.tight_layout()
plt.show()


# FB analysis
df = df_all[df_all['block'] == 3]
# plot raw and events
plt.plot(df['time'], df['raw'], alpha=0.4)
plt.plot(df['time'], df['filtfilt'])
plt.plot(df['time'], df['hilbert_env'])
plt.plot(df['time'], df['nfb_events'])
plt.plot(df['time'], df['photo_events'])
plt.legend(['raw', 'filtfilt 9-11Hz', 'hilbert env', 'nfb lab trigger', 'photo response'])
plt.ylim(-0.00005, 0.00006)
plt.xlim(df['time'].iloc[127000], df['time'].iloc[129000])
plt.xlabel('time, s')
plt.ylabel('voltage, V')
plt.tight_layout()
plt.show()

# signal lag
xcov = [df['signal'].corr(df['hilbert_env'].shift(lag)) for lag in range(200)]
opt_lag = np.argmax(xcov)
f, axes = plt.subplots(2)
axes[0].plot(xcov)
axes[0].set_xlabel('lag')
axes[0].set_ylabel('corr')
axes[0].legend(['max corr in {}s lag'.format(opt_lag/500)])
axes[1].plot(df['time'], df['signal'])
axes[1].plot(df['time'], df['hilbert_env'].shift(opt_lag))
axes[1].set_xlim(df['time'].iloc[127000], df['time'].iloc[129000])
axes[1].set_xlabel('time, s')
axes[1].set_ylabel('voltage, V')
axes[1].legend(['signal', 'hilbert env of filtfilt 9-11Hz + lag {}s'.format(opt_lag/500)])
plt.tight_layout()
plt.show()

# photo lag
xcov = [df['photo_events'].corr(df['nfb_events'].shift(lag)) for lag in range(200)]
opt_lag = np.argmax(xcov)
f, axes = plt.subplots(2)
axes[0].plot(xcov)
axes[0].set_xlabel('lag')
axes[0].set_ylabel('corr')
axes[0].legend(['max corr in {}s lag'.format(opt_lag/500)])
axes[1].plot(df['time'], df['photo_events'])
axes[1].plot(df['time'], df['nfb_events'].shift(opt_lag))
axes[1].set_xlabel('time, s')
axes[1].set_xlabel('time, s')
axes[1].legend(['photo_events', 'nfb_events + lag {}s'.format(opt_lag/500)])
plt.tight_layout()
plt.xlim(df['time'].iloc[127000], df['time'].iloc[129000])

plt.show()
