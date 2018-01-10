import pandas as pd
from scipy.signal import welch
import pylab as plt


df = pd.read_pickle(r'C:\Users\Nikolai\PycharmProjects\nfb\pynfb\postprocessing\bci_munfb_bci\bci_mu_ica_S5_d2_08-24_16-25-252.pkl', 'gzip')

print(df.loc[(df['block_name'] == 'FB'), 'block_number'].unique())
plt.plot(*welch(df.loc[df['block_number'] == 18, 'SMR'], 500, nperseg=500*4), '--')
plt.plot(*welch(df.loc[df['block_number'] == 20, 'SMR'], 500, nperseg=500*4), '--')
plt.plot(*welch(df.loc[df['block_number'] == 22, 'SMR'], 500, nperseg=500*4), '--')
plt.plot(*welch(df.loc[df['block_name'] == 'Right', 'SMR'], 500, nperseg=500*4))
plt.plot(*welch(df.loc[df['block_name'] == 'Baseline', 'SMR'], 500, nperseg=500*4))
plt.plot(*welch(df.loc[(df['block_number'] < 13) & (df['block_name'] == 'Open'), 'SMR'], 500, nperseg=500*4))
plt.plot(*welch(df.loc[(df['block_number'] > 13) & (df['block_name'] == 'Open'), 'SMR'], 500, nperseg=500*4))
plt.legend(['FB1', 'FB2', 'FB3', 'Right', 'Baseline', 'Open before', 'Open after'])
plt.xlabel('Freq., Hz')
plt.show()