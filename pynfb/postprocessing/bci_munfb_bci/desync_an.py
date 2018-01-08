import pandas as pd
import pylab as plt
import seaborn as sns
from scipy.stats import ranksums

df = pd.read_csv('stats_bci_mu_bci_nonlog.csv')
df['before'] = df['block_number'] < 13
stats = 'mean'
print(df.loc[(df['block_name'] == 'Open') & (df['block_number'] < 13)].groupby(['group', 'subj', 'day']).mean())
