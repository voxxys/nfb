import pandas as pd
import pylab as plt
import seaborn as sns
from scipy.stats import ranksums
cm = sns.color_palette()
df = pd.read_csv('stats_bci_mu_bci_nonlog.csv')

import numpy as np
# FB
fb_numbers = [18, 20, 22]
stat = '50'
df_fb = df



#df = df.loc[df['hand'] == 1]
mot = df['block_name'].isin(['Open', 'Baseline']) & (df['block_number'] < 15 )

print(df_fb[mot].groupby(['group', 'subj', 'day']).mean()[stat])
coeff = df_fb[mot].groupby(['group', 'subj', 'day']).mean()[stat].as_matrix()
print(coeff)
coeff = np.concatenate((coeff[18:], coeff[:18]))

print(df_fb.loc[df_fb['block_number'] == 18])
df_fb.loc[df_fb['block_number'] == 18, stat] /= coeff
df_fb.loc[df_fb['block_number'] == 20, stat] /= coeff
df_fb.loc[df_fb['block_number'] == 22, stat] /= coeff

df_fb = df_fb.loc[df_fb['block_name'].isin(['FB'])]
df_fb['FB Session'] = df_fb.loc[:, ['day', 'block_number']].apply(lambda x: 'Day{}-FB{}'.format(x[0], x[1]//2-8), axis=1)
sns.pointplot('FB Session', stat, 'group', data=df_fb, dodge=True)
sns.swarmplot('FB Session', stat, 'group', data=df_fb, linewidth=1)

for group in ['Real', 'Mock']:
    for subj in df_fb[df_fb['group'] == group]['subj'].unique():
        trace = df_fb.loc[(df_fb['group'] == group) & (df_fb['subj'] == subj), stat].as_matrix()

        plt.plot(trace, c=cm[0 if group == 'Real' else 1], alpha=0.5)

plt.show()
x = df_fb
print(ranksums(x[x['group']=='Real'][stat], x[x['group']=='Mock'][stat]))