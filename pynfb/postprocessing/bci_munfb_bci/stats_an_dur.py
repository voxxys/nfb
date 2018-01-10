import pandas as pd
import pylab as plt
import seaborn as sns
import numpy as np
from scipy.stats import ranksums
cm = sns.color_palette()
df = pd.read_csv('stats_bci_mu_bci_full_norm.csv')
df = df.drop(df[df['block_name'].isin(['Bci', 'Rest', 'Filters'])].index)
df = df.drop(df[df['group']=='Mock'].index)
df = df.drop(df[~df['subj'].isin([2, 3, 5])].index)
#df = df.drop(df[df['day']!=1].index)
#df = df.drop(df[df['subj']!=2].index)


df.loc[df['block_number'] < 13, 'block_name'] = df.loc[df['block_number'] < 13, 'block_name'].replace({'Open': 'Open-before', 'Left': 'Left-before', 'Right': 'Right-before'})
df.loc[df['block_number'] > 13, 'block_name'] = df.loc[df['block_number'] > 13, 'block_name'].replace({'Open': 'Open-after', 'Left': 'Left-after', 'Right': 'Right-after'})
for j, fb in enumerate(sorted(df.loc[df['block_name'] == 'FB', 'block_number'].unique())):
    df.loc[df['block_number'] == fb, 'block_name'] = 'FB'+str(j+1)


df = df.drop(df[~df['block_name'].isin(['FB1', 'FB2', 'FB3'])].index)
df['block_day'] = df[['block_name', 'day']].apply(lambda x: x[0] + ' Day'+ str(x[1]), axis=1)

stat = '50'
norm_by = ['FB1 Day1']
for group in df['group'].unique():
    for subj in df['subj'].unique():
        for day in df['day'].unique():
            mask = (df['group'] == group) & (df['subj'] == subj)

            motor_before = df['block_day'].isin(norm_by)
            coeff = df.loc[mask & motor_before, stat].median()
            df.loc[mask, stat] -= coeff


#df[stat] = np.exp(df[stat])
# df[stat] = -df[stat]
g = sns.barplot('block_day', stat, 'subj', df, estimator=np.mean)
g.set_xticklabels(g.get_xticklabels(), rotation=30)
plt.tight_layout()
plt.show()