import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import pylab as plt

with open(r'mock_topos.pickle', 'rb') as f:
    mock = pickle.load(f)
with open(r'topos.pickle', 'rb') as f:
    real = pickle.load(f)

def get_topo_sum(a):
    for k in range(len(a)):
        for j in range(len(a[k])):
            #plt.hist(a[k][j])
            #plt.show()
            n = a[k][j]/a[k][j][np.argmax(np.abs(a[k][j]))]
            a[k][j] = np.min(a[k][j]/a[k][j][np.argmax(np.abs(a[k][j]))])
    return a

print(get_topo_sum(mock))
print(get_topo_sum(real))


df = pd.DataFrame(columns=['Group', 'Day', 'Val', 'Subj'])
for ja, a in enumerate([real, mock]):


    for k in range(len(a)):

        for j in range(len(a[k])):


            df.loc[len(df)] = {'Group': 'Mock' if ja else 'Real', 'Day': j+1, 'Val': a[k][j], 'Subj': k+1}

df['gg'] = df['Group'] == 'Mock'
print(df.sort_values(['gg', 'Day']))
vals = df.sort_values(['gg', 'Day'])['Val'].as_matrix()
print(vals)
from scipy.stats import ttest_ind, ks_2samp, ranksums
x = df[df['Day']==1]
print(ranksums(x[x['Group']=='Real']['Val'], x[x['Group']=='Mock']['Val']))

x = df[df['Group']=='Real']
print(ranksums(x[x['Day']==1]['Val'], x[x['Day']==2]['Val']))

#sns.boxplot('Day', 'Min(T / T[argmax|T|])', 'Group', data=df)

sns.pointplot('Day', 'Val', 'Group', data=df, dodge=True)
#sns.swarmplot('Day', 'Min(T / T[argmax|T|])', 'Group', data=df, linewidth=1)
plt.show()

df.to_csv('topo_trick.csv', index=False)

df1 = pd.read_excel(r'C:\Users\Nikolai\Desktop\bci_nfb_bci\bci_nfb_bci\acc-points (5).xlsx')
print(df1)
stat = pd.DataFrame(columns=['mon', 'acc', 'group'])
for group_name in ['Real', 'Mock']:
    group = df1[df1['group'] == group_name]
    group_top = df[df['Group'] == group_name]
    for subj in range(1, 8):
        s = group[group['subj'] == subj]
        s_top = group_top[group_top['Subj'] == subj]
        days = sorted(s['day'].unique())
        for d in range(2, len(days)+1):
            stat.loc[len(stat)] = {
                'mon': s_top.loc[s_top['Day']==d, 'Val'].mean() - s_top.loc[s_top['Day']==d-1, 'Val'].mean(),
                'acc': s.loc[s['day']==d, 'acc'].mean() - s.loc[s['day']==d-1, 'acc'].mean(),
                'group': group_name
            }

from scipy import stats
get_p = lambda x: stats.linregress(x['mon'], x['acc']).pvalue

sns.regplot('mon', 'acc', stat, label='BOTH: $acc = k \cdot mon + const$\np-value($k$) = {:.3f}'.format(get_p(stat)))
sns.regplot('mon', 'acc', stat[stat['group']=='Real'], label='REAL: $acc = k \cdot mon + const$\np-value($k$) = {:.3f}'.format(get_p(stat[stat['group']=='Real'])))
sns.regplot('mon', 'acc', stat[stat['group']=='Mock'], label='MOCK: $acc = k \cdot mon + const$\np-value($k$) = {:.3f}'.format(get_p(stat[stat['group']=='Mock'])))
plt.legend()
plt.show()