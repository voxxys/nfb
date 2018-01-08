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

df1 = pd.read_excel(r'C:\Users\Nikolai\Desktop\bci_nfb_bci\bci_nfb_bci\acc-points (4).xlsx')
print('df1', df1)
acc = (df1.loc[df1['bef']==0, 'acc'].as_matrix()+ df1.loc[df1['bef']==1, 'acc'].as_matrix())/2
print(acc.shape, vals.shape)
from scipy import stats

acc2 =
dd = pd.DataFrame()
dd['acc'] = acc
dd['val'] = vals
dd['Group'] = df.sort_values(['gg', 'Day'])['Group'].as_matrix()

dd1 = dd
sns.regplot('val', 'acc', dd1, label='BOTH: $val = k \cdot acc + const$\np-value($k$) = {:.3f}'.format(stats.linregress(dd1['acc'], dd1['val']).pvalue))

dd1 = dd[dd['Group'] == 'Real']
sns.regplot('val', 'acc', dd1, label='REAL: $val = k \cdot acc + const$\np-value($k$) = {:.3f}'.format(stats.linregress(dd1['acc'], dd1['val']).pvalue))

dd1 = dd[dd['Group'] == 'Mock']
sns.regplot('val', 'acc', dd1, label='MOCK: $val = k \cdot acc + const$\np-value($k$) = {:.3f}'.format(stats.linregress(dd1['acc'], dd1['val']).pvalue))

plt.legend()
plt.show()