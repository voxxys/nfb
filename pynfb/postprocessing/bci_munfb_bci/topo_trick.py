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

