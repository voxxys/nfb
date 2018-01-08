import seaborn as sns
import pandas as pd
import numpy as np
import json
import pylab as plt

work_dir = r'C:\Users\Nikolai\Desktop\bci_nfb_bci\bci_nfb_bci'

def open_desc(group='Real'):
    desc_file = 'info_mock.json' if group == 'Mock' else 'info.json'
    with open('{}/{}'.format(work_dir, desc_file)) as f:
        desc = json.loads(f.read())
    return desc

fs = 500
cm = sns.color_palette()


stats = pd.DataFrame(columns=['group', 'subj', 'day', 'hand', 'mean', '50', '75', '25', 'block_number', 'block_name'])

for group in ['Real', 'Mock']:
    desc = open_desc(group)
    print(desc)
    for subj, days in enumerate(desc['subjects'][:]):
        for day, exp_name in enumerate(days[:]):
            exp_data_path = '{}\{}.pkl'.format(work_dir, exp_name)
            df = pd.read_pickle(exp_data_path, 'gzip')
            #df['before'] = df['block_number'] <= 12
            df['logenv'] = df['env']
            for block in df['block_number'].unique():
                block_name = df.loc[df['block_number']==block, 'block_name'].iloc[0]
                dfb = df.loc[df['block_number']==block, 'logenv']
                stats.loc[len(stats)] = {
                    'group': group, 'subj': subj+1, 'day': day+1, 'hand': desc['hand'][subj][day], 'mean': dfb.mean(),
                    '50': dfb.quantile(0.5), '75': dfb.quantile(0.75), '25': dfb.quantile(0.25),
                    'block_number': block, 'block_name': block_name}

stats.to_csv('stats_bci_mu_bci_nonlog.csv', index=False)