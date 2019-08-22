import os.path as opth
from libspn.examples.convspn.utils import listdirs
import pandas as pd
import json

base = 'logs/logs'
dfs = []
hyperparams = []

group_by_columns = ['l0_prior_factor', 'sample_path', 'use_unweighted', '']

# metrics = ['accuracy']
metrics = ['l2_b', 'l2_l']

for name in listdirs(base):
    for run in listdirs(opth.join(base, name)):
        path = opth.join(base, name, run, 'test_epoch.csv')

        if opth.exists(path):
            df = pd.read_csv(path).tail(n=1)
            if not all(m in df.columns for m in metrics):
                continue
            df['run'] = run
            df['name'] = name
            dfs.append(df)

        path = opth.join(base, name, run, 'hyperparams.json')
        if opth.exists(path):
            with open(path, 'r') as fp:
                hp = json.load(fp)
                hp['name'] = name
                del hp['experiment_folder']
                del hp['run_name']
                hyperparams.append(hp)

hyperparams = pd.DataFrame(hyperparams)

hyperparams = hyperparams[['name'] + [c for c in sorted(hyperparams.columns) if c != 'name']]
hyperparams.drop_duplicates().transpose().to_csv('hyperparams.csv', header=False)

more_than_one_unique = [c for c, nunique in hyperparams.nunique().iteritems() if nunique > 1]

hyperparams_unique = hyperparams[more_than_one_unique]
hyperparams_unique.drop_duplicates().transpose().to_csv('hyperparams_uniq.csv', header=False)

df = pd.concat(dfs)


results = df.groupby(by='name').agg({k: ['mean', 'std'] for k in metrics})
results.columns = ['_'.join(col).strip() for col in results.columns.values]
# results.to_csv('results.csv', index=False)

# results.columns = results.columns.get_level_values(0)
results.to_csv('results.csv')