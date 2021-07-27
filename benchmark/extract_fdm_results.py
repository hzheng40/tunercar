import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nevergrad as ng
import argparse

# args
parser = argparse.ArgumentParser()
parser.add_argument('--exp_name', type=str, required=True, nargs='+')
parser.add_argument('--npz_dir', type=str, default='../es/quad_fdm_runs/npzs/')
parser.add_argument('--pkl_dir', type=str, default='../es/quad_fdm_runs/optims_pkl/')
args = parser.parse_args()

# list of dfs for all data
dfs = []
pf_dfs = []

for exp in args.exp_name:
    print('Extracting: ' + exp)
    # loading
    data = np.load(args.npz_dir + exp + '.npz')
    optim = ng.optimizers.base.Optimizer.load(args.pkl_dir + exp + '_optim.pkl')

    # pareto front
    pfdf = pd.DataFrame(columns=['exp_name', 'arm_length', 'num_batt', 'batt_v', 'batt_cap', 'batt_m', 'Max Distance', 'Max Hover Time'])
    print('Pareto Front for: ' + exp)
    for param in sorted(optim.pareto_front(), key=lambda p:p.losses[0]):
        print(f'{param.args[0]} with max distance: {-param.losses[0]} and max hover time {-param.losses[1]}')
        ind = param.args[0]
        pfdf = pfdf.append({'exp_name': exp, 'arm_length': ind['arm_length'], 'num_batt': ind['num_batt'], 'batt_v': ind['batt_v'], 'batt_cap': ind['batt_cap'], 'batt_m': ind['batt_m'], 'Max Distance': -param.losses[0], 'Max Hover Time': -param.losses[1]}, ignore_index=True)
    pf_dfs.append(pfdf)

    batch_size = optim.num_workers
    if optim.name == 'CMA':
        batch_size = optim._popsize
    elif optim.name == 'NoisyDE' or optim.name == 'TwoPointsDE' or optim.name[:21] == 'DifferentialEvolution' or optim.name[:13] == 'ConfiguredPSO' or optim.name == 'PSO':
        batch_size = len(optim.population)
    scores_all = data['scores']
    iterations = int(scores_all.shape[0]/batch_size)

    # set up dataframes
    df = pd.DataFrame(columns=['exp_name', 'Iterations', 'Max Distance', 'Max Hover Time'])

    # load data into dataframes
    for i in range(iterations):
        for j in range(batch_size):
            max_dist = scores_all[i*batch_size+j, 0]
            max_hov = scores_all[i*batch_size+j, 1]
            if max_dist != 0.0 and max_hov != 0.0:
                df = df.append({'exp_name': exp, 'Iterations': i, 'Max Distance': max_dist, 'Max Hover Time': max_hov}, ignore_index=True)
    dfs.append(df)

# plot laptimes
sns.set_style('white')
sns.set_style('ticks')
sns.set_context('poster')
palette = sns.color_palette("mako_r", len(args.exp_name))

# plot progress
for df in dfs:
    ax = sns.lineplot(data=df, x='Iterations', y='Max Distance', ci='sd', palette='Paired', label=df['exp_name'][0][13:-11])
plt.show()

for df in dfs:
    ax = sns.lineplot(data=df, x='Iterations', y='Max Hover Time', ci='sd', palette='Paired', label=df['exp_name'][0][13:-11])
plt.show()


# plot pareto fronts
for pfdf in pf_dfs:
    ax = sns.scatterplot(data=pfdf, y='Max Distance', x='Max Hover Time', label=pfdf['exp_name'][0][13:-11])
    ax = sns.lineplot(data=pfdf, y='Max Distance', x='Max Hover Time')
# ax.set_aspect('equal')
plt.show()