import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset
import seaborn as sns
import nevergrad as ng
import argparse

# args
def str2bool(v):
    if v.lower() == 'true':
        return True
    else:
        return False
parser = argparse.ArgumentParser()
parser.add_argument('--exp_name', type=str, required=True, nargs='+')
parser.add_argument('--npz_dir', type=str, default='../es/tunercar_runs/npzs/')
parser.add_argument('--pkl_dir', type=str, default='../es/tunercar_runs/optims_pkl/')
parser.add_argument('--plot_spline_coverage', type=str2bool, default=False)
args = parser.parse_args()

# list of dfs for all data
dfs = []

for exp in args.exp_name:
    print('Extracting: ' + exp)
    # loading
    data = np.load(args.npz_dir + exp + '.npz')
    optim = ng.optimizers.base.Optimizer.load(args.pkl_dir + exp + '_optim.pkl')
    batch_size = optim.num_workers
    # batch_size = 24
    if optim.name == 'CMA':
        batch_size = optim._popsize
    elif optim.name == 'NoisyDE' or optim.name == 'TwoPointsDE' or optim.name[:21] == 'DifferentialEvolution' or optim.name[:13] == 'ConfiguredPSO' or optim.name == 'PSO':
        batch_size = len(optim.population)
    times_all = data['lap_times']
    # params_all = data['params']
    iterations = int(times_all.shape[0]/batch_size)

    # set up dataframes
    df = pd.DataFrame(columns=['exp_name', 'Iterations', 'Score (Lower is better)'])
    # cov_norm_df = pd.DataFrame(columns=['exp_name', 'Iterations', 'cov_norm'])

    # load data into dataframes
    for i in range(iterations):
        # current_gen_times = []
        for j in range(batch_size):
            current_time = times_all[i*batch_size+j]
            # current_param = params_all[:, i*batch_size+j]
            if current_time < 99999 and current_time > 10:
                # current_gen_times.append(current_time)
                df = df.append({'exp_name': exp, 'Iterations': i, 'Score (Lower is better)': current_time}, ignore_index=True)
        # if len(current_gen_times) > 0:
        #     cov_norm_df.append({'exp_name': args.exp_name, 'Iterations': i, 'cov_norm': current_time})
    dfs.append(df)

# plot laptimes
sns.set_style('white')
sns.set_style('ticks')
sns.set_context('poster')
palette = sns.color_palette("mako_r", len(args.exp_name))

print('Best laptime (2 laps) from each experiment:')

for df in dfs:
    min_time = df['Score (Lower is better)'].min()
    min_idx = df['Score (Lower is better)'].idxmin()
    print(df['exp_name'][0] + ': ' + str(min_time) + ' at index: ' + str(min_idx))
    ax = sns.lineplot(data=df, x='Iterations', y='Score (Lower is better)', ci='sd', palette='Paired', label=df['exp_name'][0][29:-11])
# ax = sns.lineplot(data=df, x='Iterations', y='Score (Lower is better)', ci='sd', palette='Paired')
# ax.set_xscale('log')
# plt.xscale('log')
plt.show()


# plot spline coverage
