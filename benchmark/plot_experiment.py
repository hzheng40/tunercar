import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset
from analyze_tunercar_runs import parse_file_name
import seaborn as sns
import nevergrad as ng
import argparse

# args
parser = argparse.ArgumentParser()
parser.add_argument('--exp_name', type=str, required=True)
parser.add_argument('--npz_dir', type=str, default='../es/tunercar_runs/npzs/')
parser.add_argument('--pkl_dir', type=str, default='../es/tunercar_runs/optims_pkl/')
args = parser.parse_args()

# loading
npz_f_name = args.exp_name + '.npz'
data = np.load(args.npz_dir + npz_f_name)
optim = ng.optimizers.base.Optimizer.load(args.pkl_dir + args.exp_name + '_optim.pkl')
popsize = optim._popsize
times_all = data['lap_times']
params_all = data['params']
gen = int(times_all.shape[0]/popsize)
map_name, optim_method, popsize, budget = parse_file_name(npz_f_name)
print(f"Minimum lap time: {np.min(data['lap_times']):.4f}")

# set up dataframes
df = pd.DataFrame(columns=['exp_name', 'gen', 'score'])
# cov_norm_df = pd.DataFrame(columns=['exp_name', 'gen', 'cov_norm'])

# load data into dataframes
for i in range(gen):
    # current_gen_times = []
    for j in range(popsize):
        current_time = times_all[i*popsize+j]
        # current_param = params_all[:, i*popsize+j]
        if current_time < 99999:
            # current_gen_times.append(current_time)
            df = df.append({'exp_name': args.exp_name, 'gen': i, 'score': current_time}, ignore_index=True)
    # if len(current_gen_times) > 0:
    #     cov_norm_df.append({'exp_name': args.exp_name, 'gen': i, 'cov_norm': current_time})


# plotting
sns.set_style('white')
sns.set_style('ticks')
sns.set_context('poster')
palette = sns.color_palette("mako_r", 5)
ax = sns.lineplot(data=df, x='gen', y='score', ci='sd', palette='Paired')
# ax.set_xscale('log')
plt.title('Initial Velocity 2 m/s')
# plt.xlabel('generation')
# plt.ylabel('lap times (s)')
plt.show()