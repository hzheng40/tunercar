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
parser.add_argument('--npz_dir', type=str, default='../es/iccps_runs/npzs/')
parser.add_argument('--heatmap_idx', type=int, default=-1)
args = parser.parse_args()

# plot digit change heat map
# y axis (pixels) will be number of different selections at each element
# x axis will be iterations
batch_size = 128
heatmap_exp = args.exp_name[args.heatmap_idx]
data = np.load(args.npz_dir + heatmap_exp + '.npz')
vectors = data['vectors']
num_vec, num_slots = vectors.shape
num_iter = int(num_vec / batch_size)
heatmap_mat = np.zeros((num_slots, num_iter))
for j in range(num_iter):
    block = vectors[j*batch_size:(j+1)*batch_size, :].round(decimals=5)
    for k in range(block.shape[1]):
        _, count = np.unique(block[:, k], return_counts=True)    
        heatmap_mat[k, j] = count.shape[0]

sns.heatmap(heatmap_mat.astype(int), annot=True, fmt='d')
plt.show()

assert False

# list of dfs for all data
dfs = []

for exp in args.exp_name:
    print('Extracting: ' + exp)
    # loading
    data = np.load(args.npz_dir + exp + '.npz')
    # optim = ng.optimizers.base.Optimizer.load(args.pkl_dir + exp + '_optim.pkl')
    # batch_size = optim.num_workers
    # # batch_size = 24
    # if optim.name == 'CMA':
    #     batch_size = optim._popsize
    # elif optim.name == 'NoisyDE' or optim.name == 'TwoPointsDE' or optim.name[:21] == 'DifferentialEvolution' or optim.name[:13] == 'ConfiguredPSO' or optim.name == 'PSO':
    #     batch_size = len(optim.population)
    batch_size = 128
    scores_all = np.sum(data['scores'], axis=1)
    scores_all = np.nan_to_num(scores_all)
    # params_all = data['params']
    iterations = int(scores_all.shape[0]/batch_size)

    # set up dataframes
    df = pd.DataFrame(columns=['exp_name', 'Iterations', 'Score (Higher is better)'])
    # cov_norm_df = pd.DataFrame(columns=['exp_name', 'Iterations', 'cov_norm'])

    # load data into dataframes
    for i in range(iterations):
        # current_gen_times = []
        for j in range(batch_size):
            current_score = scores_all[i*batch_size+j]
            # current_param = params_all[:, i*batch_size+j]
            # if not np.isnan(current_score):
                # current_gen_times.append(current_score)
            df = df.append({'exp_name': exp, 'Iterations': i, 'Score (Higher is better)': current_score}, ignore_index=True)
        # if len(current_gen_times) > 0:
        #     cov_norm_df.append({'exp_name': args.exp_name, 'Iterations': i, 'cov_norm': current_score})
    dfs.append(df)

# plot laptimes
sns.set_style('white')
sns.set_style('ticks')
sns.set_context('poster')
palette = sns.color_palette("mako_r", len(args.exp_name))

print('Best scores from each experiment:')

for df in dfs:
    max_score = df['Score (Higher is better)'].max()
    max_idx = df['Score (Higher is better)'].idxmax()
    print(df['exp_name'][0] + ': ' + str(max_score) + ' at index: ' + str(max_idx))
    ax = sns.lineplot(data=df, x='Iterations', y='Score (Higher is better)', ci='sd', palette='Paired', label=df['exp_name'][0][31:-11])
# ax = sns.lineplot(data=df, x='Iterations', y='Score (Higher is better)', ci='sd', palette='Paired')
# ax.set_xscale('log')
# plt.xscale('log')
plt.show()