import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset
import seaborn as sns
import os
# import sys
# sys.path.append('../results/')
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--results_path', type=str, required=True)
parser.add_argument('--save_csv', type=int, default=1)
parser.add_argument('--csv_path', type=str, default='../results/processed_population_time.csv')
parser.add_argument('--cov_norm_csv_path', type=str, default='../results/processed_population_covnorm.csv')
parser.add_argument('--reprocess', type=int, default=0)
parser.add_argument('--clean_score_csv', type=int, default=0)
parser.add_argument('--clean_score_csv_path', type=str, default='../results/processed_population_time_cleaned.csv')
args = parser.parse_args()

if args.clean_score_csv:
    df = pd.read_csv(args.csv_path, index_col=0)
    cleaned_df = pd.DataFrame(columns=['population_size', 'percentile', 'generation', 'score'])
    for index, row in df.iterrows():
        exp_name = row['experiment']
        exp_split = exp_name.split('e')
        pop_size = int(exp_split[1][:-1])
        percentile = float(exp_split[-1])
        new_row = {'population_size': pop_size, 'percentile': percentile, 'generation': row['generation'], 'score': row['score']}
        cleaned_df = cleaned_df.append(new_row, ignore_index=True)
    cleaned_df.to_csv(args.clean_score_csv_path)

else:
    cleaned_df = pd.read_csv(args.clean_score_csv_path, index_col=0)

if args.reprocess:
    npz_list = []
    all_time_hist_list = []
    exp_name_list = []
    cov_norm_list = []
    for filename in os.listdir(args.results_path):
        if filename.endswith('.npz'):
            exp_name = os.path.splitext(filename)[0]
            hi = np.load(args.results_path+filename)
            npz_list.append(hi)
            all_time_hist_list.append(hi['time_list_hist'])
            cov_norm_hist = hi['cov_norm_hist']
            cov_norm_hist = cov_norm_hist[cov_norm_hist>0]
            cov_norm_list.append(cov_norm_hist)
            exp_name_list.append(exp_name)
        else:
            continue

    df = pd.DataFrame(columns=['exp_name', 'gen', 'score'])
    cov_norm_df = pd.DataFrame(columns=['exp_name', 'gen', 'cov_norm'])

    for k in range(len(npz_list)):
        exp_name = exp_name_list[k]
        print('Processing run', exp_name)

        # cov norm hist of a run
        cov_norm_hist = cov_norm_list[k]
        for g in range(cov_norm_hist.shape[0]):
            cov_norm_df = cov_norm_df.append({'exp_name': exp_name, 'gen': g, 'cov_norm': cov_norm_hist[g]}, ignore_index=True)
        print('cov norm loop done')

        # time history of the full pop
        time_hist = all_time_hist_list[k]
        for i in range(time_hist.shape[0]):
            for j in range(time_hist.shape[1]):
                if time_hist[i, j] < 100 and time_hist[i, j] > 0:
                    df = df.append({'exp_name': exp_name, 'gen': i, 'score':time_hist[i, j]}, ignore_index=True)
        print('time hist loop done')
        
    if args.save_csv:
        df.to_csv(args.csv_path)
        cov_norm_df.to_csv(args.cov_norm_csv_path)

else:
    df = pd.read_csv(args.csv_path, index_col=0)
    cov_norm_df = pd.read_csv(args.cov_norm_csv_path, index_col=0)

sns.set_style('white')
sns.set_style('ticks')
sns.set_context('poster')
ax = sns.lineplot(data=df, x='generation', y='score', hue='experiment', ci='sd', palette='Paired')
ax.set_xscale('log')
plt.show()


sns.set_style('white')
sns.set_style('ticks')
sns.set_context('poster')
palette = sns.color_palette("mako_r", 5)
ax = sns.lineplot(data=cleaned_df[cleaned_df['population_size']==1000], x='generation', y='score', hue='percentile', ci='sd', palette=palette, legend='brief')
ax.set_xscale('log')
axins = zoomed_inset_axes(ax, zoom=7.5, loc='upper center')
axins.set_xlim(11.0, 170)
axins.set_ylim(14.6, 16.2)
sns.lineplot(data=cleaned_df[cleaned_df['population_size']==1000], x='generation', y='score', hue='percentile', ci='sd', palette=palette, ax=axins, legend=False)
# axins.yaxis.get_major_locator().set_params(nbins=7)
# axins.xaxis.get_major_locator().set_params(nbins=7)
axins.set_aspect(60)
axins.axhline(15.7, ls='--', c=palette[0])
axins.axhline(15.3, ls='--', c=palette[1])
axins.axhline(15.08, ls='--', c=palette[2])
axins.axhline(15.26, ls='--', c=palette[3])
axins.axhline(14.8, ls='--', c=palette[4])
# plt.setp(axins.get_xticklabels(), visible=False)
# plt.setp(axins.get_yticklabels(), visible=False)
mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")
# ax.set_yscale('log')
plt.show()



sns.set_style('white')
sns.set_style('ticks')
sns.set_context('poster')
palette = sns.color_palette("mako_r", 6)
ax0 = sns.lineplot(data=cleaned_df[cleaned_df['percentile']==0.01], x='generation', y='score', hue='population_size', ci='sd', palette=palette, legend='brief')
ax0.set_xscale('log')
axins0 = zoomed_inset_axes(ax0, zoom=7, loc='upper center')
axins0.set_xlim(16.0, 50.0)
axins0.set_ylim(13.8, 16.0)
sns.lineplot(data=cleaned_df[cleaned_df['percentile']==0.01], x='generation', y='score', hue='population_size', ci='sd', palette=palette, ax=axins0, legend=False)
axins0.set_aspect(13)
axins0.axhline(15.69, ls='--', c=palette[0])
axins0.axhline(14.74, ls='--', c=palette[1])
axins0.axhline(14.44, ls='--', c=palette[2])
axins0.axhline(14.35, ls='--', c=palette[3])
axins0.axhline(14.11, ls='--', c=palette[4])
axins0.axhline(14.08, ls='--', c=palette[5])
mark_inset(ax0, axins0, loc1=2, loc2=4, fc="none", ec="0.5")
# ax.set_yscale('log')
plt.show()


sns.set_style('white')
sns.set_style('ticks')
sns.set_context('poster')
ax1 = sns.lineplot(data=cov_norm_df, x='generation', y='covariance_norm', hue='experiment', palette='Paired')
ax1.axhline(0.01, ls='--')
ax1.text(115,0.007, "Termination Threshold")
# ax1.set_xscale('log')
ax1.set_yscale('log')
plt.show()