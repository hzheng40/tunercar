import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
# import sys
# sys.path.append('../results/')
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--results_path', type=str, required=True)
parser.add_argument('--save_csv', type=int, default=1)
parser.add_argument('--csv_path', type=str, default='../results/processed_population_time.csv')
parser.add_argument('--reprocess', type=int, default=1)
args = parser.parse_args()

if args.reprocess:
    npz_list = []
    all_time_hist_list = []
    exp_name_list = []
    for filename in os.listdir(args.results_path):
        if filename.endswith('.npz'):
            exp_name = os.path.splitext(filename)[0]
            hi = np.load(args.results_path+filename)
            npz_list.append(hi)
            all_time_hist_list.append(hi['time_list_hist'])
            exp_name_list.append(exp_name)
        else:
            continue

    df = pd.DataFrame(columns=['exp_name', 'gen', 'score'])

    for k in range(len(npz_list)):
        exp_name = exp_name_list[k]
        print('Processing run', exp_name)
        time_hist = all_time_hist_list[k]
        for i in range(time_hist.shape[0]):
            for j in range(time_hist.shape[1]):
                if time_hist[i, j] < 100 and time_hist[i, j] > 0:
                    df = df.append({'exp_name': exp_name, 'gen': i, 'score':time_hist[i, j]}, ignore_index=True)
    if args.save_csv:
        df.to_csv(args.csv_path)

else:
    df = pd.read_csv(args.csv_path, index_col=0)

sns.set_style('white')
ax = sns.lineplot(data=df, x='gen', y='score', hue='exp_name', ci='sd', palette='pastel')
plt.show()