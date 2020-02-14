import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
sys.path.append('../results/')
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--result_npz', type=str, required=True)
args = parser.parse_args()

hi = np.load(args.result_npz)
hi = hi['time_list_hist']
df = pd.DataFrame(columns=['gen', 'score'])

for i in range(hi.shape[0]):
    for j in range(hi.shape[1]):
        if hi[i, j] < 100 and hi[i, j] > 0:
            df = df.append({'gen': i, 'score':hi[i, j]}, ignore_index=True)

sns.set_style('dark')
ax = sns.lineplot(data=df, x='gen', y='score', ci='sd', markers=True)
plt.show()