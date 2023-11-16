"""step数の差分をregret風にとって箱ひげ図をplot"""
import os
import sys
import datetime
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

"""定数系"""
n_steps = 100000
n_sims = 100
name = ['RegLinRS', 'LinTS', 'LinUCB', 'Greedy']#ここを書き換える

"""引数確認"""
args = sys.argv
if len(args) <= 1:
    print('Usage: python plot.py [path of directory where csv is stored]')
    sys.exit()


result_list = []
survival_rate = float(args[2])
total_survival_line = survival_rate * n_steps
opt_reward = total_survival_line/0.75
df = pd.read_csv(args[1], index_col=0, header=None)
print(df)
print(df.T)

line_diff = df.T-opt_reward


"""結果データのプロット"""
mpl.rcParams['axes.xmargin'] = 0
mpl.rcParams['axes.ymargin'] = 0
plt.rcParams["font.size"] = 23
plt.rcParams["font.family"] = "DejaVu Serif"
plt.rcParams['figure.subplot.bottom'] = 0.06
plt.rcParams['figure.subplot.top'] =0.98
plt.rcParams['figure.subplot.left'] = 0.18
plt.rcParams['figure.subplot.right'] = 0.98


time_now = datetime.datetime.now()
results_dir = 'png_survival_rate/{0:%Y%m%d%H%M}/'.format(time_now)
os.makedirs(results_dir, exist_ok=True)

c_list = []
cmap = plt.get_cmap("tab10")

fig = plt.figure(figsize=(9, 16))
ax = fig.add_subplot(111)

sns.boxplot(data=line_diff, showfliers=False, ax=ax)
sns.stripplot(data=line_diff, jitter=.4, color='black', size=4,alpha=0.7, ax=ax)

ax.spines["top"].set_linewidth(2)
ax.spines["left"].set_linewidth(2)
ax.spines["bottom"].set_linewidth(2)
ax.spines["right"].set_linewidth(2)

ax.set_xlabel('Algorithms',fontsize=25)
ax.set_ylabel('Step loss',fontsize=25)
ax.set_ylim([0,10000])

ax.grid(alpha=0.5,color = "lightgray", linestyle="--")

fig.savefig(results_dir + 'survival_rate', 
                pad_inches=0)

plt.clf()
