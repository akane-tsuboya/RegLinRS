"""MSEをplot"""
import os
import sys
import datetime
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


"""定数系"""
n_steps = 100000
name = ['RegLinRS', 'LinTS', 'LinUCB', 'Greedy']

"""引数確認"""
args = sys.argv
if len(args) <= 1:
    print('Usage: python plot.py [path of directory where csv is stored]')
    sys.exit()

"""csvデータの取得"""
directory = os.listdir(args[1])
files = [f for f in directory if os.path.isfile(os.path.join(args[1], f))]
print(directory)
policy_names = [file_name[:-4].replace('_', ' ') for file_name in files]
f = [files[policy_names.index(i)] for i in name]
policy_names = [file_name[:-4].replace('_', ' ') for file_name in f]

result_list = []
for file_name in f:
    df = pd.read_csv(args[1] + '/' + file_name)
    dict_type = df.to_dict(orient='list')
    result_list.append(dict_type)
    #print(result_list)
result_list = pd.DataFrame(result_list)
result_list = result_list.rename(columns={'0': 'Arm 1', '1': 'Arm 2', '2': 'Arm 3', '3': 'Arm 4', '4': 'Arm 5', '5': 'Arm 6', '6': 'Arm 7', '7': 'Arm 8'})

"""結果データのプロット"""
mpl.rcParams['axes.xmargin'] = 0
mpl.rcParams['axes.ymargin'] = 0
plt.rcParams["font.family"] = "DejaVu Serif"

time_now = datetime.datetime.now()
results_dir = 'png_mse/{0:%Y%m%d%H%M}/'.format(time_now)
os.makedirs(results_dir, exist_ok=True)

for i, policy_name in enumerate(policy_names):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)
    for j, data_name in enumerate(['Arm 1', 'Arm 2', 'Arm 3', 'Arm 4']):
        #  'Arm 4', 'Arm 5', 'Arm 6', 'Arm 7' はなし
        cmap = plt.get_cmap("tab10")
        b = np.ones(2000) / 2000.0
        y3 = np.convolve(result_list.at[i, data_name], b, mode='same')  # 移動平均
        ax.plot(np.linspace(1, n_steps, num=n_steps), y3, label=data_name, linewidth=3, alpha=0.8)
    ax.set_xlabel('Step', fontsize=30)
    ax.xaxis.offsetText.set_fontsize(30)
    ax.ticklabel_format(style="sci", axis="x", scilimits=(4,4))   # 10^3単位の指数で表示する。
    ax.set_ylabel('Squared error',fontsize=35)
    ax.set_ylim([-5.0, 100.0])
    ax.yaxis.offsetText.set_fontsize(30)
    ax.spines["top"].set_linewidth(2)
    ax.spines["left"].set_linewidth(2)
    ax.spines["bottom"].set_linewidth(2)
    ax.spines["right"].set_linewidth(2)
    leg = ax.legend(loc='upper right', fontsize=30)
    plt.tick_params(labelsize=30)
    ax.text(0.5, -0.23, policy_name, ha='center', transform=ax.transAxes, fontsize=35)
    ax.grid(alpha=0.5,color = "lightgray", linestyle="--")
    fig.savefig(results_dir + policy_name, bbox_inches='tight',
                pad_inches=0)

plt.clf()
