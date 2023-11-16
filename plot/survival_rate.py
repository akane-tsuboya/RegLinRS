import os
import sys
import datetime
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import statistics
import yaml
import csv

def my_index(l, x, default=False):
    if x in l:
        return l.index(x)
    else:
        return default

def first_survival_result(df):
    """1日単位で合計、sim数で平均"""
    df_result = pd.DataFrame()
    len_df = len(df.columns)
    len_index = len(df.index)
    df_day_reward = np.zeros(len_index)
    first_index = [0]*len_index
    #全stepを漸化式的に足し合わせる、超えるまで、全simに対して同じ動作をする
    for i in range(len_df):
        df_day = df.iloc[:, i]
        #print(df_day)
        df_day_reward = np.add(df_day_reward,df_day)
        #print(df_day_reward)
        #df_day_reward = df_day.sum(axis=1)
        df_result=pd.concat([df_result, df_day_reward], axis=1)
        if all(df_day_reward >= total_survival):
            survival = np.where(df_result >= total_survival, 1, 0)
            break
        

    #合計から生存したかいなか出す
    #survival = np.where(df_result >= total_survival, 1, 0)
    #sim数で平均。独立の生存率
    print(df_result)
    print(survival[0])
    survival = survival.tolist()
    for i in range(len_index):
        first_index[i] = my_index(survival[i],1,-1)

    first_index = [idx for idx in first_index if idx != -1]
    #first_index =np.nonzero(survival==1)
    print(first_index)
    
    #survival_result_tmp = survival.mean(axis=0)
    #本来の生存率を出す(前の生存率にかける)
    #survival_result = np.zeros(len(survival_result_tmp))
    #survival_result[0] = survival_result_tmp[0]

    #for j in range(len(survival_result_tmp)-1):
    #    survival_result[j+1] = survival_result[j] * survival_result_tmp[j+1]


    return first_index
        

"""定数系"""
n_steps = 100000
name = ['RegLinRS', 'LinTS', 'LinUCB', 'Greedy']#ここを書き換える

"""引数確認"""
args = sys.argv
if len(args) <= 1:
    print('Usage: python plot.py [path of directory where csv is stored]')
    sys.exit()

""""生存率定義"""
survival_rate = float(args[2])

"""1日の単位"""
#day = int(args[3])

"""csvデータの取得"""
directory = os.listdir(args[1])
name_rate = args[1]
target = '_'
idx = name_rate.rfind(target)
# opt = float(name_rate[idx+1:idx+4])
# opt += 0.05
# total_opt = opt*n_steps
# total_survival = survival_rate * total_opt
total_survival = survival_rate * n_steps
#total_survival=10 #チェック用
print(total_survival)

files = [f for f in directory if os.path.isfile(os.path.join(args[1], f))]
print(directory)
policy_names = [file_name[:-4].replace('_', ' ') for file_name in files]
f = [files[policy_names.index(i)] for i in name]
policy_names = [file_name[:-4].replace('_', ' ') for file_name in f]

result_list = []
for file_name in f:
    df = pd.read_csv(args[1] + '/' + file_name, index_col=0)
    df = first_survival_result(df)
    #df = df.tolist()
    #dict_type = df.to_dict(orient='list')
    result_list.append(df)
#result_list = pd.DataFrame(result_list)
print(result_list)#生存率


"""結果データのプロット"""
mpl.rcParams['axes.xmargin'] = 0
mpl.rcParams['axes.ymargin'] = 0

time_now = datetime.datetime.now()
results_dir = 'png/survival_rate/{0:%Y%m%d%H%M}/'.format(time_now)
os.makedirs(results_dir, exist_ok=True)

result_dict={}
for j, policy_name in enumerate(policy_names):
    print(result_list[j])
    mean = statistics.mean(result_list[j])
    median = statistics.median(result_list[j])
    p_var = statistics.pvariance(result_list[j])
    p_stdev = statistics.pstdev(result_list[j])
    var = statistics.variance(result_list[j])
    stdev = statistics.stdev(result_list[j])
    result_dict[policy_name] = {'mean':mean,'median':median,'var':var,'stdev':stdev,'p_var':p_var,'p_stdev':p_stdev}
with open(results_dir+'result_dict.yml','w')as f:
    yaml.dump(result_dict, f, default_flow_style=False, allow_unicode=True)

with open(results_dir+'result_dict.csv', 'w') as f:
    writer = csv.writer(f)
    for i, row in zip(policy_names, result_list):
        writer.writerow([i] + row)
    #writer.writerows(result_list)
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111)
ax.boxplot(result_list,labels=policy_names,whis=0.5)
#for j, policy_name in enumerate(policy_names):
    #cmap = plt.get_cmap("tab10")
    #ax.plot(np.linspace(1, n, num=n),result_list[j],label=policy_name, linewidth=3, alpha=0.8)
    #print(policy_name)
    #ax.boxplot(result_list[j],labels=policy_name)
#sns.boxplot(x='variable', y='value', data=result_list, showfliers=False, ax=ax)
#sns.stripplot(x='variable', y='value', data=result_list, jitter=True, color='black', ax=ax)
ax.set_ylim([60000,67000])

#ax.set_xlabel(fontsize=23)
#ax.set_ylabel(fontsize=23)
#leg = ax.legend(loc='lower right', fontsize=23)
#plt.tick_params(labelsize=23)
#ax.grid(alpha=0.8,color = "gray", linestyle="--")

fig.savefig(results_dir + 'survival_rate', bbox_inches='tight',
                pad_inches=0)

plt.clf()
