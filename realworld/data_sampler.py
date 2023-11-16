""" dataを加工するモジュール"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import numpy as np
import os
import pandas as pd
import math

base_route = os.getcwd()
data_route = 'datasets'

def sample_artificial_data(data_type, num_contexts,context_dim):
    path_1 = os.path.join(base_route, data_route, 'artificial_feature_data_' + data_type[11:] + '.csv')
    df_x = pd.read_csv(path_1)

    ind = np.random.choice(range(df_x.shape[0]), num_contexts, replace=True)# num_contextsの数だけランダムに抽出
    df = df_x.iloc[ind,:].values

    exp_rewards = df_x.iloc[ind, context_dim:].values#報酬箇所

    """最適期待報酬と最適行動を求める"""
    opt_actions = np.argmax(exp_rewards, axis=1)#ユーザの評価値の中で1番大きい行動をopt_actionとする
    opt_rewards = np.array([exp_rewards[i,a] for i, a in enumerate(opt_actions)])#iはindex,aは要素(行動)
    opt_values = (opt_rewards, opt_actions)

    return df, opt_values, exp_rewards

def sample_mixed_artificial_data(data_type, num_contexts,context_dim):
    path_1 = os.path.join(base_route, data_route, 'artificial_feature_data_mixed_' + data_type[17:] + '.csv')
    df_x = pd.read_csv(path_1)

    ind = np.random.choice(range(df_x.shape[0]), num_contexts, replace=True)# num_contextsの数だけランダムに抽出
    df = df_x.iloc[ind,:].values

    exp_rewards = df_x.iloc[ind, context_dim:].values#報酬箇所

    """最適期待報酬と最適行動を求める"""
    opt_actions = np.argmax(exp_rewards, axis=1)#ユーザの評価値の中で1番大きい行動をopt_actionとする
    opt_rewards = np.array([exp_rewards[i,a] for i, a in enumerate(opt_actions)])#iはindex,aは要素(行動)
    opt_values = (opt_rewards, opt_actions)

    return df, opt_values, exp_rewards




