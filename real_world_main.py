# -*- coding: utf-8 -*-
"""実行モジュール"""

from sim.real_sim import ContextualBanditSimulator
from bandit.contextual_bandit import ContextualBandit
from policy.linucb import LinUCB
from policy.linear_full_posterior_sampling_fixed import LinearTSfixed
from policy.regional_linrs import RegionalLinRS
from policy.uniform import Uniform
from policy.greedy import Greedy
from realworld.setup_context import ContextData

def main():
    """main

    Args:
        N_SIMS(int) : sim数
        N_STEPS(int) : step数
        n_contexts(int) : データの個数(step数)
        N_ARMS(int) : 行動の選択肢
        N_FEATURES(int) : 特徴量の次元数

        aleph(float) : 満足化基準値

        policy_list(list[str]) : 検証する方策クラスをまとめたリスト
        bandit : バンディット環境
        bs : バンディットシュミレーター
        data_type(str) : データの種類[artificial_0.55, artificial_0.75, artificial_0.95, 
                                        mixed_artificial_0.55, mixed_artificial_0.75, mixed_artificial_0.95]
    """

    n_contexts =100000
    data_type = 'mixed_artificial_0.75'
    num_actions, context_dim = ContextData.get_data_info(data_type)

    N_SIMS = 100
    N_STEPS = n_contexts
    N_ARMS = num_actions
    N_FEATURES = context_dim

    #artificial
    policy_list = [RegionalLinRS(n_arms=N_ARMS, n_features=N_FEATURES, warmup=10, batch_size=20, aleph=0.6, k = 50, memory_capacity = 10000, zeta = 0.008, epsilon = 0.0001)
                , LinearTSfixed(n_arms=N_ARMS, n_features=N_FEATURES, warmup=10, batch_size=20, lambda_prior=0.25, a0=6, b0=6)
                , LinUCB(n_arms=N_ARMS, n_features=N_FEATURES, warmup=10, batch_size=20, alpha=0.1)
                , Greedy(n_arms=N_ARMS, n_features=N_FEATURES, warmup=10, batch_size=20,n_steps=N_STEPS)]
  
    bandit = ContextualBandit(n_arms=N_ARMS, n_features=N_FEATURES, n_contexts=N_STEPS, data_type=data_type)
    bs = ContextualBanditSimulator(policy_list=policy_list, bandit=bandit, n_sims=N_SIMS,
                         n_steps=N_STEPS, n_arms=N_ARMS, n_features=N_FEATURES, data_type=data_type)

    bs.run()

main()
