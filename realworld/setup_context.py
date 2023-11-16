"""データのセットアップをおこなうモジュール"""
from realworld.data_sampler import sample_artificial_data
from realworld.data_sampler import sample_mixed_artificial_data

class ContextData(object):
    """特徴のデータの基本情報をセットするクラス"""
    def sample_data(data_type, num_contexts=None):
        """データセットの最適行動、報酬期待値をセット
        Args:
            data_type(str):データセット名
            num_contexts(int):データ数
        Returns:
            dataset:加工済みデータ
            opt_rewards(int, float):最適報酬
            opt_actions(int):最適行動
            exp_rewards(int, float):報酬
        Raises:
            DATA NAME ERROR: data_typeがどれも当てはまらない場合
        """

        if data_type.startswith('artificial'):
            num_actions = 8
            context_dim = 128
            dataset, opt_artificial, exp_rewards = sample_artificial_data(data_type, num_contexts, context_dim)
            opt_rewards, opt_actions = opt_artificial
        elif data_type.startswith('mixed_artificial'):
            num_actions = 8
            context_dim = 128
            dataset, opt_artificial, exp_rewards = sample_mixed_artificial_data(data_type, num_contexts, context_dim)
            opt_rewards, opt_actions = opt_artificial
        else:
            print("DATA NAME ERROR.")

        return dataset, opt_rewards, opt_actions, exp_rewards

    def get_data_info(data_type):
        """dataの基本情報の取得
        Returns:
            num_actions(int):行動数
            context_dim(int):特徴量の次元
        """
        if data_type.startswith('artificial'):
            num_actions = 8
            context_dim = 128
        elif data_type.startswith('mixed_artificial'):
            num_actions = 8
            context_dim = 128
        return num_actions, context_dim
