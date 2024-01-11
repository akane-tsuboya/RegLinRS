# 概要
これは「文脈的採餌行動における逐次的意思決定モデル」のプログラムコードです。\
実験設定などの詳細は論文を参照してください。\
コードを使用する際は、以下を引用してください。

```bash
@article{Tsuboya2024,
title = {文脈的採餌行動における逐次的意思決定モデル},
author = {坪谷 朱音 and 甲野 佑 and 高橋 達二},
journal = {知能と情報},
year = {2024},
volume = {36},
number = {1},
pages = {501--512}
}
```
 
# Requirement
* Python>=3.7
* numpy
* tensorflow
* typing
* pandas
* matplotlib
* scikit-learn
* scipy
* seaborn

# Usage
## (0) 環境作成
各種必要なパッケージをインストールする。
```bash
conda env create --file config.yml
```

## (1) 人工データセットを生成
最大報酬期待値 $z=0.75$ ・選択肢の数 $K=8$ ・特徴ベクトルの次元数 $d=128$ ・データ数 $100,000$ の人工データセットを生成する。\
※ なお、one-hot な特徴ベクトル $h_t$ を生成後、そこにノイズを加えた特徴ベクトル $c_t$ を作成するため引数 False → True で 2 回実行する必要がある。
```bash
python artificial_data_generator.py False 0.75
python artificial_data_generator.py True 0.75
```
$z$ 以外の設定は artificial_data_generator.py 内で変更可能。\
作成したデータセットは ./datasets/ に格納される。

## (2) 実行
```bash
python real_world_main.py
```
用いるアルゴリズムや各種パラメータなどシミュレーション設定はreal_world_main.py内で変更可能。\
基本的な結果は ./csv/ ・生存率の計算に必要な結果は ./csv_reward_count/ ・報酬期待値が同じ腕ごとの平均二乗誤差の結果は ./csv_mse/ に格納される。

## (3)プロット
### 基本的な結果
```bash
python plot/plot.py csv/結果が入っているディレクトリ
```
実行した結果は ./png/ に保存される。\
出力される図は

* regrets.png
* rewards.png
* greedy_rate.png
    * エージェントが最適だと思う行動を選択した割合(greedy率)
* accuracy.png
* errors.png
    * 平均誤差率 (MPE)
* entropy_of_reliability
    * RegLinRS の信頼度の推定値に対するエントロピー
    また,各アルゴリズムの 1 sim あたりの平均実行時間は 1 sim time: [ ] でprint される。

### 真の報酬期待値が同じ腕ごとのMSEの平均の結果のプロット
```bash
python plot/plot_mse.py csv_mse/結果が入っているディレクトリ
```
実行した結果はpng_mseディレクトリに保存される。

### エネルギー目標量を獲得するのにかかった step 数
全 step数 の 6 割を生存ラインとした時、それを超えるまでに何 step かかったかを計算。
```bash
python plot/survival_rate.py csv_reward_count/結果が入っているディレクトリ  0.6
```
./png/survival_rate/ に csv と dict で結果が格納される。\
次に、計算した step 数と常に 1 番良い餌を獲得し続けた場合の差分を可視化。
```bash
python plot/plot_survival_hako.py png/survival_rate/結果が入っているディレクトリ/result_dict.csv
```
実行した結果は ./png_survival_rate/ に保存される。

# Note
./datasets/ に任意のデータセットを追加することも可能。データセットの情報を管理する ./realworld/setup_context.py とデータを取得する ./realworld/data_sampler.py に追記する必要あり。

# Author
東京電機大学先端科学技術研究科\
坪谷朱音(Tsuboya Akane)\
