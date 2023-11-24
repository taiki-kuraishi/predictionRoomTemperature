# %% [markdown]
# 予測モデルの構築
# リサンプリングした resampled_switchBot_data.csv を使用
#

# %%
# import paskages
import pandas as pd
import matplotlib.pyplot as plt

# %%
# read csv
niigata_data = pd.read_csv(
    "../data/formatted_niigata_weather_data.csv", encoding="utf-8", index_col=None
)
switchBot_data = pd.read_csv(
    "../data/resampled_switchBot_data.csv", encoding="utf-8", index_col=None
)

# %%
niigata_data.head()

# %%
switchBot_data.head()

# %%
# datetime型に変換
niigata_data["N_datetime"] = pd.to_datetime(
    niigata_data["N_datetime"], format="%Y-%m-%d %H:%M:%S"
)
switchBot_data["S_datetime"] = pd.to_datetime(
    switchBot_data["S_datetime"], format="%Y-%m-%d %H:%M:%S"
)

# %%
# datetimeのカラム名の統一
niigata_data = niigata_data.rename(columns={"N_datetime": "datetime"})
switchBot_data = switchBot_data.rename(columns={"S_datetime": "datetime"})

# %%
# niigata_dataとswitchBot_dataをdatetimeをキーにして左結合
niigata_data = niigata_data.set_index("datetime")
switchBot_data = switchBot_data.set_index("datetime")
merged_data = niigata_data.join(switchBot_data, how="left")
merged_data.head()

# %%
# indexをリセット
merged_data = merged_data.reset_index()
merged_data.head()

# %%
# 横軸がdatetime、縦軸が気温のグラフを作成
plt.figure(figsize=(20, 10))
plt.plot(merged_data["datetime"], merged_data["N_temp"], label="Niigata_temp")
plt.plot(merged_data["datetime"], merged_data["S_temp"], label="SwitchBot_temp")
plt.legend(loc="upper left", fontsize=20)
plt.ylabel("temp")
plt.tight_layout()
plt.show()

# %%
# datetimeを月と日と時間に分割
merged_data["month"] = merged_data["datetime"].dt.month
merged_data["day"] = merged_data["datetime"].dt.day
merged_data["hour"] = merged_data["datetime"].dt.hour
merged_data.head()

# datetimeを削除
merged_data = merged_data.drop(["datetime"], axis=1)

# titleの順番を入れ替え
merged_data = merged_data[
    [
        "month",
        "day",
        "hour",
        "N_temp",
        "N_press",
        "N_hum",
        "N_DPT",
        "S_temp",
        "S_press",
        "S_hum",
        "S_DPT",
    ]
]

# %% [markdown]
# model の作成
#

# %%
# testデータとtrainデータに半分に分割
split_pos = int(len(merged_data) * 0.7)
train_data = merged_data[:split_pos]
test_data = merged_data[split_pos:]
# test_dataのindexをリセット
test_data = test_data.reset_index(drop=True)

# %%
# 目的変数と説明変数の設定
drop_columns = ["N_press", "N_hum", "N_DPT", "S_temp", "S_hum", "S_press", "S_DPT"]
# drop_columns = ['S_temp', 'S_hum', 'S_press', 'S_DPT']
y_train = train_data["S_temp"]
x_train = train_data.drop(drop_columns, axis=1)
y_test = test_data["S_temp"]
x_test = test_data.drop(drop_columns, axis=1)

# %%
print(x_train.head())
print(y_train.head())
print(x_test.head())
print(y_test.head())

# %%
x_train.dtypes

# %%
x_test.dtypes

# %%
# # モデルの学習
# model = LinearRegression()
# model.fit(x_train, y_train)

# %%
# # 予測
# y_pred = model.predict(x_test)

# %%
# # 予測結果をy_testと比較
# plt.figure(figsize=(20, 10))
# plt.plot(y_test, label='test')
# plt.plot(y_pred, label='pred')
# plt.legend(loc='upper left', fontsize=20)
# plt.ylabel('temp')
# plt.tight_layout()
# plt.show()

# %%
# import packages
import sklearn

# from sklearn.utils_testing import all_estimators
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import numpy as np
import warnings

warnings.filterwarnings("ignore")

# %%
# 最適modelの探索
allAlgorithms = sklearn.utils.all_estimators(type_filter="regressor")
# ignore_algorithms = ['PLSRegression','AnotherAlgorithm', 'CCA']
ignore_algorithms = [
    "CCA",
    "IsotonicRegression",
    "MultiOutputRegressor",
    "MultiTaskElasticNet",
    "MultiTaskElasticNetCV",
    "MultiTaskLasso",
    "MultiTaskLassoCV",
    "PLSCanonical",
    "QuantileRegressor",
    "RadiusNeighborsRegressor",
    "RegressorChain",
    "StackingRegressor",
    "VotingRegressor",
]

result_dict = {}

plt.figure(figsize=(10, 100))
index = 1

for name, algorithm in allAlgorithms:
    if name in ignore_algorithms:
        continue

    print(index, name)

    # model 作成
    clf = algorithm()
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)

    # 平均二乗誤差
    mse = np.sqrt(mean_squared_error(y_test, y_pred))
    print("\t平均二乗誤差 : ", mse)

    # 絶対値平均
    mae = mean_absolute_error(y_test, y_pred)
    print("\t絶対値平均 : ", mae)

    # 正答率の計算　+=1度以上の誤差があれば不正解
    diff = abs(y_test - y_pred)
    correct = diff[diff < 1].count()
    answer_rate = correct / len(y_test)
    print("\t正答率 : ", answer_rate)
    print("")

    # 横軸がdatetime、縦軸が気温のグラフを作成
    plt.subplot(50, 1, index)
    x = (
        "2023-"
        + x_test["month"].astype(str)
        + "-"
        + x_test["day"].astype(str)
        + "-"
        + x_test["hour"].astype(str)
    )
    x = pd.to_datetime(x, format="%Y-%m-%d-%H")
    plt.plot(x, y_test, label="test")
    plt.plot(x, y_pred, label="pred")
    plt.plot(x, x_test["N_temp"], label="train")
    plt.legend(loc="upper left", fontsize=10)
    plt.title(name)
    plt.ylabel("temp")
    plt.tight_layout()

    # resultを格納
    result_dict[index] = {
        "name": name,
        "mse": mse,
        "mae": mae,
        "answer_rate": answer_rate,
    }
    index += 1

plt.show()

# %%
# result_dictをDataFrameに変換
result_df = pd.DataFrame.from_dict(result_dict, orient="index")

# anser_rateを%表示に変換
result_df["answer_rate"] = result_df["answer_rate"].apply(lambda x: x * 100)

# answer_rateの少数第二位を四捨五入
result_df["answer_rate"] = result_df["answer_rate"].round(1)


result_df

# %%
