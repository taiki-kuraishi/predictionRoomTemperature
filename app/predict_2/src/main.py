# %%
# import packages
import re
import datetime
import pandas as pd
import matplotlib.pyplot as plt

# %% [markdown]
# read csv
#

# %%
# read csv
raw_niigata_data = pd.read_csv(
    "../data/raw_niigata_weather_data.csv.csv", encoding="cp932", header=2
)
niigata_data = raw_niigata_data

raw_switchBot_data = pd.read_csv("../data/raw_switchBot_data.csv", encoding="utf-8")
switchBot_data = raw_switchBot_data

# %%
niigata_data.head()

# %%
switchBot_data.head()

# %% [markdown]
# format niigata_data
#

# %%
# niigata_dataの必要のないカラムの削除
niigata_data = niigata_data.drop(["降水量(mm).1", "風速(m/s).1", "日照時間(時間).1"], axis=1)

# niigata_dataの必要のない行の削除
niigata_data = niigata_data.drop([0, 1])

# niigata_dataのindexのリセット
niigata_data = niigata_data.reset_index(drop=True)

# niigata_dataのカラム名の変更
niigata_data = niigata_data.rename(
    columns={
        "年月日時": "N_datetime",
        "気温(℃)": "N_temp",
        "降水量(mm)": "N_ppt",
        "日射量(MJ/㎡)": "N_srad",
        "風速(m/s)": "N_wspd",
        "相対湿度(％)": "N_hum",
        "雲量(10分比)": "N_ca",
        "現地気圧(hPa)": "N_press",
        "日照時間(時間)": "N_sd",
        "露点温度(℃)": "N_dpt",
    }
)

# niigata_dataのカラムの並び替え
niigata_data = niigata_data[
    [
        "N_datetime",
        "N_temp",
        "N_hum",
        "N_dpt",
        "N_ppt",
        "N_srad",
        "N_wspd",
        "N_ca",
        "N_press",
        "N_sd",
    ]
]

niigata_data.head()

# %%
# niigata_dataのデータ型の確認
niigata_data.dtypes

# %%
# niigata_dataの欠損地の確認
niigata_data.isnull().sum()

# %%
# niigata_dataのN_sradの欠損地の補完

# niigata_dataのN_sradが欠損地の行のindexを確認]
index_list = list(niigata_data[niigata_data["N_srad"].isnull()].index)
print(index_list)

# niigata_dataのN_sradが欠損地の行の前後の値を確認
for index in index_list:
    print(niigata_data.iloc[index - 5 : index + 5])

# niigata_dataのN_sradが欠損地の行の前後の値の平均値を計算
for index in index_list:
    niigata_data["N_srad"][index] = (
        niigata_data["N_srad"][index - 1] + niigata_data["N_srad"][index + 1]
    ) / 2

# %%
# niigata_dataのN_wspdの欠損地の補完

# niigata_dataのN_wspdが欠損地の行のindexを確認
index_list = list(niigata_data[niigata_data["N_wspd"].isnull()].index)
print(index_list)

# niigata_dataのN_wspdが欠損地の行の前後の値を確認
for index in index_list:
    print(niigata_data.iloc[index - 5 : index + 5])

# niigata_dataのN_wspdが欠損地の行の前後の値の平均値を計算
for index in index_list:
    niigata_data["N_wspd"][index] = (
        niigata_data["N_wspd"][index - 1] + niigata_data["N_wspd"][index + 1]
    ) / 2

# %%
# niigata_dataのN_caの欠損地を0で補完
niigata_data["N_ca"] = niigata_data["N_ca"].fillna(0)

# %%
# niigata_dataのN_sdの欠損地を0で補完

# niigata_dataのN_sdが欠損地の行のindexを確認
index_list = list(niigata_data[niigata_data["N_sd"].isnull()].index)
print(index_list)

# niigata_dataのN_sdが欠損地の行の前後の値を確認
for index in index_list:
    print(niigata_data.iloc[index - 5 : index + 5])

# niigata_dataのN_sdが欠損地の行の前後の値の平均値を計算
for index in index_list:
    niigata_data["N_sd"][index] = (
        niigata_data["N_sd"][index - 1] + niigata_data["N_sd"][index + 1]
    ) / 2

# %%
# niigata_dataのカラムにVPDを追加
niigata_data["N_vpd"] = (
    6.1078 * 10 ** (7.5 * niigata_data["N_temp"] / (niigata_data["N_temp"] + 237.5))
) * (1 - niigata_data["N_hum"] / 100)
# niigata_data = niigata_data.drop(["N_dpt"], axis=1)
niigata_data.head()

# %%
# niigata_dataのカラムの並び替え
niigata_data = niigata_data[
    [
        "N_datetime",
        "N_temp",
        "N_hum",
        "N_dpt",
        "N_vpd",
        "N_ppt",
        "N_srad",
        "N_wspd",
        "N_ca",
        "N_press",
        "N_sd",
    ]
]

niigata_data.head()

# %%
# niigata_weather_dataのdatetimeをdatetime型に変換
niigata_data["N_datetime"] = niigata_data["N_datetime"].apply(
    lambda x: datetime.datetime.strptime(
        x.split(" ")[0] + " " + x.split(" ")[1].split(":")[0], "%Y/%m/%d %H"
    )
)
niigata_data.head()

# %%
# niigata_dataのN_caに含まれる値の種類をすべて確認
print(list(niigata_data["N_ca"].unique()))

# niigata_dataのN_caの値から数字を抜き出して再代入、int型に変換
niigata_data["N_ca"] = niigata_data["N_ca"].apply(
    lambda x: int(re.sub(r"\D", "", str(x)))
)

print(list(niigata_data["N_ca"].unique()))

# %%
niigata_data.dtypes

# %%
# niigata_dataの欠損地の確認
niigata_data.isnull().sum()

# %%
niigata_data.describe()

# %% [markdown]
# format switchBot_data
#

# %%
# niigata_dataの必要のないカラムの削除
switchBot_data = switchBot_data.drop(["Absolute_Humidity(g/m³)"], axis=1)

# switchBot_dataのカラム名の変更
switchBot_data = switchBot_data.rename(
    columns={
        "Timestamp": "S_datetime",
        "Temperature_Celsius(°C)": "S_temp",
        "Relative_Humidity(%)": "S_hum",
        "DPT_Celsius(°C)": "S_dpt",
        "VPD(kPa)": "S_vpd",
    }
)
switchBot_data.head()

# %%
# swichBot_dataのdatetimeをdatetime型に変換
switchBot_data["S_datetime"] = switchBot_data["S_datetime"].apply(
    lambda x: datetime.datetime.strptime(x, "%Y-%m-%d %H:%M:%S")
)
print(switchBot_data.dtypes)
switchBot_data.head()

# %%
# switchBot_dataから2023-10-10 01:00:00~2023-11-15 00:00:59の期間のデータをを抽出
switchBot_data = switchBot_data[
    (switchBot_data["S_datetime"] >= datetime.datetime(2023, 10, 10, 1))
    & (switchBot_data["S_datetime"] <= datetime.datetime(2023, 11, 15, 0, 59))
]

# switvhBot_dataのindexをリセット[
switchBot_data = switchBot_data.reset_index(drop=True)

switchBot_data.head()

# %%
# switchBot_dataののデータを１時間起きにresample
switchBot_data = switchBot_data.resample("1H", on="S_datetime").mean()

# indexをリセット
switchBot_data = switchBot_data.reset_index()

switchBot_data.head()

# %%
switchBot_data.isnull().sum()

# %%
switchBot_data.dtypes

# %%
switchBot_data.describe()

# %%
# niigata_dataとswitchBot_dataをdatetimeをキーにして左結合
niigata_data = niigata_data.set_index("N_datetime")
switchBot_data = switchBot_data.set_index("S_datetime")
merged_data = niigata_data.join(switchBot_data, how="left")

# indexをリセット
merged_data = merged_data.reset_index()

# カラム名の変更
merged_data = merged_data.rename(columns={"N_datetime": "datetime"})

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

# merged_dataのカラムをリストで取得
# merged_data_columns = list(merged_data.columns)
# print(merged_data_columns)

# titleの順番を入れ替え
merged_data = merged_data[
    [
        "month",
        "day",
        "hour",
        "N_temp",
        "N_hum",
        "N_dpt",
        "N_vpd",
        "N_ppt",
        "N_srad",
        "N_wspd",
        "N_ca",
        "N_press",
        "N_sd",
        "S_temp",
        "S_hum",
        "S_dpt",
        "S_vpd",
    ]
]

merged_data.head()

# %%
# merge_dataをcsvファイルに出力
merged_data.to_csv("../data/merged_data.csv", index=False)

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
# drop_columns = ["N_press", "N_hum", "N_DPT", "S_temp", "S_hum", "S_press", "S_DPT"]
drop_columns = [
    "S_temp",
    "S_hum",
    "S_dpt",
    "S_vpd",
]
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
