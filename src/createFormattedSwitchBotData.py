# %% [markdown]
# 自室に設置された SwitchBot 温湿度計によって計測されたデータを整形するプログラム
#
# - raw_switchBot_data
#   - 期間 : 2022-07-24 15:40:01 ~ 2023-11-21 13:53:10
#   - 出典 : 自宅に設置された SwichBot 音湿度計
#

# %%
# import packages
import datetime
import pandas as pd

# %%
# read csv
raw_switchBot_data = pd.read_csv('../data/switchBot_data.csv', encoding='utf-8')
switchBot_data = raw_switchBot_data

# %%
switchBot_data.head()

# %%
# switchBot_dataから必要なカラムを抽出
switchBot_data.drop(['Absolute_Humidity(g/m³)'], axis=1, inplace=True)
switchBot_data.head()

# %%
# switchBot_dataのカラム名を変更
switchBot_data.rename(
    columns={'Timestamp': 'S_datetime', 'Temperature_Celsius(°C)': 'S_temp', 'Relative_Humidity(%)': 'S_hum', 'DPT_Celsius(°C)': 'S_DPT', 'VPD(kPa)': 'S_press'}, inplace=True)

# %%
# switchBot_dataののカラムの順番を変更
switchBot_data = raw_switchBot_data.reindex(
    columns=['S_datetime', 'S_temp', 'S_press', 'S_hum', 'S_DPT'])

# %%
# SwitchBotのデータ型を確認
switchBot_data.dtypes

# %%
# 欠損値の確認
switchBot_data.isnull().sum()

# %%
# swichBot_dataのdatetimeをdatetime型に変換
switchBot_data['S_datetime'] = switchBot_data['S_datetime'].apply(
    lambda x: datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
print(switchBot_data.dtypes)
switchBot_data.head()

# %%
# describe switchBot_data
switchBot_data.describe()

# %%
# switchBot_dataから2023-10-10 01:00:00~2023-11-15 00:00:59の期間のデータをを抽出
switchBot_data = switchBot_data[(switchBot_data['S_datetime'] >= datetime.datetime(
    2023, 10, 10, 1)) & (switchBot_data['S_datetime'] <= datetime.datetime(2023, 11, 15, 0, 59))]

# %%
switchBot_data.describe()

# %%
# csvファイルとして出力
switchBot_data.to_csv('../data/formatted_switchBot_data.csv', index=False)
