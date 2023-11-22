# %% [markdown]
# 気象庁のwebサイトからダウンロードした気象データを整形するプログラム
# - niigata_weather_data.csv
#   - 期間 : 2023 年 10 月 10 日 ~ 2023 年 11 月 14 日
#   - 出典 : 気象庁データ(https://www.data.jma.go.jp/gmd/risk/obsdl/#)

# %%
# import packages
import datetime
import pandas as pd

# %%
# read csv
# model構築に不要なheaderを除いて読み込み
raw_niigata_weather_data = pd.read_csv(
    '../data/niigata_weather_data.csv', encoding='shift-jis', header=2)
niigata_weather_data = raw_niigata_weather_data

# %%
niigata_weather_data.head()

# %%
# niiagata_weather_dataから必要なカラムを抽出
print(niigata_weather_data.columns)

# niigata_weather_dataの1行目を削除
niigata_weather_data.drop(0, axis=0, inplace=True)

# 削除するカラム
select_columns = [
    '気温(℃).1', '気温(℃).2', '現地気圧(hPa).1', '現地気圧(hPa).2', '相対湿度(％).1', '相対湿度(％).2', '露点温度(℃).1', '露点温度(℃).2'
]

# 削除するカラムを削除
niigata_weather_data.drop(select_columns, axis=1, inplace=True)
niigata_weather_data.head()

# %%
# niigata_weather_dataのカラム名を変更
niigata_weather_data.rename(
    columns={'年月日時': 'N_datetime', '気温(℃)': 'N_temp', '現地気圧(hPa)': 'N_press', '相対湿度(％)': 'N_hum', '露点温度(℃)': 'N_DPT'}, inplace=True)

# %%
# 　niigata_weather_dataののデータ型を確認
niigata_weather_data.dtypes

# %%
# niigata_weather_dataのdatetimeをdatetime型に変換
niigata_weather_data['N_datetime'] = niigata_weather_data['N_datetime'].apply(
    lambda x: datetime.datetime.strptime(x.split(' ')[0] + ' ' + x.split(' ')[1].split(':')[0], '%Y/%m/%d %H'))
print(niigata_weather_data.dtypes)
niigata_weather_data.head()

# %%
# describe niigata_weather_data
niigata_weather_data.describe()

# %%
# csvに出力
niigata_weather_data.to_csv(
    '../data/formatted_niigata_weather_data.csv', index=False)
