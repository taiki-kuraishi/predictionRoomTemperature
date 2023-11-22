# %% [markdown]
# switchBot 温湿度計にて計測したデータを１時間間隔での平均でリサンプリング
#

# %%
# import packages
import datetime
import pandas as pd

# %%
# read csv
switchBot_data = pd.read_csv(
    '../data/formatted_switchBot_data.csv', encoding='utf-8', index_col=None)

# %%
switchBot_data.head()

# %%
switchBot_data.dtypes

# %%
# convert to datetime
switchBot_data['S_datetime'] = pd.to_datetime(
    switchBot_data['S_datetime'], format='%Y-%m-%d %H:%M:%S')
switchBot_data.head()

# %%
# switchBot_dataののデータを１時間起きにresample
switchBot_data = switchBot_data.resample('1H', on='S_datetime').mean()

# indexをリセット
switchBot_data = switchBot_data.reset_index()

# %%
switchBot_data.describe()

# %%
# to csv
switchBot_data.to_csv('../data/resampled_switchBot_data.csv', index=False)
