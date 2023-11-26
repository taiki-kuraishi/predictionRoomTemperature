# %%
import pandas as pd
import matplotlib.pyplot as plt

# %%
data = pd.read_csv("../data/merged_data.csv", encoding="utf-8")
data.head()

# %%
# testデータとtrainデータに半分に分割
split_pos = int(len(data) * 0.7)
train_data = data[:split_pos]
test_data = data[split_pos:]

# test_dataのindexをリセット
test_data = test_data.reset_index(drop=True)

# %%
# 目的変数と説明変数の設定
drop_columns = ["S_temp", "S_hum", "S_dpt", "S_vpd", "N_dpt", "N_vpd", "N_ca", "N_sd"]
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
# model構築
from sklearn.svm import SVR

model = SVR(kernel="rbf")

model.fit(x_train, y_train)

y_pred = model.predict(x_test)

# %%
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score

# 平均二乗誤差
mse = mean_squared_error(y_test, y_pred)
print("\t平均二乗誤差 : ", mse)

# 絶対値平均
mae = mean_absolute_error(y_test, y_pred)
print("\t絶対値平均 : ", mae)

# 決定係数
r2 = r2_score(y_test, y_pred)
print("\t決定係数 : ", r2)

# 正答率の計算　+=1度以上の誤差があれば不正解
diff = abs(y_test - y_pred)
correct = diff[diff < 1].count()
answer_rate = correct / len(y_test)
print("\t正答率 : ", answer_rate)

# %%
# 　グラフ化
plt.figure(figsize=(20, 5))
plt.rcParams["font.size"] = 15

# year,month,day,hourを結合,datetime型に変換
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
plt.legend(loc="lower left", fontsize=15)
plt.title("SVR (kernel=rbf)")
plt.ylabel("temp")
plt.tight_layout()
