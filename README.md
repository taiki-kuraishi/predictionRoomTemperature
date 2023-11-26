# はじめに
実際に自分で集めたデータを用いて、機械学習の予測モデル構築をしたい。
自室の温度の予測モデルを構築することにしました。
その、データ取得からモデル作成、評価までの道のりを書きました。

この記事は、以下の記事と関係しています。

https://qiita.com/taiki-kuraishi/items/7739cffc79611d27045c

# 予測内容
- 説明変数
    - 新潟県の気温(℃)
    - 新潟県の湿度(%)
    - 新潟県の降水量(mm)
    - 新潟県の日射量(MJ/㎡)
    - 新潟県の風速(m/s)
    - 新潟県の気圧(hPa)

- 目的変数
    - 室内温度(switchBot 温湿度計によって計測される室内の温度(℃))

# データの準備
- 気象庁のＨＰか新潟県のら気象データのダウンロード
    - url : https://www.data.jma.go.jp/gmd/risk/obsdl/
    - 地点 : 新潟県
    - 期間 : 2023 年 10 月 10 日 ~ 2023 年 11 月 14 日
    - 項目 : 気温、降水量、日照時間、風向・風速、全天日照量、現地気圧、相対湿度、露点温度、雲量

- 室内の気温データ
    - switchBotから温湿度計のデータをダウンロード
    - 地点 : 新潟県新潟市某所
    - 期間 : 2023 年 10 月 10 日 ~ 2023 年 11 月 14 日
    - 項目 : 温度

- 新潟県の気温と室内温度のグラフ
<img src="/img/before_predict.png">


- 訓練データとテストデータの割合
    - 70% : 30%

# 予測モデル構築
- LinearRegression
    ```python
    from sklearn.linear_model import LinearRegression
    
    model = LinearRegression()
    
    model.fit(x_train, y_train)
    
    y_pred = model.predict(x_test)
    ```
    test : 実際の測定データ　pred : 予測値　train : 新潟県の気温データ
    <img src="/img/LR.png">

    <br>

- SVR (kernel=linear)
    ```python
    from sklearn.svm import SVR
    
    model = SVR(kernel="linear")
    
    model.fit(x_train, y_train)
    
    y_pred = model.predict(x_test)
    ```
    test : 実際の測定データ　pred : 予測値　train : 新潟県の気温データ
    <img src="/img/SVR_linear.png">
    <br>

- Lars
    ```python
    from sklearn.linear_model import Lars
    
    model = Lars()
    
    model.fit(x_train, y_train)
    
    y_pred = model.predict(x_test)
    ```
    test : 実際の測定データ　pred : 予測値　train : 新潟県の気温データ
    <img src="/img/Lars.png">
    <br>

- KernelRidge
    ```python
    from sklearn.kernel_ridge import KernelRidge
    
    model = KernelRidge()
    
    model.fit(x_train, y_train)
    
    y_pred = model.predict(x_test)
    ```
    test : 実際の測定データ　pred : 予測値　train : 新潟県の気温データ
    <img src="/img/KR.png">
    <br>


# モデル評価
- モデルの評価方法
    - **MAE** : 平均絶対誤差
        - 予測値と実際の値との絶対的な差の平均を表します。
        - MAEの最小値は0で、値が0に近いほどモデルとデータがよく当てはまっていることを示します。
        - 例えば、`mae = 1.323942`だった場合、<br>そのモデルの予測は平均 1.323942 ずれているということがわかります。
        ```math
        MAE = \frac{1}{N} \sum|y_i - \hat{y_i}|
        ```
        - python
            ```python
            from sklearn.metrics import mean_absolute_error
            mae = mean_absolute_error(y_test, y_pred)
            ```
            <br>
    - **MSE** : 平均二乗誤差
        - 予測値と実際の値との差の二乗の平均を表します。これは、大きなエラーをより重視するためのメトリクスで、予測エラーが大きい場合にはMAEよりも大きな値を示します。
        - MSEの最小値は0で、値が0に近いほどモデルとデータがよく当てはまっていることを示します。
        ```math
        MSE = \frac{\sum_{i=0}^{N-1} (\mathbf{y}_i - \hat{\mathbf{y}}\_i)^2}{N} 
        ```
        - python
            ```python
            from sklearn.metrics import mean_squared_error
            mse = mean_squared_error(y_test, y_pred)
            ```
            <br>
    - **R2** : 決定係数
        - R2は、モデルがデータの変動をどれだけ説明できるかを示す指標です。
        - 1に近いほどモデルの予測精度が高いと評価されます。
        - 具体的には、全変動（全データが平均からどれだけばらついているか）のうち、モデルが説明できる変動の割合を示します。
        ```math
        R^2 = 1 - \frac{\sum_{i = 1}^n ( y_i - \hat{y}_i ) ^2}{\sum_{i = 1}^n ( y_i - \bar{y} ) ^2}
        ```
        - python
            ```python
            from sklearn.metrics import r2_score
            r2 = r2_score(y_test, y_pred)
            ```
            <br>
- 評価結果<br>

    | モデル | MAE | MSE |R2 |
    |:---|:---:|:--:|:--:|
    |LinearRegression|1.4475982744109843|3.1562012498508447|0.5523273101586398|
    |SVR (kernel=linear)|1.4782935724471926|3.3019821622216803|0.5316498792845268|
    |Lars|1.355224203216034|2.5246931640257815|0.6418998377188626|
    |KernelRidge|1.2644409319197556|2.606269407541983|0.6303291381749798|

# まとめ
LarsのMAE(平均絶対誤差)が1.355224203216034だったので、そこそこの精度で予測できているとおもいます。
しかし、データんぼ数が少なく、季節性変動をうまく予測できていないように見受けられます。グラフを見ると、テストデータの10月のはじめは、うまく予測できていますが、11月に入ると予測精度が落ちます。
そのため、もっと多くのデータを収集する必要があると感じました。
