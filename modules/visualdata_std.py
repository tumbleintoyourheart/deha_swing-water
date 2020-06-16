import pickle
import numpy as np
import renom as rm
import pandas as pd
import requests
import matplotlib.pyplot as plt
from sklearn import preprocessing
from renom_rg.api.interface.regressor import Regressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

HOST = "localhost"
PORT = "8080"
url = "http://{}".format(HOST)
# 任意のモデルをデプロイ（モデルIDは変更してください）
model_id = '3' # model_idを選択する必要がある
deploy_api = url + ':' + PORT + '/api/renom_rg/models/' + model_id + '/deploy'
requests.post(deploy_api)
# regressorモジュールpull
regressor = Regressor(url, PORT)
regressor.pull()

#予測用の影響・制御因子データ読み込み
data = pd.read_csv("200302_atg_dsp_visual.csv")
pred_data = data.drop(columns=["day", "moisture_per"])
# with open('./datasrc/prediction_set/pred.pickle', mode='wb') as f:
#     pickle.dump(pred_data, f)

# # pickleデータの読み込み
# with open("./datasrc/prediction_set/pred.pickle", mode='rb') as f:
#     p_data = pickle.load(f)

# 【前処理2】: pred.pickle と同階層にあるoutputディレクトリの中から、必要なスケーラーを使用する
with open("../scaler_x.pickle", mode='rb') as f:
    scaler_X_standardization = pickle.load(f)
with open("../scaler_y.pickle", mode='rb') as f:
    scaler_y_standardization = pickle.load(f)

# （以下は標準化されたデータセットで学習したモデルの場合）
print(pred_data.head())
print(np.array(pred_data).shape)
np_x_col = scaler_X_standardization.transform(np.array(pred_data))

# 予測実行
pre = regressor.predict(np_x_col)

# 値を戻して、表示
pred_result = scaler_y_standardization.inverse_transform(pre)
print(pred_result)

data["prediction"] = pred_result
print(data)

#この行以降は、次回依頼いたします。
#可視化
print(data.columns)
plt.scatter(data["moisture_per"], pred_result)
plt.show()
plt.clf()

#誤差
score = round(r2_score(data["moisture_per"], pred_result), 2)
rmse  = round(np.sqrt(mean_squared_error(data["moisture_per"], pred_result)), 2)
print("R2    : "+str(score))
print("RMSE  : "+str(rmse))
