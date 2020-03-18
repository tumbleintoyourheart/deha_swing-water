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
model_id = '6' # model_idを選択する必要がある
deploy_api = url + ':' + PORT + '/api/renom_rg/models/' + model_id + '/deploy'
requests.post(deploy_api)
# regressorモジュールpull
regressor = Regressor(url, PORT)
regressor.pull()

#予測用の影響・制御因子データ読み込み
data = pd.read_csv("200302_atg_dsp_visual.csv")
pred_data = data.drop(columns=["day", "moisture_per"])
with open('./datasrc/prediction_set/pred.pickle', mode='wb') as f:
    pickle.dump(pred_data, f)

# pickleデータの読み込み
with open("./datasrc/prediction_set/pred.pickle", mode='rb') as f:
    p_data = pickle.load(f)

# 【前処理1】: 目的変数と説明変数に分ける（モデル学習に使用したデータセットの変数のみを抽出する）
setsumei_list = list(pred_data.columns)
y_col = pd.DataFrame(data, columns=['mokuteki'])
x_col = pd.DataFrame(data, columns=setsumei_list)

# 予測実行
pred_result = regressor.predict(np.array(x_col))
print(pred_result)

data["prediction"] = pred_result
print(data)

##この行以降は、次回依頼いたします。
##可視化
#print(data.columns)
#plt.scatter(data["moisture_per"], pred_result)
#plt.show()
#plt.clf()
#
##誤差
#score = round(r2_score(data["moisture_per"], pred_result), 2)
#rmse  = round(np.sqrt(mean_squared_error(data["moisture_per"], pred_result)), 2)
#print("R2    : "+str(score))
#print("RMSE  : "+str(rmse))
