import pickle
import numpy as np
import renom as rm
import pandas as pd
import requests
from sklearn import preprocessing
from renom_rg.api.interface.regressor import Regressor

HOST = "13.230.125.129"
PORT = "80"
url = "http://{}".format(HOST)
# 任意のモデルをデプロイ（モデルIDは変更してください）
model_id = '6' # model_idを選択する必要がある
deploy_api = url + ':' + PORT + '/api/renom_rg/models/' + model_id + '/deploy'
requests.post(deploy_api)
# regressorモジュールpull
regressor = Regressor(url, PORT)
regressor.pull()

#予測用の影響・制御因子データ読み込み
pred_data = pd.read_csv("200302_atg_dsp_prediction.csv")
with open('./datasrc/prediction_set/pred.pickle', mode='wb') as f:
    pickle.dump(pred_data, f)

# pickleデータの読み込み
with open("./datasrc/prediction_set/pred.pickle", mode='rb') as f:
    data = pickle.load(f)

# 【前処理1】: 目的変数と説明変数に分ける（モデル学習に使用したデータセットの変数のみを抽出する）
setsumei_list = list(pred_data.columns)
y_col = pd.DataFrame(data, columns=['mokuteki'])
x_col = pd.DataFrame(data, columns=setsumei_list)

# 予測実行
pred_result = regressor.predict(np.array(x_col))

# 値を戻して、表示
print(pred_result)
