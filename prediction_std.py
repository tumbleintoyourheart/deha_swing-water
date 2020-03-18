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
model_id = '3' # model_idを選択する必要がある
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

# 【前処理2】: pred.pickle と同階層にあるoutputディレクトリの中から、必要なスケーラーを使用する
with open("./scaler_x.pickle", mode='rb') as f:
    scaler_X_standardization = pickle.load(f)
with open("./scaler_y.pickle", mode='rb') as f:
    scaler_y_standardization = pickle.load(f)

# （以下は標準化されたデータセットで学習したモデルの場合）
np_x_col = scaler_X_standardization.transform(np.array(x_col))

# 予測実行
pre = regressor.predict(np_x_col)

# 値を戻して、表示
pred_result = scaler_y_standardization.inverse_transform(pre)
print(pred_result)
