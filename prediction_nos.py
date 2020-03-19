import pickle
import numpy as np
import renom as rm
import pandas as pd
import requests
from sklearn import preprocessing
from renom_rg.api.interface.regressor import Regressor

HOST = "localhost"
PORT = "8080"
url = "http://{}".format(HOST)

model_id = '6'
deploy_api = url + ':' + PORT + '/api/renom_rg/models/' + model_id + '/deploy'
requests.post(deploy_api)

regressor = Regressor(url, PORT)
regressor.pull()

pred_data = pd.read_csv("200302_atg_dsp_prediction.csv")
with open('./datasrc/prediction_set/pred.pickle', mode='wb') as f:
    pickle.dump(pred_data, f)

with open("./datasrc/prediction_set/pred.pickle", mode='rb') as f:
    data = pickle.load(f)

setsumei_list = list(pred_data.columns)
y_col = pd.DataFrame(data, columns=['mokuteki'])
x_col = pd.DataFrame(data, columns=setsumei_list)

pred_result = regressor.predict(np.array(x_col))

print(pred_result)