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
model_id = '7'
deploy_api = url + ':' + PORT + '/api/renom_rg/models/' + model_id + '/deploy'
requests.post(deploy_api)
regressor = Regressor(url, PORT)
regressor.pull()

data = pd.read_csv("prediction.csv")

setsumei_list = list(data.columns)
y_col = pd.DataFrame(data, columns=['mokuteki'])
x_col = pd.DataFrame(data, columns=setsumei_list)

with open("../6/scaler_x_of_model_id_7.pickle", mode='rb') as f:
    scaler_X_standardization = pickle.load(f)
with open("../6/scaler_y_of_model_id_7.pickle", mode='rb') as f:
    scaler_y_standardization = pickle.load(f)

np_x_col = scaler_X_standardization.transform(np.array(x_col))

pre = regressor.predict(np_x_col)

pred_result = scaler_y_standardization.inverse_transform(pre)
print(pred_result)