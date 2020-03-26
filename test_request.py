import pickle
import requests


url = 'http://0.0.0.0:5000/upload_model'
files = {'models': open('./module1_sklearn/models/200310_atg_dsp_sk_rf_nos.pickle', 'rb')}
res = requests.post(url, files=files)
print(res.text)