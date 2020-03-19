import pickle
import numpy as np
import renom as rm
import pandas as pd
import requests
from sklearn import preprocessing
from renom_rg.api.interface.regressor import Regressor

url = 'http://0.0.0.0:80/'
files = {'scaler_x': open('./scaler_x.pickle', 'rb'),
         'scaler_y': open('./scaler_y.pickle', 'rb')}
res = requests.post(url, files=files)
print(res.text)