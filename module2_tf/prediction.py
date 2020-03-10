# ニューラルネット
import pandas as pd
import numpy as np
import pickle
import keras
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib

from pathlib import *

import warnings
for w in [UserWarning, FutureWarning, DeprecationWarning]:
    warnings.filterwarnings("ignore", category=w)
    
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['KMP_WARNINGS'] = '0'

from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False
    
    




def predict(csv_input, scaler, model=None):
    #予測用の影響・制御因子データ読み込み
    inp = pd.read_csv(csv_input)

    #標準化データのロードと標準化の実行
    # scaler = pickle.load(open("scaler.pickle", "rb"))
    inp = scaler.transform(inp)

    #予測実行
    pred_normed = model.predict(inp)
    print(f'Normalized prediction: {pred_normed}.')

    return pred_normed

if __name__ == '__main__':
    csv_input = './200302_A01_prediction.csv'
    
    scaler = pickle.load(open("scaler.pickle", "rb"))
    model = keras.models.load_model('nn_model.hdf5')
    
    predict(csv_input, scaler, model)