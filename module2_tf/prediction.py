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
    
    




def predict(csv_input, scaler, model1, model2):
    inp = pd.read_csv(csv_input)
    
    pred_unnormed = model1.predict(inp)
    print(f'nos: {pred_unnormed}.')

    inp = scaler.transform(inp)
    pred_normed = model2.predict(inp)
    print(f'std: {pred_normed}.')

    return pred_unnormed, pred_normed

if __name__ == '__main__':
    csv_input = Path('./csv')/'200302_atg_dsp_prediction.csv'
    
    scaler = pickle.load(open(Path('./models')/'scaler.pickle', 'rb'))
    model1 = keras.models.load_model(Path('./models')/'200310_atg_dsp_tf_nn_nos.hdf5')
    model2 = keras.models.load_model(Path('./models')/'200310_atg_dsp_tf_nn_std.hdf5')
    
    predict(csv_input, scaler, model1, model2)