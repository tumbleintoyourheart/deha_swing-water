# 機械学習
import pandas as pd
import numpy as np
import pickle
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



def predict(csv_input, model_modes, scaler, model1, model2):
    inp = pd.read_csv(csv_input)
    
    if 'nos' in model_modes:
        '''
        Unnormalized
        '''
        pred_unnormed = model1.predict(inp)
        print(f'nos: {pred_unnormed}.')
    else: pred_unnormed = None

    if 'std' in model_modes:
        '''
        Normalized
        '''
        inp = scaler.transform(inp)
        pred_normed = model2.predict(inp)
        print(f'std: {pred_normed}.')
    else: pred_normed = None
    return pred_unnormed, pred_normed
    
if __name__ == '__main__':
    csv_input = Path('./csv')/'200302_atg_dsp_prediction.csv'
    
    scaler = pickle.load(open(Path('./models')/'scaler.pickle', 'rb'))
    model1 = pickle.load(open(Path('./models')/'200310_atg_dsp_sk_rf_nos.pickle', mode='rb'))
    model2 = pickle.load(open(Path('./models')/'200310_atg_dsp_sk_rf_std.pickle', mode='rb'))
    
    predict(csv_input, scaler, model1, model2)