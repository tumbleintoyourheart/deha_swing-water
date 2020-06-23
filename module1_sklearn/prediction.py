# 機械学習
import json
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



def predict(input_json, model_modes, scaler, model1, model2):
    inp = pd.read_json(input_json, typ='series')
    inp = pd.DataFrame([inp])
    
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
