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

import warnings
for w in [UserWarning, FutureWarning, DeprecationWarning]:
    warnings.filterwarnings("ignore", category=w)



def predict(csv_input, model1, scaler, model2):
    '''
    Unnormalized
    '''
    #予測用の影響・制御因子データ読み込み
    inp = pd.read_csv(csv_input)

    #AIモデルファイルのロード
    # model1 = pickle.load(open('rf_model.pickle', mode='rb'))

    #予測実行
    pred_unnormed = model1.predict(inp)
    print(f'Unnormalized prediction: {pred_unnormed}.')


    '''
    Normalized
    '''
    inp = pd.read_csv(csv_input)
    
    #標準化データのロードと標準化の実行
    # scaler = pickle.load(open("scaler.pickle", "rb"))
    inp = scaler.transform(inp)

    #AIモデルファイルのロード
    # model2 = pickle.load(open('rfstd_model.pickle', mode='rb'))

    #予測実行
    pred_normed = model2.predict(inp)
    print(f'Normalized prediction: {pred_normed}.')
    
    return pred_unnormed, pred_normed
    
if __name__ == '__main__':
    csv_input = './200302_A01_prediction.csv'
    
    model1 = pickle.load(open('rf_model.pickle', mode='rb'))
    scaler = pickle.load(open("scaler.pickle", "rb"))
    model2 = pickle.load(open('rfstd_model.pickle', mode='rb'))
    
    predict(csv_input, model1, scaler, model2)