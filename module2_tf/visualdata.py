# 機械学習
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




def visualize(csv_input, figs_savedir, show, scaler, model):
    inp = pd.read_csv(csv_input)
    inp_pred = inp.drop(columns=["day", "moisture_per"])
    
    inp_pred = scaler.transform(inp_pred)

    normed_pred = model.predict(inp_pred)

    plt.figure(0)
    plt.scatter(inp["moisture_per"], normed_pred)
    normed_savepath = figs_savedir/'tf_normalized_visualization.png'
    plt.savefig(normed_savepath)
    if show:
        plt.show()
        plt.clf()

    normed_r2 = round(r2_score(inp["moisture_per"], normed_pred), 2)
    normed_rmse = round(np.sqrt(mean_squared_error(inp["moisture_per"], normed_pred)), 2)
    print(f'R2: {normed_r2}')
    print(f'RMSE: {normed_rmse}')
    
    return normed_savepath, normed_r2, normed_rmse
    
if __name__ == '__main__':
    figs_savedir = Path('./visualizations')
    os.makedirs(figs_savedir, exist_ok=True)
    csv_input = './200302_A01_visual.csv'
    
    scaler = pickle.load(open("scaler.pickle", "rb"))
    model = keras.models.load_model('nn_model.hdf5')
    
    visualize(csv_input, figs_savedir, True, scaler, model)