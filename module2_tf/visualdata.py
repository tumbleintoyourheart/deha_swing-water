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




def visualize(csv_input, figs_savedir, show, scaler, model1, model2):
    '''
    Unnormalized
    '''
    inp = pd.read_csv(csv_input)
    inp_pred = inp.drop(columns=["day", "moisture_per"])
    
    
    unnormed_pred = model1.predict(inp_pred).flatten()

    unnormed_res = [x for _, x in sorted(zip(inp['day'].tolist(), unnormed_pred.tolist()), key=lambda Zip: Zip[0])]
    
    plt.figure(0)
    plt.scatter(inp["moisture_per"], unnormed_pred)
    unnormed_savepath = figs_savedir/'tf_no_visualization.png'
    plt.savefig(unnormed_savepath)
    if show:
        plt.show()
        plt.clf()
    
    unnormed_r2 = round(r2_score(inp["moisture_per"], unnormed_pred), 2)
    unnormed_rmse = round(np.sqrt(mean_squared_error(inp["moisture_per"], unnormed_pred)), 2)
    print(f'R2: {unnormed_r2}')
    print(f'RMSE: {unnormed_rmse}')
    
    
    '''
    Normalized
    '''
    inp_pred = scaler.transform(inp_pred)

    normed_pred = model2.predict(inp_pred).flatten()

    normed_res = [x for _, x in sorted(zip(inp['day'].tolist(), normed_pred.tolist()), key=lambda Zip: Zip[0])]
    
    plt.figure(1)
    plt.scatter(inp["moisture_per"], normed_pred)
    normed_savepath = figs_savedir/'tf_std_visualization.png'
    plt.savefig(normed_savepath)
    if show:
        plt.show()
        plt.clf()

    normed_r2 = round(r2_score(inp["moisture_per"], normed_pred), 2)
    normed_rmse = round(np.sqrt(mean_squared_error(inp["moisture_per"], normed_pred)), 2)
    print(f'R2: {normed_r2}')
    print(f'RMSE: {normed_rmse}')
    
    return (unnormed_savepath, unnormed_r2, unnormed_rmse, unnormed_res), (normed_savepath, normed_r2, normed_rmse, normed_res)
    
if __name__ == '__main__':
    figs_savedir = Path('./visualizations')
    os.makedirs(figs_savedir, exist_ok=True)
    
    csv_input = Path('./csv')/'200302_atg_dsp_visual.csv'
    
    scaler = pickle.load(open(Path('./models')/'scaler.pickle', 'rb'))
    model1 = keras.models.load_model(Path('./models')/'200310_atg_dsp_tf_nn_nos.hdf5')
    model2 = keras.models.load_model(Path('./models')/'200310_atg_dsp_tf_nn_std.hdf5')
    
    visualize(csv_input, figs_savedir, True, scaler, model1, model2)