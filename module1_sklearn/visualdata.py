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
import os, json

from pathlib import *

import warnings
for w in [UserWarning, FutureWarning, DeprecationWarning]:
    warnings.filterwarnings("ignore", category=w)


def visualize(input_json, model_modes, figs_savedir, show, scaler, model1, model2):
    # inp = pd.read_json(input_json, typ='series')
    # inp = pd.DataFrame([inp])
    input_json  = json.loads(input_json)
    input_df    = pd.DataFrame.from_dict(input_json)
    input_pred  = input_df.drop(columns=["day", "moisture_per"])
    if 'nos' in model_modes:
        '''
        Unnormalized
        '''
        unnormed_pred = model1.predict(input_pred)
        
        unnormed_res = [x for _, x in sorted(zip(input_df['day'].tolist(), unnormed_pred.tolist()), key=lambda Zip: Zip[0])]
        
        plt.figure(0)
        plt.scatter(input_df["moisture_per"], unnormed_pred)
        unnormed_savepath = figs_savedir/'sklearn_no_visualization.png'
        plt.savefig(unnormed_savepath)
        if show:
            plt.show()
            plt.clf()

        unnormed_r2 = round(r2_score(input_df["moisture_per"], unnormed_pred), 2)
        unnormed_rmse = round(np.sqrt(mean_squared_error(input_df["moisture_per"], unnormed_pred)), 2)
        print(f'R2: {unnormed_r2}')
        print(f'RMSE: {unnormed_rmse}')
    else: unnormed_res, unnormed_savepath, unnormed_r2, unnormed_rmse = None, None, None, None

    if 'std' in model_modes:
        '''
        Normalized
        '''
        input_pred_sc   = scaler.transform(input_pred)

        normed_pred     = model2.predict(input_pred_sc)

        normed_res      = [x for _, x in sorted(zip(input_df['day'].tolist(), normed_pred.tolist()), key=lambda Zip: Zip[0])]
        
        plt.figure(1)
        plt.scatter(input_df["moisture_per"], normed_pred)
        normed_savepath = figs_savedir/'sklearn_std_visualization.png'
        plt.savefig(normed_savepath)
        if show:
            plt.show()
            plt.clf()

        normed_r2 = round(r2_score(input_df["moisture_per"], normed_pred), 2)
        normed_rmse = round(np.sqrt(mean_squared_error(input_df["moisture_per"], normed_pred)), 2)
        print(f'R2: {normed_r2}')
        print(f'RMSE: {normed_rmse}')
    else: normed_res, normed_savepath, normed_r2, normed_rmse = None, None, None, None
    
    return (unnormed_savepath, unnormed_r2, unnormed_rmse, unnormed_res), (normed_savepath, normed_r2, normed_rmse, normed_res)