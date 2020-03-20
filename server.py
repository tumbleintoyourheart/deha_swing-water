import os, sys, argparse, pickle, re, copy
from pathlib import *

from flask import Flask, jsonify, request
from flask_cors import CORS
import requests
from werkzeug.utils import secure_filename
import traceback

import numpy as np
import pandas as pd
from sklearn import preprocessing
from renom_rg.api.interface.regressor import Regressor

import warnings
for w in [UserWarning, FutureWarning, DeprecationWarning]:
    warnings.filterwarnings("ignore", category=w)

app = Flask(__name__)
CORS(app)


def init(host='localhost', port='8080', model_id=0):
    HOST = host
    PORT = port
    url = 'http://{}'.format(HOST)

    model_id = model_id
    deploy_api = url + ':' + PORT + '/api/renom_rg/models/' + model_id + '/deploy'
    requests.post(deploy_api)

    regressor = Regressor(url, PORT)
    regressor.pull()
    return regressor




@app.route('/renom_init', methods=['GET', 'POST'])
def renom_init():
    if request.method == 'POST':
        values = request.values.to_dict()
        host = values.get('host')
        port = values.get('port')
        model_id = values.get('model_id')
        
        global models
        models[model_id] = init(host, port, model_id)
        
        return 'Successfully initialized Regressor with model_id={}.'.format(model_id)
        
    else: return 'Not allowed method.'
    



def get_input(files, key):
    if files.get(key):
        inp = files[key]
        inp_name = secure_filename(inp.filename)
        return inp, inp_name
    else: return None, None
    
def save_input(inp, inp_name, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    abs_path = os.path.join(save_dir, inp_name)
    inp.save(abs_path)
    return abs_path
                


def prediction(regressor, csv_inp, mode):
    pred_data = pd.read_csv(csv_input)
    pickle.dump(pred_data, open('./datasrc/prediction_set/pred.pickle', mode='wb'))
    data = pickle.load(open("./datasrc/prediction_set/pred.pickle", mode='rb'))  
    
    setsumei_list = list(pred_data.columns)
    x_col = pd.DataFrame(data, columns=setsumei_list)
    
    if mode == 'nos':
        pred = regressor.predict(np.array(x_col))
        return pred

    elif mode == 'std':
        scaler_X_standardization = pickle.load(open("./scaler_x.pickle", mode='rb'))
        scaler_y_standardization = pickle.load(open("./scaler_y.pickle", mode='rb'))
        np_x_col = scaler_X_standardization.transform(np.array(x_col))
        pred = regressor.pred(np_x_col)
        pred = scaler_y_standardization.inverse_transform(pred)
        return pred
    

@app.route('/upload_scaler', methods=['GET', 'POST'])
def new_scaler():
    if request.method == 'POST':
        files = request.files.to_dict()
        
        scaler_x, scaler_x_name = get_input(files, 'scaler_x')
        scaler_y, scaler_y_name = get_input(files, 'scaler_y')
        
        save_dir = ('./')
        
        if scaler_x != None:
            if scaler_x_name != 'scaler_x.pickle': return 'Wrong scaler_x.pickle.'
            save_input(scaler_x, scaler_x_name, save_dir)
        else: return 'scaler_x.pickle is required.'
        if scaler_y != None:
            if scaler_y_name != 'scaler_y.pickle': return 'Wrong scaler_x.pickle.'
            save_input(scaler_y, scaler_y_name, save_dir)
        else: return 'scaler_y.pickle is required.'
        
        return '{}, {}'.format(scaler_x_name, scaler_y_name)
        
    else: return 'Not allowed method.'
    
    
    
@app.route('/ai', methods=['GET', 'POST'])
def ai():
    if request.method == 'POST':
        mode_pred, mode_vis = False, False
        response = {}
        
        values = request.values.to_dict()
        model_id = values.get('model_id'); regressor = models[model_id]
        mode = values.get('mode')

        files = request.files.to_dict()
        csv_pred, csv_pred_name = get_input(files, 'csv_prediction')
        if csv_pred != None:
            csv_pred_abspath = save_input(csv_pred, csv_pred_name, csv_savedir)
            mode_pred = True
            if 'prediction' not in csv_pred_name: return 'Not legal file for csv_prediction.'
        
        csv_vis, csv_vis_name = get_input(files, 'csv_visual')
        if csv_vis != None: 
            csv_vis_abspath = save_input(csv_vis, csv_vis_name, csv_savedir)
            mode_vis = True
            if 'visual' not in csv_vis_name: return 'Not legal file for csv_visual.'
        
        
        response['Renom'] = {'nos': {},
                             'std': {}}
        if mode_pred:
            pred_nos = prediction(regressor, csv_pred_abspath, 'nos')
            pred_nos = '{:.2f}'.format(pred_nos[0][0])
            response['Renom']['nos']['Prediction'] = pred_nos
            
            pred_std = prediction(regressor, csv_pred_abspath, 'std')
            pred_std = '{:.2f}'.format(pred_nos[0][0])
            response['Renom']['std']['Prediction'] = pred_std

            # if mode_vis:
            #     vis_res = sklearn_vis(Path(csv_vis_abspath), figs_savedir, False, *sklearn_models)
            #     if sklearn_nos: response['Scikit-learn']['nos']['Visualization'] = {'Figure path': str(PurePosixPath(vis_res[0][0])),
            #                                                 'R2': vis_res[0][1],
            #                                                 'RMSE': vis_res[0][2],
            #                                                 'Sorted predictions': vis_res[0][3]}
            #     if sklearn_std: response['Scikit-learn']['std']['Visualization'] = {'Figure path': str(PurePosixPath(vis_res[1][0])),
            #                                                 'R2': vis_res[1][1],
            #                                                 'RMSE': vis_res[1][2],
            #                                                 'Sorted predictions': vis_res[1][3]}
                
        return jsonify(response)
                
    else: return 'Not allowed method.'


@app.route('/', methods=['GET', 'POST'])
def hello():
    return 'hello'



if __name__ == '__main__':
    csv_savedir = ('./csv')
    figs_savedir = ('./visualizations')
    for d in [csv_savedir, figs_savedir]: os.makedirs(d, exist_ok=True)
    
    models = {}
    app.run(host='0.0.0.0', port=5000)