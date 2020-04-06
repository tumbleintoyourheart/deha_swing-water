import os, sys, argparse, pickle, re, copy
# from pathlib import *

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
        try:
            models[model_id] = init(host, port, model_id)
        except Exception as e:
            return jsonify(Error='Unable to pull model_id {}.'.format(model_id))
        
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
                


def prediction(device_id, regressor, scaler_x_path, scaler_y_path, csv_input, mode):
    inp = pd.read_csv(csv_input)
    pickle.dump(inp, open('./datasrc/prediction_set/pred.pickle', mode='wb'))
    inp = pickle.load(open("./datasrc/prediction_set/pred.pickle", mode='rb'))  
    
    setsumei_list = list(inp.columns)
    x_col = pd.DataFrame(inp, columns=setsumei_list)
    
    if mode == 'nos':
        pred = regressor.predict(np.array(x_col))
        return pred

    elif mode == 'std':
        scaler_X_standardization = pickle.load(open(scaler_x_path, mode='rb'))
        scaler_y_standardization = pickle.load(open(scaler_y_path, mode='rb'))
        np_x_col = scaler_X_standardization.transform(np.array(x_col))
        pred = regressor.predict(np_x_col)
        pred = scaler_y_standardization.inverse_transform(pred)
        return pred



def visualization(device_id, regressor, scaler_x_path, scaler_y_path, csv_input, mode):
    inp = pd.read_csv(csv_input)
    inp_pred = inp.drop(columns=["day", "moisture_per"])
    pickle.dump(inp_pred, open('./datasrc/prediction_set/pred.pickle', mode='wb'))
    inp_pred = pickle.load(open("./datasrc/prediction_set/pred.pickle", mode='rb'))  
    
    setsumei_list = list(inp_pred.columns)
    x_col = pd.DataFrame(inp, columns=setsumei_list)
    
    if mode == 'nos':
        pred = regressor.predict(np.array(x_col)).flatten()
        pred = [x for _, x in sorted(zip(inp['day'].tolist(), pred.tolist()), key=lambda Zip: Zip[0])]
        return pred

    elif mode == 'std':
        scaler_X_standardization = pickle.load(open(scaler_x_path, mode='rb'))
        scaler_y_standardization = pickle.load(open(scaler_y_path, mode='rb'))
        np_x_col = scaler_X_standardization.transform(np.array(x_col))
        pred = regressor.predict(np_x_col).flatten()
        pred = scaler_y_standardization.inverse_transform(pred)
        pred = [x for _, x in sorted(zip(inp['day'].tolist(), pred.tolist()), key=lambda Zip: Zip[0])]
        return pred
    


@app.route('/upload_scaler', methods=['GET', 'POST'])
def new_scaler():
    if request.method == 'POST':
        # values
        values = request.values.to_dict()
        model_id = values.get('model_id')
        if model_id == None: return 'Please specify model_id.'
        device_id = values.get('device_id')
        if device_id == None: return 'Please specify device_id.'
        
        # files
        files = request.files.to_dict()
        scaler_x, scaler_x_name = get_input(files, 'scaler_x')
        scaler_y, scaler_y_name = get_input(files, 'scaler_y')
        save_dir = ('./{}').format(device_id)
        
        # cases
        if scaler_x != None:
            if scaler_x_name != 'scaler_x.pickle': return 'Wrong scaler_x.pickle.'
            scaler_x_name = 'scaler_x_of_model_id_{}.pickle'.format(model_id)
            save_input(scaler_x, scaler_x_name, save_dir)
        else: return 'scaler_x.pickle is required.'
        
        if scaler_y != None:
            if scaler_y_name != 'scaler_y.pickle': return 'Wrong scaler_x.pickle.'
            scaler_y_name = 'scaler_y_of_model_id_{}.pickle'.format(model_id)
            save_input(scaler_y, scaler_y_name, save_dir)
        else: return 'scaler_y.pickle is required.'
        
        return '{}, {}'.format(scaler_x_name, scaler_y_name)
    else: return 'Not allowed method.'
    
    
    
@app.route('/renom_ai', methods=['GET', 'POST'])
def ai():
    if request.method == 'POST':
        response = {}
        
        # values
        values = request.values.to_dict()
        
        # model
        model_id = values.get('model_id')
        try:
            regressor = models[model_id]
        except Exception as e:
            tb = traceback.format_exc()
            print(tb)
            return jsonify(Error='Init model first.')
        
        device_id = values.get('device_id')
        if device_id == None: return 'Please specify device_id.'
        
        # scalers
        scaler_x_path = './{}/scaler_x_of_model_id_{}.pickle'.format(device_id, model_id)
        if not os.path.isfile(scaler_x_path):
            return 'scaler_x.pickle not found for device_id {}'.format(device_id)
        
        scaler_y_path = './{}/scaler_y_of_model_id_{}.pickle'.format(device_id, model_id)
        if not os.path.isfile(scaler_y_path):
            return 'scaler_y.pickle not found for device_id {}'.format(device_id)
        
        scalers_path = [scaler_x_path, scaler_y_path]
        
        mode = values.get('mode')

        files = request.files.to_dict()
        csv_pred, csv_pred_name = get_input(files, 'csv_prediction')
        mode_pred, mode_vis = False, False
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
            if mode == 'nos':
                try:
                    pred_nos = prediction(device_id, regressor, *scalers_path, csv_pred_abspath, 'nos')
                except Exception as e:
                    tb = traceback.format_exc()
                    print(tb)
                    tb = tb.split('\n')[-2]
                    return jsonify(Error=tb)
                pred_nos = '{:.2f} %'.format(pred_nos[0][0])
                response['Renom']['nos']['Prediction'] = pred_nos
            
            elif mode == 'std':
                try:
                    pred_std = prediction(device_id, regressor, *scalers_path, csv_pred_abspath, 'std')
                except Exception as e:
                    tb = traceback.format_exc()
                    print(tb)
                    tb = tb.split('\n')[-2]
                    return jsonify(Error=tb)
                pred_std = '{:.2f} %'.format(pred_std[0][0])
                response['Renom']['std']['Prediction'] = pred_std

        if mode_vis:
            if mode == 'nos':
                try:
                    pred_nos = visualization(device_id, regressor, *scalers_path, csv_vis_abspath, 'nos')
                except Exception as e:
                    tb = traceback.format_exc()
                    print(tb)
                    tb = tb.split('\n')[-2]
                    return jsonify(Error=tb)
                response['Renom']['nos']['Visualization'] = {'Sorted predictions': pred_nos}
            
            elif mode == 'std':
                try:
                    pred_std = visualization(device_id, regressor, *scalers_path, csv_vis_abspath, 'std')
                except Exception as e:
                    tb = traceback.format_exc()
                    print(tb)
                    tb = tb.split('\n')[-2]
                    return jsonify(Error=tb)
                response['Renom']['std']['Visualization'] = {'Sorted predictions': pred_std}
                
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