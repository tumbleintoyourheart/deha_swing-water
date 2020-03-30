import os, sys, argparse, pickle, re, copy
from pathlib import *

from flask import Flask, jsonify, request
from flask_cors import CORS
from werkzeug.utils import secure_filename
import traceback

from module1_sklearn.prediction import predict as sklearn_pred
from module1_sklearn.visualdata import visualize as sklearn_vis

from module2_tf.prediction import predict as tf_pred
from module2_tf.visualdata import visualize as tf_vis

import keras, tensorflow as tf
from tensorflow.python.keras.backend import set_session
from tensorflow.python.keras.models import load_model


import warnings
for w in [UserWarning, FutureWarning, DeprecationWarning]:
    warnings.filterwarnings("ignore", category=w)
    
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['KMP_WARNINGS'] = '0'

from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False


app = Flask(__name__)
CORS(app)

        
sklearn_default_models = ['200310_atg_dsp_sk_rf_nos.pickle', '200310_atg_dsp_sk_rf_std.pickle']
tf_default_models = ['200310_atg_dsp_tf_nn_nos.hdf5', '200310_atg_dsp_tf_nn_std.hdf5']

def init(sklearn_defaults=sklearn_default_models, tf_defaults=tf_default_models):
    sklearn_scaler = [pickle.load(open(sklearn_path/'models'/'scaler.pickle', 'rb'))]
    sklearn_models = [pickle.load(open(sklearn_path/'models'/model_name, 'rb')) for model_name in sklearn_defaults]
    sklearn_models = sklearn_scaler + sklearn_models
    
    tf_scaler = [pickle.load(open(tf_path/'models'/'scaler.pickle', 'rb'))]
    global graph, sess
    sess = tf.Session()
    graph = tf.get_default_graph()
    set_session(sess)
    tf_models = [load_model(tf_path/'models'/model_name) for model_name in tf_defaults]
    tf_models = tf_scaler + tf_models
    
    return sklearn_models, tf_models
  
    

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
                


@app.route('/upload_model', methods=['GET', 'POST'])
def new_model():
    if request.method == 'POST':
        files = request.files.to_dict()
        
        model, model_name = get_input(files, 'model')
        scaler, scaler_name = get_input(files, 'scaler')
        
        if model == None: return 'Please specify a model to upload.'
        
        if [('std' in model_name), scaler != None].count(True) == 1:
            return 'std model and scaler must come together. Please make sure to upload both.'
        
        if 'sk' in model_name:
            save_dir = sklearn_path/'models'
            save_input(model, model_name, save_dir)
            if scaler != None: save_input(scaler, scaler_name, save_dir)
        elif 'tf' in model_name:
            save_dir = tf_path/'models'
            save_input(model, model_name, save_dir)
            if scaler != None: save_input(scaler, scaler_name, save_dir)

        return f'{model_name}'
        
    else: return 'Not allowed method.'
    
    
    
@app.route('/ai', methods=['GET', 'POST'])
def ai():
    if request.method == 'POST':
        mode_pred, mode_vis = False, False
        response = {}
        
        
        values = request.values.to_dict()
        
        global sklearn_default_models, tf_default_models
        sklearn_nos, sklearn_std, tf_nos, tf_std = False, False, False, False
        modules = set()
        if values.get('models'):
            models = values['models'].replace(' ', '').split(',')
            available_models = [m.name for m in list(sklearn_path.rglob('*.pickle')) if 'scaler' not in m.name] + [m.name for m in list(tf_path.rglob('*.hdf5'))]
            
            for m in models:
                if m not in available_models: return f'Not available models. Please choose from: {available_models}.'
                elif re.search(r'sk\w*nos.pickle', m): sklearn_default_models[0] = m; modules.add('sklearn'); sklearn_nos = True
                elif re.search(r'sk\w*std.pickle', m): sklearn_default_models[1] = m; modules.add('sklearn'); sklearn_std = True
                elif re.search(r'tf\w*nos.hdf5', m): tf_default_models[0] = m; modules.add('tensorflow'); tf_nos = True
                elif re.search(r'tf\w*std.hdf5', m): tf_default_models[1] = m; modules.add('tensorflow'); tf_std = True
                else: return f'Not available models. Please choose from: {available_models}.'
        else: return 'Please specify models to use.'
        
        
        # init models
        sklearn_models, tf_models = init()
        
        
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
        
        for module in modules:
            if module == 'sklearn':
                response['Scikit-learn'] = {'nos': {},
                                            'std': {}}
                
                if sklearn_nos: response['Scikit-learn']['nos']['Model'] = sklearn_default_models[0]
                if sklearn_std: response['Scikit-learn']['std']['Model'] = sklearn_default_models[1]
                
                if mode_pred:
                    pred_res = sklearn_pred(Path(csv_pred_abspath), *sklearn_models)
                    pred_res = [f'{res[0]:.2f} %' for res in pred_res]
                    if sklearn_nos: response['Scikit-learn']['nos']['Prediction'] = pred_res[0]
                    if sklearn_std: response['Scikit-learn']['std']['Prediction'] = pred_res[1]

                if mode_vis:
                    vis_res = sklearn_vis(Path(csv_vis_abspath), figs_savedir, False, *sklearn_models)
                    if sklearn_nos: response['Scikit-learn']['nos']['Visualization'] = {'Figure path': str(PurePosixPath(vis_res[0][0])),
                                                                'R2': vis_res[0][1],
                                                                'RMSE': vis_res[0][2],
                                                                'Sorted predictions': vis_res[0][3]}
                    if sklearn_std: response['Scikit-learn']['std']['Visualization'] = {'Figure path': str(PurePosixPath(vis_res[1][0])),
                                                                'R2': vis_res[1][1],
                                                                'RMSE': vis_res[1][2],
                                                                'Sorted predictions': vis_res[1][3]}
        
            elif module == 'tensorflow':
                response['Tensorflow'] = {'nos': {},
                                          'std': {}}
                
                if tf_nos: response['Tensorflow']['nos']['Model'] = tf_default_models[0]
                if tf_std: response['Tensorflow']['std']['Model'] = tf_default_models[1]
                
                global graph, sess
                with graph.as_default():
                    set_session(sess)
                    
                    if mode_pred:
                        pred_res = tf_pred(Path(csv_pred_abspath), *tf_models)
                        pred_res = [f'{res[0][0]:.2f} %' for res in pred_res]
                        if tf_nos: response['Tensorflow']['nos']['Prediction'] = pred_res[0]
                        if tf_std: response['Tensorflow']['std']['Prediction'] = pred_res[1]
                    
                    if mode_vis:
                        vis_res = tf_vis(Path(csv_vis_abspath), figs_savedir, False, *tf_models)
                        if tf_nos: response['Tensorflow']['nos']['Visualization'] = {'Figure path': str(PurePosixPath(vis_res[0][0])),
                                                                                     'R2': vis_res[0][1],
                                                                                     'RMSE': vis_res[0][2],
                                                                                     'Sorted predictions': vis_res[0][3]}
                        if tf_std: response['Tensorflow']['std']['Visualization'] = {'Figure path': str(PurePosixPath(vis_res[1][0])),
                                                                                     'R2': vis_res[1][1],
                                                                                     'RMSE': vis_res[1][2],
                                                                                     'Sorted predictions': vis_res[1][3]}
                        
        return jsonify(response)
                
        
        
    else: return 'Not allowed method.'


 
if __name__ == '__main__':
    sklearn_path = Path('./module1_sklearn')
    tf_path = Path('./module2_tf')
    
    
    csv_savedir = Path('./csv')
    figs_savedir = Path('./visualizations')
    for d in [csv_savedir, figs_savedir]: os.makedirs(d, exist_ok=True)
    
    app.run(host='0.0.0.0', port=5000)