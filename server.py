import os, sys, argparse, pickle, re
from pathlib import *

from flask import Flask, jsonify, request
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
        
        if 'csv' in inp_name: save_path = Path('./csv')
        elif 'sk' in inp_name: save_path = Path(sklearn_path/'models')
        elif 'tf' in inp_name: save_path = Path(tf_path/'models')
        os.makedirs(save_path, exist_ok=True)
        
        abs_path = os.path.join(save_path, inp_name)
        inp.save(abs_path)
        
        return True, inp_name, abs_path
    
    else: return False, None, None
                


@app.route('/upload_model', methods=['GET', 'POST'])
def new_model():
    if request.method == 'POST':
        files = request.files.to_dict()
        _, name, abs_path = get_input(files, 'new_model')
        
        return f'Successfully uploaded {name}'
        
    else: return 'Not allowed method.'
    
    
    
@app.route('/ai', methods=['GET', 'POST'])
def ai():
    if request.method == 'POST':
        # final response
        response = {'Results': {'nos': {},
                                'std': {}}}
        
        
        values = request.values.to_dict()
        # module to use, either sklearn or tensorflow
        if values.get('module'): module = values['module']
        else: return 'No module specified. Allowed ones: sklearn, tensorflow.'
        
        # check if user want to use new models instead of the defaults
        global sklearn_default_models, tf_default_models
        if values.get('models'):
            models = values['models'].replace(' ', '').split(',')
            for m in models:
                if re.search(r'sk\w*nos', m): sklearn_default_models[0] = m
                elif re.search(r'sk\w*std', m): sklearn_default_models[1] = m
                elif re.search(r'tf\w*nos', m): tf_default_models[0] = m
                elif re.search(r'tf\w*std', m): tf_default_models[1] = m
        
        
        # init models
        sklearn_models, tf_models = init()
        
        
        # request files
        files = request.files.to_dict()
        mode_pred, _, pred_abs_path = get_input(files, 'csv_prediction')
        mode_vis, _, vis_abs_path = get_input(files, 'csv_visual')
        
        if module == 'sklearn':
            response['Models'] = {'nos': sklearn_default_models[0],
                                  'std': sklearn_default_models[1]}
            if mode_pred:
                pred_res = sklearn_pred(Path(pred_abs_path), *sklearn_models)
                pred_res = [f'{res[0]:.2f} %' for res in pred_res]
                response['Results']['nos']['Prediction'] = pred_res[0]
                response['Results']['std']['Prediction'] = pred_res[1]

            if mode_vis:
                vis_res = sklearn_vis(Path(vis_abs_path), figs_savedir, False, *sklearn_models)
                response['Results']['nos']['Visualization'] = {'Figure path': str(PurePosixPath(vis_res[0][0])),
                                                               'R2': vis_res[0][1],
                                                               'RMSE': vis_res[0][2]}
                response['Results']['std']['Visualization'] = {'Figure path': str(PurePosixPath(vis_res[1][0])),
                                                               'R2': vis_res[1][1],
                                                               'RMSE': vis_res[1][2]}
            return jsonify(response)

        elif module == 'tensorflow':
            response['Models'] = {'nos': tf_default_models[0],
                                  'std': tf_default_models[1]}
            global graph, sess
            with graph.as_default():
                set_session(sess)
                
                if mode_pred:
                    pred_res = tf_pred(Path(pred_abs_path), *tf_models)
                    pred_res = [f'{res[0][0]:.2f} %' for res in pred_res]
                response['Results']['nos']['Prediction'] = pred_res[0]
                response['Results']['std']['Prediction'] = pred_res[1]
                
                if mode_vis:
                    vis_res = tf_vis(Path(vis_abs_path), figs_savedir, False, *tf_models)
                    response['Results']['nos']['Visualization'] = {'Figure path': str(PurePosixPath(vis_res[0][0])),
                                                                'R2': vis_res[0][1],
                                                                'RMSE': vis_res[0][2]}
                    response['Results']['std']['Visualization'] = {'Figure path': str(PurePosixPath(vis_res[1][0])),
                                                                'R2': vis_res[1][1],
                                                                'RMSE': vis_res[1][2]}
            return jsonify(response)
                
        else: return 'module must be either sklearn or tensorflow.'
        
    else: return 'Not allowed method.'


 
if __name__ == '__main__':
    sklearn_path = Path('./module1_sklearn')
    tf_path = Path('./module2_tf')
    
    figs_savedir = Path('./visualizations')
    os.makedirs(figs_savedir, exist_ok=True)
    
    app.run(host='0.0.0.0', port=5000)