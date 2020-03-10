import os, sys, argparse, pickle
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
    
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['KMP_WARNINGS'] = '0'

from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False


app = Flask(__name__)

        

def init():
    sklearn_models = [pickle.load(open(sklearn_path/model_path, 'rb')) for model_path in ['rf_model.pickle', 'scaler.pickle', 'rfstd_model.pickle']]
    
    global graph, sess
    tf_scaler = pickle.load(open(tf_path/'scaler.pickle', 'rb'))
    sess = tf.Session()
    graph = tf.get_default_graph()
    set_session(sess)
    tf_model = load_model(tf_path/'nn_model.hdf5')
    tf_models = [tf_scaler, tf_model]
    
    return sklearn_models, tf_models
  
    
    
def get_input(files, key):
    if files.get(key):
        csv_input = files[key]
        csv_name = secure_filename(csv_input.filename)
        abs_path = os.path.join(save_path, csv_name)
        csv_input.save(abs_path)
        
        return True, abs_path
    
    else: return False, None
                
                
@app.route('/api_ai', methods=['GET', 'POST'])
def api():
    if request.method == 'POST':
        response = {}
        
        values = request.values.to_dict()
        module = values.get('module')
        
        files = request.files.to_dict()
        mode_pred, pred_abs_path = get_input(files, 'csv_prediction')
        mode_vis, vis_abs_path = get_input(files, 'csv_visual')
        
        if module == 'sklearn':
            if mode_pred:
                pred_res = sklearn_pred(Path(pred_abs_path), *sklearn_models)
                pred_res = [f'{res[0]:.2f} %' for res in pred_res]
                response['Prediction'] = {'Unnormalized': pred_res[0],
                                          'Normalized': pred_res[1]}

            if mode_vis:
                vis_res = sklearn_vis(Path(vis_abs_path), figs_savedir, False, *sklearn_models)
                response['Visualization'] = {'Unnormalized': {'Figure path': str(PurePosixPath(vis_res[0][0])),
                                                              'R2': vis_res[0][1],
                                                              'RMSE': vis_res[0][2]},
                                             'Normalized': {'Figure path': str(PurePosixPath(vis_res[1][0])),
                                                            'R2': vis_res[1][1],
                                                            'RMSE': vis_res[1][2]}}
            return jsonify(response)

        elif module == 'tensorflow':
            global graph, sess
            with graph.as_default():
                set_session(sess)
                
                if mode_pred:
                    pred_res = tf_pred(Path(pred_abs_path), *tf_models)
                    pred_res = f'{pred_res[0][0]:.2f} %'
                    response['Prediction'] = {'Normalized': pred_res}
                
                if mode_vis:
                    vis_res = tf_vis(Path(vis_abs_path), figs_savedir, False, *tf_models)
                    response['Visualization'] = {'Normalized': {'Figure path': str(PurePosixPath(vis_res[0])),
                                                                'R2': vis_res[1],
                                                                'RMSE': vis_res[2]}}
                return jsonify(response)
                
        else: return 'module must be either sklearn or tensorflow.'
    else: return 'Not allowed method.'


 
if __name__ == '__main__':
    sklearn_path = Path('./module1_sklearn')
    tf_path = Path('./module2_tf')
    
    save_path = Path('./upload')
    os.makedirs(save_path, exist_ok=True)
    
    figs_savedir = Path('./visualizations')
    os.makedirs(figs_savedir, exist_ok=True)
    
    sklearn_models, tf_models = init()
    app.run(host='0.0.0.0', port=5000)