# -*- coding: utf-8 -*- 
import  os, sys, argparse, pickle, re, copy, argparse, warnings
from    pathlib                             import *

from    flask                               import Flask, jsonify, request
from    flask_cors                          import CORS
from    werkzeug.utils                      import secure_filename
import  traceback

from    module1_sklearn.prediction          import predict      as sklearn_pred
from    module1_sklearn.visualdata          import visualize    as sklearn_vis

from    module2_tf.prediction               import predict      as tf_pred
from    module2_tf.visualdata               import visualize    as tf_vis

from    module3_heatmap.heatmap             import *

import  keras
import  tensorflow                                              as tf
from    tensorflow.python.keras.backend     import set_session
from    tensorflow.python.keras.models      import load_model
from    tensorflow.python.util              import deprecation



for w in [UserWarning, FutureWarning, DeprecationWarning]: warnings.filterwarnings("ignore", category=w)

os.environ['TF_CPP_MIN_LOG_LEVEL']      = '3'
os.environ['KMP_WARNINGS']              = '0'

deprecation._PRINT_DEPRECATION_WARNINGS = False

parser  = argparse.ArgumentParser()
parser.add_argument('--port', '-p', type=int, default=5000)

app     = Flask(__name__); CORS(app)



def get_input(files, key):
    if files.get(key):
        inp         = files[key]
        inp_name    = secure_filename(inp.filename)
        return      inp, inp_name
    else: return    None, None
    
    
def save_input(inp, inp_name, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    abs_path        = os.path.join(save_dir, inp_name)
    inp.save(abs_path)
    return          abs_path
                


@app.route('/upload_model', methods=['GET', 'POST'])
def new_model():
    if request.method       == 'POST':
        # values
        values              = request.values.to_dict()
        device_id           = values.get('device_id')
        # if device_id == None: return f'Please specify device_id.'
        if device_id        == None: return 'モデルIDを指示してください。'
        
        # files
        files               = request.files.to_dict()
        model, model_name   = get_input(files, 'model')
        scaler, _           = get_input(files, 'scaler')
        
        
        # cases
        # if model == None: return 'Please specify a model to upload.'
        if model            == None: return 'モデルを指示してください。'
        
        if [('std' in model_name), scaler != None].count(True) == 1:
            # return 'std model and scaler must come together. Please make sure to upload both.'
            return          '標準化ファイルも指示してください。'
        
        scaler_name         = f'scaler_of_{model_name.split(".")[0]}.pickle'
        if 'sk' in model_name:
            save_dir        = sklearn_path/'models'/device_id
            save_input(model, model_name, save_dir)
            if scaler != None: save_input(scaler, scaler_name, save_dir)
        elif 'tf' in model_name:
            save_dir        = tf_path/'models'/device_id
            save_input(model, model_name, save_dir)
            if scaler != None: save_input(scaler, scaler_name, save_dir)

        return              f'{model_name}'
    else: return            'Not allowed method.'
    
    
    
@app.route('/ai', methods=['GET', 'POST'])
def ai():
    if request.method       == 'POST':
        response            = {}
        
        
        # from request
        values              = request.values.to_dict()
        files               = request.files.to_dict()
        
        
        # device_id
        device_id           = values.get('device_id')
        if device_id        == None: return '設備IDを指示してください。'
        
        
        # init models
        sklearn_nos, sklearn_std, tf_nos, tf_std                            = False, False, False, False
        sklearn_nos_model, sklearn_std_model, tf_nos_model, tf_std_model    = None, None, None, None
        modules, model_modes                                                = set(), set()
        available_sk_models     = [m.name for m in list((sklearn_path/'models'/device_id).rglob('*.pickle')) if 'scaler' not in m.name]
        available_tf_models     = [m.name for m in list((tf_path/'models'/device_id).rglob('*.hdf5'))]
        available_models        = available_sk_models + available_tf_models
        
        if values.get('models'):
            models          = values['models'].replace(' ', '').split(',')
            
            global          graph, sess
            sess            = tf.Session()
            graph           = tf.get_default_graph()
            set_session(sess)
            
            for m in models:
                if m not in available_models: return f'Not available models. Current available models for device_id {device_id}: {available_models}.'
                
                elif re.search(r'sk\w*nos.pickle', m):
                    sklearn_nos             = True
                    modules.add('sklearn')
                    model_modes.add('nos')
                    sklearn_nos_model       = pickle.load(open(sklearn_path/'models'/device_id/m, 'rb'))
                    sklearn_nos_model_name  = m
                    
                elif re.search(r'sk\w*std.pickle', m):
                    sklearn_std             = True
                    modules.add('sklearn')
                    model_modes.add('std')
                    sklearn_std_model       = pickle.load(open(sklearn_path/'models'/device_id/m, 'rb'))
                    sklearn_std_model_name  = m
                    
                elif re.search(r'tf\w*nos.hdf5', m):
                    tf_nos                  = True
                    modules.add('tensorflow')
                    model_modes.add('nos')
                    tf_nos_model            = load_model(tf_path/'models'/device_id/m)
                    tf_nos_model_name       = m
                    
                elif re.search(r'tf\w*std.hdf5', m):
                    tf_std                  = True
                    modules.add('tensorflow')
                    model_modes.add('std')
                    tf_std_model            = load_model(tf_path/'models'/device_id/m)
                    tf_std_model_name       = m
        else: return                        'モデルを指示してください。'

        
        # init scalers
        if 'sklearn' in modules:
            if sklearn_std:
                sklearn_scaler_path         = sklearn_path/'models'/device_id/f'scaler_of_{sklearn_std_model_name.split(".")[0]}.pickle'
                if not os.path.isfile(sklearn_scaler_path):
                    return                  f'scaler_of_{sklearn_std_model_name.split(".")[0]}.pickle not found.'
                sklearn_scaler              = pickle.load(open(sklearn_scaler_path, 'rb'))
            else: sklearn_scaler            = None
            sklearn_models                  = [sklearn_scaler, sklearn_nos_model, sklearn_std_model]
            
        if 'tensorflow' in modules:
            if tf_std:
                tf_scaler_path              = tf_path/'models'/device_id/f'scaler_of_{tf_std_model_name.split(".")[0]}.pickle'
                if not os.path.isfile(tf_scaler_path):
                    return                  f'scaler_of_{tf_std_model_name.split(".")[0]}.pickle not found.'
                tf_scaler                   = pickle.load(open(tf_scaler_path, 'rb'))
            else: tf_scaler                 = None
            tf_models                       = [tf_scaler, tf_nos_model, tf_std_model]
        
        
        # input dataframes
        pred_df                             = values.get('prediction_dataframe')
        vis_df                              = values.get('visualize_dataframe')
        
        # modes
        modes                               = values.get('modes').replace(' ', '').split(',')
        mode_pred                       = True if (('prediction' in modes) and pred_df) else False
        mode_heatmap                    = True if (('heatmap'    in modes) and pred_df) else False
        
        mode_vis                        = True if (('visualize'  in modes) and vis_df)  else False
        mode_summary                    = True if (('summary'    in modes) and vis_df)  else False
        print(mode_pred, mode_heatmap, mode_vis, mode_summary)
        
        
        for module in modules:
            if module == 'sklearn':
                response['Scikit-learn'] = {'nos': {},
                                            'std': {}}
                
                if sklearn_nos: response['Scikit-learn']['nos']['Model'] = sklearn_nos_model_name
                if sklearn_std: response['Scikit-learn']['std']['Model'] = sklearn_std_model_name
                
                if mode_pred:
                    try:
                        pred_res = sklearn_pred(pred_df, model_modes, *sklearn_models)
                    except Exception as e:
                        tb = traceback.format_exc()
                        print(tb)
                        tb = tb.split('\n')[-2]
                        return jsonify(Error='インポートしたCSVファイルに誤りがあります。')
                    def beautify(res): 
                        if res != None: return f'{res[0]:.2f}'
                        else: return res
                    pred_res = list(map(beautify, pred_res))
                    if sklearn_nos: response['Scikit-learn']['nos']['Prediction'] = pred_res[0]
                    if sklearn_std: response['Scikit-learn']['std']['Prediction'] = pred_res[1]

                if mode_vis:
                    try:
                        vis_res = sklearn_vis(vis_df, model_modes, figs_savedir, False, *sklearn_models)
                    except Exception as e:
                        tb = traceback.format_exc()
                        print(tb)
                        tb = tb.split('\n')[-2]
                        return jsonify(Error='インポートしたCSVファイルに誤りがあります。')
                    if sklearn_nos:
                        response['Scikit-learn']['nos']['Visualization'] = {'Figure path': str(PurePosixPath(vis_res[0][0])),
                                                                'R2': vis_res[0][1],
                                                                'RMSE': vis_res[0][2],
                                                                'Sorted predictions': vis_res[0][3]}
                    if sklearn_std:
                        response['Scikit-learn']['std']['Visualization'] = {'Figure path': str(PurePosixPath(vis_res[1][0])),
                                                                'R2': vis_res[1][1],
                                                                'RMSE': vis_res[1][2],
                                                                'Sorted predictions': vis_res[1][3]}
                
                if mode_heatmap:
                    sim_range1  = values.get('sim_range1')
                    sim_range2  = values.get('sim_range2')
                    sim_input   = get_sim_input(pred_df, sim_range1, sim_range2)
                    
                    if sklearn_nos_model != None:
                        response['Scikit-learn']['nos']['Heatmap'] = simulation(sim_input, sklearn_nos_model, None)
                    if sklearn_std_model != None:
                        response['Scikit-learn']['std']['Heatmap'] = simulation(sim_input, sklearn_std_model, sklearn_scaler)
                        
                
                if mode_summary:
                    if sklearn_nos_model != None:
                        response['Scikit-learn']['nos']['Summary'] = summary(vis_df, sklearn_nos_model, None)
                    if sklearn_std_model != None:
                        response['Scikit-learn']['std']['Summary'] = summary(vis_df, sklearn_std_model, sklearn_scaler)
                        
                        
        
            elif module == 'tensorflow':
                response['Tensorflow'] = {'nos': {},
                                          'std': {}}
                
                if tf_nos: response['Tensorflow']['nos']['Model'] = tf_nos_model_name
                if tf_std: response['Tensorflow']['std']['Model'] = tf_std_model_name
                
                with graph.as_default():
                    set_session(sess)
                    
                    if mode_pred:
                        try:
                            pred_res = tf_pred(pred_df, model_modes, *tf_models)
                        except Exception as e:
                            tb = traceback.format_exc()
                            print(tb)
                            tb = tb.split('\n')[-2]
                            return jsonify(Error='インポートしたCSVファイルに誤りがあります。')
                        def beautify(res): 
                            if res != None: return f'{res[0][0]:.2f}'
                            else: return res
                        pred_res = list(map(beautify, pred_res))
                        if tf_nos: response['Tensorflow']['nos']['Prediction'] = pred_res[0]
                        if tf_std: response['Tensorflow']['std']['Prediction'] = pred_res[1]
                    
                    if mode_vis:
                        try:
                            vis_res = tf_vis(vis_df, model_modes, figs_savedir, False, *tf_models)
                        except Exception as e:
                            tb = traceback.format_exc()
                            print(tb)
                            tb = tb.split('\n')[-2]
                            return jsonify(Error='インポートしたCSVファイルに誤りがあります。')
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
    args            = parser.parse_args()
    
    sklearn_path    = Path('./module1_sklearn')
    tf_path         = Path('./module2_tf')
    
    
    csv_savedir     = Path('./csv')
    figs_savedir    = Path('./visualizations')
    for d in [csv_savedir, figs_savedir]: os.makedirs(d, exist_ok=True)
    
    app.run(host='0.0.0.0', port=args.port)