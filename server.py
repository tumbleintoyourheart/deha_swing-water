# -*- coding: utf-8 -*- 
from    modules.imports                     import *

from    flask                               import Flask, jsonify, request, Response
from    flask_cors                          import CORS
from    werkzeug.utils                      import secure_filename

import  keras
import  tensorflow                                              as tf
from    tensorflow.python.keras.backend     import set_session
from    tensorflow.python.keras.models      import load_model
from    tensorflow.python.util              import deprecation

from    modules.module1_sklearn.prediction  import predict      as sk_pred
from    modules.module1_sklearn.visualdata  import visualize    as sk_vis
from    modules.module2_tf.prediction       import predict      as tf_pred
from    modules.module2_tf.visualdata       import visualize    as tf_vis
from    modules.module3_heatmap.heatmap     import *



for w in [UserWarning, FutureWarning, DeprecationWarning]:
    warnings.filterwarnings("ignore", category=w)

os.environ['TF_CPP_MIN_LOG_LEVEL']          = '3'
os.environ['KMP_WARNINGS']                  = '0'

deprecation._PRINT_DEPRECATION_WARNINGS     = False

parser                                      = argparse.ArgumentParser()
parser.add_argument('--port', '-p', type=int, default=5000)

app                                         = Flask(__name__); CORS(app)



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
        global             graph, sess
        sess               = tf.Session()
        graph              = tf.get_default_graph()
        set_session(sess)
        
        response            = {}
        
        
        # from request
        values              = request.values.to_dict()
        files               = request.files.to_dict()
        
        # input df
        pred_df                             = values.get('prediction_dataframe')
        vis_df                              = values.get('visualize_dataframe')
        
        # modes
        modes                               = values.get('modes').replace(' ', '').split(',')
        mode_pred                           = True if (('prediction' in modes) and pred_df) else False
        mode_heatmap                        = True if (('heatmap'    in modes) and pred_df) else False
        mode_vis                            = True if (('visualize'  in modes) and vis_df)  else False
        mode_summary                        = True if (('summary'    in modes) and vis_df)  else False
        print(mode_pred, mode_heatmap, mode_vis, mode_summary)
        
        if mode_summary:
            print('summary mode')
            response['Summary'] = {}
            summary_df          = summary(vis_df)
            return              Response(summary_df.to_json(orient="records"), mimetype='application/json')
        
        
        # device_id
        device_id           = values.get('device_id')
        if device_id        == None: return '設備IDを指示してください。'
        
        
        # init models
        sklearn_nos, sklearn_std, tf_nos, tf_std                            = False, False, False, False
        sklearn_nos_model, sklearn_std_model, tf_nos_model, tf_std_model    = None, None, None, None

        available_sk_models  = [m.name for m in list((sklearn_path/'models'/device_id).rglob('*.pickle')) if 'scaler' not in m.name]
        available_tf_models  = [m.name for m in list((tf_path/'models'/device_id).rglob('*.hdf5'))]
        available_models     = available_sk_models + available_tf_models
        
        if values.get('models'):
            model_name                  = values['models']
            
            
            print(model_name)
            if model_name not in available_models:
                print('not available model')
                return f'Not available models. Current available models for device_id {device_id}: {available_models}.'
            
            # nos or std?
            if      'nos'   in model_name:
                model_mode              = 'nos'
            elif    'std'   in model_name:
                model_mode              = 'std'
            # else: return              f''
            print(model_mode)
            
            # sklearn or tf?
            if      'sk'    in model_name:
                module                  = 'sklearn'
                sklearn_model_name      = model_name
                sklearn_model           = pickle.load(open(sklearn_path/'models'/device_id/model_name, 'rb'))
                
            elif    'tf'    in model_name:
                module                  = 'tensorflow'
                tf_model_name           = model_name
                tf_model                = load_model(tf_path/'models'/device_id/model_name)
            # else: return                f''
            print(f'module: {module}')
        print(f'Loaded model: {model_name}')
        


        # init scalers
        sklearn_scaler, tf_scaler           = None, None
        sklearn_scaler_name, tf_scaler_name = None, None
        
        if model_mode                       == 'std':
            if      module                  == 'sklearn':
                sklearn_scaler_name         = f'scaler_of_{sklearn_model_name.split(".")[0]}.pickle'
                sklearn_scaler_path         = sklearn_path/'models'/device_id/sklearn_scaler_name
                if not os.path.isfile(sklearn_scaler_path):
                    return                  f'{sklearn_scaler_name} not found.'
                sklearn_scaler              = pickle.load(open(sklearn_scaler_path, 'rb'))
                print(f'Loaded scaler: {sklearn_scaler_path}')
            elif    module                  == 'tensorflow':
                tf_scaler_name              = f'scaler_of_{tf_model_name.split(".")[0]}.pickle'
                tf_scaler_path              = tf_path/'models'/device_id/tf_scaler_name
                if not os.path.isfile(tf_scaler_path):
                    return                  f'{tf_scaler_name} not found.'
                tf_scaler                   = pickle.load(open(tf_scaler_path, 'rb'))
                print(f'Loaded scaler: {tf_scaler_path}')
        
        
        
        if module == 'sklearn':
            response['Scikit-learn']        = {model_mode: {'Model': sklearn_model_name}}
            
            if mode_pred:
                try:
                    pred_res = sk_pred(pred_df, sklearn_model, sklearn_scaler)
                except Exception as e:
                    tb = traceback.format_exc()
                    print(tb)
                    tb = tb.split('\n')[-2]
                    return jsonify(Error='インポートしたCSVファイルに誤りがあります。')
                
                response['Scikit-learn'][model_mode]['Prediction'] = f'{pred_res[0]:.2f}'

            if mode_vis:
                try:
                    vis_res = sk_vis(vis_df, sklearn_model, sklearn_scaler)
                except Exception as e:
                    tb = traceback.format_exc()
                    print(tb)
                    tb = tb.split('\n')[-2]
                    return jsonify(Error='インポートしたCSVファイルに誤りがあります。')
                
                response['Scikit-learn'][model_mode]['Visualization'] = {'Sorted predictions'   : vis_res['sorted_pred'],
                                                                         'R2'                   : vis_res['r2'],
                                                                         'MAE1'                 : vis_res['mae1'],
                                                                         'MAE2'                 : vis_res['mae2'],
                                                                         'MSE'                  : vis_res['mse'],
                                                                         'RMSE'                 : vis_res['rmse']}
 
            if mode_heatmap:
                response['Scikit-learn'][model_mode].update({'Heatmap': {}, 'Download': {}})
          
                range1                                  = values.get('range1').replace(' ', '').split(',')
                range2                                  = values.get('range2').replace(' ', '').split(',')
                sim_input, sim_name1, sim_name2         = get_sim_input(pred_df, range1, range2)
                sim_df, download_df                     = simulation(sim_input, sklearn_model, sklearn_scaler, sim_name1, sim_name2)
                
                for col in list(sim_df.columns):
                    response['Scikit-learn'][model_mode]['Heatmap'][col]    = sim_df[col].to_numpy().tolist()
                for col in list(download_df.columns):
                    response['Scikit-learn'][model_mode]['Download'][col]   = download_df[col].to_numpy().tolist()
                print(sim_df.head())
                
            return jsonify(response)
                        
        
        elif module == 'tensorflow':
            response['Tensorflow']                  = {model_mode: {'Model': tf_model_name}}

            with graph.as_default():
                set_session(sess)
                
                if mode_pred:
                    try:
                        pred_res = tf_pred(pred_df, tf_model, tf_scaler)
                    except Exception as e:
                        tb = traceback.format_exc()
                        print(tb)
                        tb = tb.split('\n')[-2]
                        return jsonify(Error='インポートしたCSVファイルに誤りがあります。')
                    
                    response['Tensorflow'][model_mode]['Prediction'] = f'{pred_res[0][0]:.2f}'
                
                if mode_vis:
                    try:
                        vis_res = tf_vis(vis_df, tf_model, tf_scaler)
                    except Exception as e:
                        tb = traceback.format_exc()
                        print(tb)
                        tb = tb.split('\n')[-2]
                        return jsonify(Error='インポートしたCSVファイルに誤りがあります。')
                    
                    response['Tensorflow'][model_mode]['Visualization'] = {'Sorted predictions'   : vis_res['sorted_pred'],
                                                                        'R2'                   : vis_res['r2'],
                                                                        'MAE1'                 : vis_res['mae1'],
                                                                        # 'MAE2'                 : vis_res['mae2'],
                                                                        'MSE'                  : vis_res['mse'],
                                                                        'RMSE'                 : vis_res['rmse']}
                    
                if mode_heatmap:
                    response['Tensorflow'][model_mode].update({'Heatmap': {}, 'Download': {}})
            
                    range1                                  = values.get('range1').replace(' ', '').split(',')
                    range2                                  = values.get('range2').replace(' ', '').split(',')
                    sim_input, sim_name1, sim_name2         = get_sim_input(pred_df, range1, range2)
                    sim_df, download_df                     = simulation(sim_input, tf_model, tf_scaler, sim_name1, sim_name2)
                    
                    for col in list(sim_df.columns):
                        response['Tensorflow'][model_mode]['Heatmap'][col]    = sim_df[col].to_numpy().tolist()
                    for col in list(download_df.columns):
                        response['Tensorflow'][model_mode]['Download'][col]   = download_df[col].to_numpy().tolist()
                    print(sim_df.head())
                        
            return jsonify(response)
        

                    
    else: return 'Not allowed method.'


 
if __name__ == '__main__':
    args            = parser.parse_args()
    
    sklearn_path    = Path('./modules/module1_sklearn')
    tf_path         = Path('./modules/module2_tf')
    
    
    csv_savedir     = Path('./csv')
    figs_savedir    = Path('./visualizations')
    for d in [csv_savedir, figs_savedir]: os.makedirs(d, exist_ok=True)
    
    app.run(host='0.0.0.0', port=args.port)