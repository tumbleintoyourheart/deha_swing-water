from modules.imports        import *
from modules.prediction     import predict
from modules.visualization  import visualize
from modules.heatmap        import *
from modules.utils          import *



app = Flask(__name__)
CORS(app)

for w in [UserWarning, FutureWarning, DeprecationWarning]:
    warnings.filterwarnings("ignore", category=w)


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
            return jsonify(Error='Renomサーバーに該当するモデルがまだ用意されていません。')
        
        return 'Successfully initialized Regressor with model_id={}.'.format(model_id)
        
    else: return 'Not allowed method.'
    


@app.route('/upload_scaler', methods=['GET', 'POST'])
def new_scaler():
    if request.method == 'POST':
        values                          = request.values.to_dict()
        files                           = request.files.to_dict()
        
        model_id                        = values.get('model_id')
        if model_id                     == None: return 'モデルIDを指示してください。'
        device_id                       = values.get('device_id')
        if device_id                    == None: return '設備IDを指示してください。'
        
        
        scaler_x, scaler_x_name         = get_input(files, 'scaler_x')
        scaler_y, scaler_y_name         = get_input(files, 'scaler_y')
        save_dir                        = ('./{}').format(device_id)
        
        if scaler_x                     != None:
            if scaler_x_name            != 'scaler_x.pickle': return '標準化ファイルXは不正です。'
            scaler_x_name               = 'scaler_x_of_model_id_{}.pickle'.format(model_id)
            save_input(scaler_x, scaler_x_name, save_dir)
        else: return                    '標準化ファイルXは必須です。'
        
        if scaler_y                     != None:
            if scaler_y_name            != 'scaler_y.pickle': return '標準化ファイルYは不正です。'
            scaler_y_name = 'scaler_y_of_model_id_{}.pickle'.format(model_id)
            save_input(scaler_y, scaler_y_name, save_dir)
        else: return                    '標準化ファイルYは必須です。'
        
        return                          '{}, {}'.format(scaler_x_name, scaler_y_name)
    else: return                        'Not allowed method.'
    
    
    
@app.route('/renom_ai', methods=['GET', 'POST'])
def ai():
    if request.method == 'POST':
        response                = {}
        
        # from request
        values                  = request.values.to_dict()
        files                   = request.files.to_dict()
        
        
        # check
        model_id                = values.get('model_id')
        if model_id             == None: return 'モデルIDを指示してください。'
        try:
            print('model_id: {}'.format(model_id))
            regressor           = models[model_id]
        except Exception as e:
            tb                  = traceback.format_exc()
            print(tb)
            return jsonify(Error='モデルを実装してください。')
        
        device_id               = values.get('device_id')
        if device_id            == None: return '設備IDを指示してください。'
        
        
        mode                    = values.get('mode')
        if mode                 == 'std':
            scaler_x_path       = './{}/scaler_x_of_model_id_{}.pickle'.format(device_id, model_id)
            if not os.path.isfile(scaler_x_path):
                return          '{}に該当する標準化ファイルXは見つかりません。'.format(device_id)
            
            scaler_y_path       = './{}/scaler_y_of_model_id_{}.pickle'.format(device_id, model_id)
            if not os.path.isfile(scaler_y_path):
                return          '{}に該当する標準化ファイルYは見つかりません。'.format(device_id)
            
            scalers_path        = [scaler_x_path, scaler_y_path]
        else: scalers_path      = [None, None]
        
        

        
        # input dataframes
        pred_df                 = values.get('prediction_dataframe')
        vis_df                  = values.get('visualize_dataframe')
        
        # modes
        modes                   = values.get('modes').replace(' ', '').split(',')
        mode_pred               = True if (('prediction' in modes) and pred_df) else False
        mode_heatmap            = True if (('heatmap'    in modes) and pred_df) else False
        
        mode_vis                = True if (('visualize'  in modes) and vis_df)  else False
        mode_summary            = True if (('summary'    in modes) and vis_df)  else False
        print(mode_pred, mode_heatmap, mode_vis, mode_summary)
        
        
        response['Renom']       = {mode: {}}
        if mode_pred:
            try:
                pred_res        = predict(regressor, *scalers_path, pred_df, mode)
            except Exception as e:
                tb              = traceback.format_exc()
                print(tb)
                tb              = tb.split('\n')[-2]
                return          jsonify(Error='インポートしたCSVファイルに誤りがあります。')
            pred_res            = '{:.2f}'.format(pred_res[0][0])
            response['Renom'][mode]['Prediction'] = pred_res

        if mode_vis:
            try:
                vis_res         = visualize(regressor, *scalers_path, vis_df, mode)
            except Exception as e:
                tb              = traceback.format_exc()
                print(tb)
                tb              = tb.split('\n')[-2]
                return          jsonify(Error='インポートしたCSVファイルに誤りがあります。')
            response['Renom'][mode]['Visualization'] = {'Sorted predictions'   : vis_res['sorted_pred'],
                                                        'R2'                   : vis_res['r2'],
                                                        'MAE1'                 : vis_res['mae1'],
                                                        'MAE2'                 : vis_res['mae2'],
                                                        'MSE'                  : vis_res['mse'],
                                                        'RMSE'                 : vis_res['rmse']}
                
        if mode_heatmap:
            response['Renom'][mode].update({'Heatmap': {}, 'Download': {}})

            
            range1                              = values.get('range1').replace(' ', '').split(',')
            range2                              = values.get('range2').replace(' ', '').split(',')
            sim_input, sim_name1, sim_name2     = get_sim_input(pred_df, range1, range2)
            
            sim_df, download_df                 = simulation(sim_input, regressor, *scalers_path, mode, sim_name1, sim_name2)
    
            for col in list(sim_df.columns):
                response['Renom'][mode]['Heatmap'][col]     = sim_df[col].to_numpy().tolist()
            for col in list(download_df.columns):
                response['Renom'][mode]['Download'][col]    = download_df[col].to_numpy().tolist()

                
        return                  jsonify(response)
    else: return                'Not allowed method.'


@app.route('/', methods=['GET', 'POST'])
def hello():
    return 'hello'

if __name__ == '__main__':
    csv_savedir = ('./csv')
    figs_savedir = ('./visualizations')
    for d in [csv_savedir, figs_savedir]: os.makedirs(d, exist_ok=True)
    
    models = {}
    app.run(host='0.0.0.0', port=5000)