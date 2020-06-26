from .imports import *



def predict(regressor, scaler_x_path, scaler_y_path, input_json, mode):
    print('original input_json: {}'.format(input_json))
    input_dict  = json.loads(input_json, object_pairs_hook=OrderedDict)
    print('input_json: {}'.format(input_dict))
    # input_json  = pd.read_json(input_json, typ='series')
    # input_df    = pd.DataFrame([input_json])
    input_df    = pd.DataFrame.from_dict(input_dict, columns=input_dict.keys())
    print(input_df)
    # x_col       = pd.DataFrame(input_df, columns=list(input.columns))
    x_col       = input_df
    
    if mode     == 'nos':
        pred    = regressor.predict(np.array(x_col))

    elif mode   == 'std':
        scaler_x    = pickle.load(open(scaler_x_path, mode='rb'))
        scaler_y    = pickle.load(open(scaler_y_path, mode='rb'))
        np_x_col    = scaler_x.transform(np.array(x_col))
        pred        = regressor.predict(np_x_col)
        pred        = scaler_y.inverse_transform(pred)
    
    print(pred)
    return pred