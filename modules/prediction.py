from .imports import *



def prediction(regressor, scaler_x_path, scaler_y_path, input_json, mode):
    print(input_json, type(input_json))
    input_json  = OrderedDict(input_json)
    print(input_json)
    input_json  = pd.read_json(input_json, typ='series')
    print(input_json, type(input_json))
    input_df    = pd.DataFrame([input_json])
    print(input_df, type(input_df))
    # input_df    = input_df.reindex(columns=list(input_json.keys()))
    
    x_col       = pd.DataFrame(input, columns=list(input.columns))
    print(x_col, type(x_col))
    
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