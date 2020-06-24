from .imports import *



def prediction(regressor, scaler_x_path, scaler_y_path, input_json, mode):
    input       = pd.read_json(input_json, typ='series')
    input       = pd.DataFrame([input])
    print(input)
    
    x_col       = pd.DataFrame(input, columns=list(input.columns))
    print(mode)
    print(x_col)
    print(scaler_x_path, scaler_y_path)
    
    if mode     == 'nos':
        pred    = regressor.predict(np.array(x_col))
        return  pred

    elif mode   == 'std':
        scaler_x    = pickle.load(open(scaler_x_path, mode='rb'))
        scaler_y    = pickle.load(open(scaler_y_path, mode='rb'))
        np_x_col    = scaler_x.transform(np.array(x_col))
        pred        = regressor.predict(np_x_col)
        pred        = scaler_y.inverse_transform(pred)
        return      pred