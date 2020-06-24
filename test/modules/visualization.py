from .imports import *



def visualization(regressor, scaler_x_path, scaler_y_path, input_json, mode):
    input       = json.loads(input_json)
    input       = pd.DataFrame(input)
    input_pred  = input.drop(columns=["day", "moisture_per"])

    x_col       = input_pred
    print(x_col.head(), x_col.shape)
    
    if mode     == 'nos':
        pred    = regressor.predict(np.array(x_col)).flatten()
        pred    = [x for _, x in sorted(zip(input['day'].tolist(), pred.tolist()), key=lambda Zip: Zip[0])]
        return  pred

    elif mode   == 'std':
        scaler_x    = pickle.load(open(scaler_x_path, mode='rb'))
        scaler_y    = pickle.load(open(scaler_y_path, mode='rb'))
        np_x_col    = scaler_x.transform(np.array(x_col))
        pred        = regressor.predict(np_x_col).flatten()
        pred        = scaler_y.inverse_transform(pred)
        pred        = [x for _, x in sorted(zip(input['day'].tolist(), pred.tolist()), key=lambda Zip: Zip[0])]
        return      pred