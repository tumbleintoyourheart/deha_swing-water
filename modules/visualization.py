from .imports import *



def visualize(regressor, scaler_x_path, scaler_y_path, input_json, mode):
    input_json      = json.loads(input_json)
    input_df        = pd.DataFrame(input_json)
    input_pred      = input_df.drop(columns=["day", "moisture_per"])

    x_col           = input_pred
    print(x_col.head(), x_col.shape)
    
    if mode         == 'nos':
        pred        = regressor.predict(np.array(x_col)).flatten()
        sorted_pred = [x for _, x in sorted(zip(input_df['day'].tolist(), pred.tolist()), key=lambda Zip: Zip[0])]

    elif mode       == 'std':
        scaler_x    = pickle.load(open(scaler_x_path, mode='rb'))
        scaler_y    = pickle.load(open(scaler_y_path, mode='rb'))
        np_x_col    = scaler_x.transform(np.array(x_col))
        pred        = regressor.predict(np_x_col).flatten()
        pred        = scaler_y.inverse_transform(pred)
        sorted_pred = [x for _, x in sorted(zip(input_df['day'].tolist(), pred.tolist()), key=lambda Zip: Zip[0])]
    
    r2              = round(r2_score(input_df["moisture_per"], pred), 2)
    mae1            = round(mean_absolute_error(input_df["moisture_per"], pred), 2)
    mae2            = round(max(abs(input_df["moisture_per"] - pred)), 2)
    mse             = round(mean_squared_error(input_df["moisture_per"], pred), 2)
    rmse            = round(np.sqrt(mean_squared_error(input_df["moisture_per"], pred)), 2)
    
    return {'sorted_pred': sorted_pred, 'r2': r2, 'mae1': mae1, 
            'mae2': mae2, 
            'mse': mse, 'rmse': rmse}