from    ..imports       import *


def visualize(input_json, model, scaler):
    input_json          = json.loads(input_json)
    input_df            = pd.DataFrame(input_json)
    input_pred          = input_df.drop(columns=["date", "objective_variable"])
    input_pred_scaled   = scaler.transform(input_pred) if scaler != None else input_pred
    
    pred                = model.predict(input_pred_scaled)
    sorted_pred         = [x[0] for _, x in sorted(zip(input_df['date'].tolist(), pred.tolist()), key=lambda Zip: Zip[0])]

    r2                  = round(r2_score(input_df["objective_variable"], pred), 2)
    mae1                = round(mean_absolute_error(input_df["objective_variable"], pred), 2)
    mae2                = round(max(abs((input_df["objective_variable"] - pred.flatten()))), 2)
    mse                 = round(mean_squared_error(input_df["objective_variable"], pred), 2)
    rmse                = round(np.sqrt(mean_squared_error(input_df["objective_variable"], pred)), 2)
    
    return {'sorted_pred': sorted_pred, 'r2': r2,
             'mae1': mae1, 
             'mae2': mae2, 
             'mse': mse, 
             'rmse': rmse}