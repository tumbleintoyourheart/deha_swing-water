from    ..imports       import *



def predict(input_json, model, scaler):
    input_json          = pd.read_json(input_json, typ='series')
    input_df            = pd.DataFrame([input_json])
    print(input_df)
    input_pred          = input_df
    input_pred_scaled   = scaler.transform(input_pred) if scaler != None else input_pred
    
    pred                = model.predict(input_pred_scaled)
    print(pred)
    
    return              pred