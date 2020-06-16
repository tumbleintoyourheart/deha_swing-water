from .imports import *



def get_sim_input(input_json, sim_name1='no2poly_m3_h', sim_range1=(0, 2, 0.025), sim_name2='fe_m3_h', sim_range2=(15, 25, 0.1)):
    input_df                = pd.read_json(input_json, typ='series')
    input_df                = pd.DataFrame([input_df])

    sim_names       = [sim_name1, sim_name2]
    
    sim_range1      = np.arange(*sim_range1)
    sim_range2      = np.arange(*sim_range2)
    sim_df          = pd.DataFrame(list(product(sim_range1, sim_range2)), columns=[sim_name1, sim_name2])

    matrix          = np.random.randn(sim_df.shape[0], input_df.shape[1])
    mat_df          = pd.DataFrame(matrix, columns=list(input_df.columns))
    mat_df          = mat_df.drop(sim_names, axis=1)

    for i in list(mat_df):
        mat_df[i]   = input_df.loc[0, i]

    sim_mat_df      = sim_df.join(mat_df)
    return          sim_mat_df


def simulation(input_df, regressor, scaler_x_path, scaler_y_path, mode, sim_name1='no2poly_m3_h', sim_name2='fe_m3_h'):
    pred_df         = input_df
    
    scaler_x    = pickle.load(open(scaler_x_path, mode='rb'))
    scaler_y    = pickle.load(open(scaler_y_path, mode='rb'))
    
    pred_df_scaled  = scaler_x.transform(np.array(pred_df))     if (mode == 'std') else pred_df
    pred            = regressor.predict(pred_df_scaled)
    pred            = scaler_y.inverse_transform(pred)          if (mode == 'std') else pred
    pred_df['pred'] = pred
    
    return          pred_df[[sim_name1, sim_name2, 'pred']], pred_df