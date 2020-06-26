import  pickle, json
import  pandas                      as pd
import  numpy                       as np
import  matplotlib.pyplot           as plt
from    itertools                   import product
from    pathlib                     import *

from    sklearn.model_selection     import train_test_split
from    sklearn.ensemble            import RandomForestRegressor
from    sklearn.metrics             import mean_absolute_error, mean_squared_error, r2_score
from    sklearn.preprocessing       import StandardScaler
from    sklearn.externals           import joblib



def get_sim_input(input_json, range1, range2):
    input_df                = pd.read_json(input_json, typ='series')
    input_df                = pd.DataFrame([input_df])
    print(f'input_df columns: {input_df.columns}')

    sim_name1, sim_range1   = range1[0], [float(x) for x in range1[1:]]
    sim_name2, sim_range2   = range2[0], [float(x) for x in range2[1:]]
    
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
    sim_mat_df       = sim_mat_df.loc[:, input_df.columns]
    sim_mat_df.to_csv('./modules/module3_heatmap/simulation.csv')
    return          sim_mat_df, sim_name1, sim_name2


def simulation(input_df, model, scaler, sim_name1, sim_name2):
    print(f'input_df columns: {input_df.columns}')
    pred_df         = input_df

    pred_df_scaled  = scaler.transform(pred_df) if (scaler != None) else pred_df
    pred            = model.predict(pred_df_scaled)
    pred_df['pred'] = pred

    return          pred_df[[sim_name1, sim_name2, 'pred']], pred_df

def summary(input_json):
    input_json              = json.loads(input_json)
    input_df                = pd.DataFrame(input_json)
    input_df                = input_df.drop(columns=["day"])
    summary_df              = round(input_df.describe(), 2)
    summary_df.index.name   = 'category'
    summary_df.reset_index(inplace=True)
    
    print(summary_df)
    return              summary_df