import  pickle
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



def get_sim_input(input_df, sim_range1=(0, 2, 0.025), sim_range2=(15, 25, 0.1)):
    # input_df              = pd.read_csv(Path(input_df))
    input_df                = input_df
    input_df["ts_mg_L"]     = 30000

    sim_name1       = "no2poly_m3_h"
    sim_name2       = "fe_m3_h"
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
    sim_mat_df.to_csv('./simulation.csv')
    return          sim_mat_df


def simulation(input_df, model, scaler, show=False):
    pred_df         = input_df
    
    if scaler != None:
        pred_df     = scaler.transform(pred_df)
        
    pred            = model.predict(pred_df)
    pred_df['pred'] = pred
    
    if show:
        sim_name1   = "no2poly_m3_h"
        sim_name2   = "fe_m3_h"

        pred_df.plot(
                    kind="scatter",
                    x=sim_name1,
                    y=sim_name2,
                    alpha=0.4,
                    s=pred,
                    label="moisture%",
                    figsize=(10,7),
                    cmap=plt.get_cmap("jet"),
                    colorbar=True,
                    c="pred"
                    )
        plt.legend()
        plt.show()
    
    return          pred_df


def summary(input_df, model, scaler, show=False):
    pred_df     = input_df.drop(columns=['day', 'moisture_per'])
    
    if scaler != None:
        pred_df = scaler.transform(pred_df)
        
    pred        = model.predict(pred_df)

    if show:
        plt.scatter(input_df['moisture_per'], pred)
        plt.show()
        plt.clf()

    score       = round(r2_score(input_df['moisture_per'], pred), 2)
    rmse        = round(np.sqrt(mean_squared_error(input_df['moisture_per'], pred)), 2)

    summary_df  = input_df.drop(columns=['day'])
    return      round(summary_df.describe(), 2)


if __name__=='__main__':
    sim_input       = get_sim_input('./200302_A01_prediction.csv')
    model           = pickle.load(open('./200310_atg_dsp_sk_rf_std.pickle', 'rb'))
    scaler          = pickle.load(open('./scaler.pickle', 'rb'))
    simulation(sim_input, model, scaler, True)