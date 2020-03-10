import pickle
import numpy as np
import pandas as pd

# 学習用データセットの読み込み
data = pd.read_csv("train.csv")

# 予測用データの読み込み
pred = pd.read_csv("test.csv")

# save as pickle file
with open('./datasrc/data.pickle', mode='wb') as f:
	pickle.dump(data, f)
    
with open('./datasrc/prediction_set/pred.pickle', mode='wb') as f:
	pickle.dump(pred, f)


