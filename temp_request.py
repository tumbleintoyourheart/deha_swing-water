import pickle
import numpy as np
import renom as rm
import pandas as pd
import requests
from sklearn import preprocessing
from renom_rg.api.interface.regressor import Regressor

url = '0.0.0.0:80/'
requests.post(url)