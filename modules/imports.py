import  os, sys, argparse, pickle, re, copy, json, requests, traceback, numpy as np, pandas as pd, warnings

from    flask                               import Flask, jsonify, request
from    flask_cors                          import CORS
from    werkzeug.utils                      import secure_filename
from    itertools                           import product
from    collections                         import OrderedDict 

from    renom_rg.api.interface.regressor    import Regressor
from    sklearn.metrics                     import mean_absolute_error, mean_squared_error, r2_score