import os, sys, argparse, pickle, re, copy

from flask import Flask, jsonify, request
from flask_cors import CORS
import requests
from werkzeug.utils import secure_filename
import traceback

import numpy as np
import pandas as pd
from sklearn import preprocessing
from renom_rg.api.interface.regressor import Regressor

import warnings
for w in [UserWarning, FutureWarning, DeprecationWarning]:
    warnings.filterwarnings("ignore", category=w)