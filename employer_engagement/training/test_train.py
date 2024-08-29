import pickle
import os
import pandas as pd
import numpy as np
import pyarrow.parquet as pq
import joblib
import datetime
import shap
import math
import xgboost as xgb
import matplotlib.pyplot as plt
from azureml.core.run import Run
from azureml.core import Dataset, Datastore, Model
from azureml.data.datapath import DataPath
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn import datasets, linear_model
from scipy import stats
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

import levy_train_sql_functions as levy_train_functions
import generic_sql_functions as generic_train_functions
import pickle 
# Set up config of workspace and datastore
aml_workspace = Run.get_context().experiment.workspace
#datastore = Datastore.get(aml_workspace, datastore_name='datamgmtdb')
run = Run.get_context()
#run.log('levy_model_train','levy_model_train')

#prevent SettingWithCopyWarning message from appearing
#pd.options.mode.chained_assignment = None

### Get XGBoost version -pkl load might be a bit funky if XGB is different to the model saved
try:
    run.log("INFO: XGB VERSION {}".format(xgb.__version__))
except:
    pass

try:
    models=Model.list(aml_workspace)
    run.log("INFO: Number of models registered: {}".format(len(models)))
    run.log("INFO : List of models: {}".format(str([x.name for x in models])))
    run.log("INFO:  MODEL PATHS: {}".format([x.get_model_path() for x in models]))
except Exception as e:
    run.log("MODEL REGISTRY ERROR: {}".format(e))
    


modelpath="./dummy_model.pkl"
if(os.path.exists(modelpath)):
    try:
        with open(modelpath,'rb') as rf:
            model=pickle.load(rf)
            #check that we have the model at runtime.
            run.log("Model diagnostic: {}".format(str(model)))
    except Exception as E:
        run.log("ERROR: MODEL LOAD: {}".format(E))
        pass 







try:
    # Create model build data into dataframe
    # Create df with all accounts and early adopter flag
   
    run.log('Success 01','Accounts Success')
 
except Exception:
    run.log('EXCEPTION 18','ModelCreationFailed')
