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
import test_train_sql_functions as test_train_functions
import pickle 
import time
# Set up config of workspace and datastore
aml_workspace = Run.get_context().experiment.workspace
#datastore = Datastore.get(aml_workspace, datastore_name='datamgmtdb')
run = Run.get_context()
#run.log('levy_model_train','levy_model_train')

#prevent SettingWithCopyWarning message from appearing
#pd.options.mode.chained_assignment = None

### Get XGBoost version -pkl load might be a bit funky if XGB is different to the model saved
try:
    run.log("INFO 1","INFO: XGB VERSION {}".format(xgb.__version__))
except:
    pass

try:
    models=Model.list(aml_workspace)
    run.log("INFO 2","INFO: Number of models registered: {}".format(len(models)))
    run.log("INFO 3","INFO : List of models: {}".format(str([x.name for x in models])))
    run.log("INFO 4","INFO:  MODEL PATHS: {}".format([Model.get_model_path(x.name) for x in models]))
    #Get latest model
    model=models[-1]
    model.download(target_dir=".",exist_ok=False)
    run.log("INFO 5", "MODEL DOWNLOADED SUCCESSFULLY")
except Exception as e:
    run.log("EXCEPTION 1","MODEL REGISTRY ERROR: {}".format(e))
    


modelpath="./dummy_model.pkl"
if(os.path.exists(modelpath)):
    try:
        with open(modelpath,'rb') as rf:
            model=pickle.load(rf)
            #check that we have the model at runtime.
            run.log("DIAGNOSTIC 1","Model diagnostic: {}".format(str(model)))
    except Exception as E:
        run.log('EXCEPTION 2',"ERROR: MODEL LOAD: {}".format(E))
        pass 
try:
    df_out=test_train_functions.test_train_sql_exec_PDS_view(str(100))
    run.log("INFO 6A","Columns: {}".format(str(list(df_out.columns))))
    run.log("INFO 7A","Number of rows: {}".format(len(df_out)))
except Exception as E:
    run.log("EXCEPTION 3A:","DATASTORE LOAD: {}".format(E))
    pass

# try:
#     df_out=test_train_functions.test_train_sql_exec(str(100))
#     run.log("INFO 6","Columns: {}".format(str(list(df_out.columns))))
#     run.log("INFO 7","Number of rows: {}".format(len(df_out)))
# except Exception as E:
#     run.log("EXCEPTION 3:","DATASTORE LOAD: {}".format(E))
#     pass


#write a dummy dataframe to CSV
try:
    np.random.seed=42
    v1=np.random.randn(400)
    v2=[]
    for x in range(0,len(v1)):
        v2.append(np.random.choice(['Email A', 'Email B']))
    df_proc=pd.DataFrame({'ApprenticeshipId':v1,
                'Email Classification': v2              
                })
    currtime=datetime.datetime.now()
    hh=currtime.hour
    mm=currtime.minute
    DD=currtime.day
    MM=currtime.month
    YYYY=currtime.year
    fname_base=f'WithdrawalRateAIOutput_{mm}_{hh}_{DD}_{MM}_{YYYY}'
    if(not os.path.exists("./outputs/")):
        os.mkdir("outputs/")
    
    df_proc.to_csv("./outputs/"+fname_base+".csv")
    df_proc.to_parquet("./outputs/"+fname_base+".parquet")
    run.log("INFO 8", "DATA SAVED TO DISK")
except Exception as P:
    run.log("EXCEPTION 4", "Exception: {}".format(P))


# try:
#     df_out=levy_train_functions.levy_train_01_accounts(7)
#     run.log("INFO 9A", "Test sql query exec'ed correctly")
#     run.log("INFO 9B", "NRows: {}".format(len(df_out)))
#     run.log("INFO 9C","Columns: {}".format(str(list(df_out.columns))))    
# except Exception as E:
#     run.log("EXCEPTION 5",f'Exception: {E}')
#ensure deletion of model file at end of job:
if(os.path.exists(modelpath)):
    os.remove(modelpath)

print("*****************************")
print("END OF JOB")
print("METRICS:")
print(run.get_metrics())
print("***************************")
print("PRESENTING LOG:")
#print(run.get_all_logs())


try:
    # Create model build data into dataframe
    # Create df with all accounts and early adopter flag
   
    run.log('Success 01','Accounts Success')
 
except Exception:
    run.log('EXCEPTION 18','ModelCreationFailed')
