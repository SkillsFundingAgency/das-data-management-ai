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

import levy_score_sql_functions as levy_score_functions
import generic_sql_functions as generic_train_functions

# Set up config of workspace and datastore
aml_workspace = Run.get_context().experiment.workspace
datastore = Datastore.get(aml_workspace, datastore_name='datamgmtdb')
run = Run.get_context()
run.log('levy_model_score','levy_model_score')
try:
    #prevent SettingWithCopyWarning message from appearing
    pd.options.mode.chained_assignment = None

    # Create df with all accounts and early adopter flag

    levy_score_set=levy_score_functions.levy_score_01_accounts(7)

    # select account_ids into list 
    account_list = levy_score_set['A3'].tolist()

    #Remove brackets from list
    sql_account_list=str(account_list)[1:-1]

    #get current date to use as today's date later
    getCurrentDateTime = datetime.datetime.now()

    # months since apprenticeship account sign-up
    levy_score_set["months_since_sign_up"] = (getCurrentDateTime - pd.to_datetime(levy_score_set["A2"]))/ np.timedelta64(1, "M")
    # make the months since sign-up discrete for analysis purposes
    levy_score_set["months_since_sign_up2"] =levy_score_set["months_since_sign_up"].apply(np.floor)

    # current cohort Part 1

    levy_commitments_p1=levy_score_functions.levy_score_02_levy_commitments_part1(sql_account_list)

    # part 2

    levy_commitments_p2=levy_score_functions.levy_score_03_levy_commitments_part2(sql_account_list)

    # join the two parts together
    levy_commitments = pd.merge(levy_commitments_p1, \
                      levy_commitments_p2, \
                      left_on=['A3'], \
                      right_on=['A3'], \
                      how='left')

    # add commitment data onto account selection
    levy_score_set = pd.merge(levy_score_set, \
                      levy_commitments, \
                      left_on=['A3'], \
                      right_on=['A3'], \
                      how='left')

    # Fill commitments with 0 if missing
    levy_score_set = levy_score_set.fillna(0)

    # TPR Data
    levy_tpr_aggregated=generic_train_functions.generic_01_tpr(sql_account_list)

    # Join TPR data to model set
    levy_score_set = pd.merge(levy_score_set, \
                      levy_tpr_aggregated, \
                      left_on='A3', \
                      right_on='A3', \
                      how='left')

    # Create dummy variables for company type
    company_type=pd.get_dummies(levy_score_set['company_type'],prefix='comp_type')
    levy_score_set = levy_score_set.merge(company_type, left_index=True, right_index=True)

    # Create year account created variable
    levy_score_set['years_since_tpr_signup']=datetime.datetime.today().year-levy_score_set['scheme_start_year']

    # Function for new company flag

    def fn_new_company(row):
        if row['months_since_sign_up2']<=6 :
            val=1
        else:
            val=0
        return val

    levy_score_set['new_company']=levy_score_set.apply(fn_new_company,axis=1)
    print (levy_score_set)
    run.log('Success 01','Data Creation')
except Exception:
    run.log('EXCEPTION 01','Data Creation')


try:
    # SIC Data
    sic_aggregated=generic_train_functions.generic_02_sic(sql_account_list)

    # Match SIC data onto accounts
    levy_model_set = pd.merge(levy_model_set, \
                      sic_aggregated, \
                      left_on='A3',\
                      right_on='d15',\
                      how='left')
                      
    # Create dummy variables for SIC code
    sic_code=pd.get_dummies(levy_model_set['new_sic_code'],prefix='sic_code')
    levy_model_set = levy_model_set.merge(sic_code, left_index=True, right_index=True)

    # Create dummy variables for SIC section
    sic_section=pd.get_dummies(levy_model_set['sic_section'],prefix='sic_section')
    levy_model_set = levy_model_set.merge(sic_section, left_index=True, right_index=True)

    # Create dummy variables for SIC division
    sic_division=pd.get_dummies(levy_model_set['sic_division'],prefix='sic_division')
    levy_model_set = levy_model_set.merge(sic_division, left_index=True, right_index=True)

    levy_model_set['log_employees'] = np.log2(levy_model_set['employees']+1)

    # Only keep relevant variables and rename accordingly
    print(levy_model_set)
    run.log('Success 02','SIC')
except Exception:
    run.log('EXCEPTION 02','SIC')


try:
    # load registered model 
    global loaded_model

    model_path = Model.get_model_path('levy_model')
    loaded_model = joblib.load(model_path)

    # understand order of columns that went into the model build
    cols_when_model_builds = loaded_model.get_booster().feature_names

    X=levy_score_set

    # Add columns into scoring df which don't exist in list - give value of 0
    for col in cols_when_model_builds:
        if col not in X.columns:
            X[col]=0
            
    # reorder columns in new scoring df
    X = X[cols_when_model_builds]

    #score dataframe using saved model onto the base
    scored=loaded_model.predict(X)

    levy_scored=levy_score_set

    levy_scored['levy_model_prediction']=np.exp2(scored)-1

    levy_scored2=levy_scored[['A3','levy_model_prediction']]
                            
    levy_scored2.rename(columns = {'A3':'account_id'}, inplace = True)

    print(levy_scored2)

    levy_scored2.to_csv("./outputs/levy_model_scored.csv")
    print(levy_scored2)
    run.log('Success 03','Model Scored')
except Exception:
    run.log('EXCEPTION 03','Model Scored')

