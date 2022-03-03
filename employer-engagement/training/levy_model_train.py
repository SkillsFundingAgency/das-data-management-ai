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
from employer-engagement.training.sql.training.levy_train_sql_functions import levy_train_01_accounts
from employer-engagement.training.sql.training.levy_train_sql_functions import levy_train_02_levy_model_set_2018_2019_part1
from employer-engagement.training.sql.training.levy_train_sql_functions import levy_train_03_levy_model_set_2018_2019_part2
from employer-engagement.training.sql.training.levy_train_sql_functions import levy_train_04_levy_model_set_2019_2020_part1
from employer-engagement.training.sql.training.levy_train_sql_functions import levy_train_05_levy_model_set_2019_2020_part2
from employer-engagement.training.sql.training.levy_train_sql_functions import levy_train_06_levy_model_set_2022_part1
from employer-engagement.training.sql.training.levy_train_sql_functions import levy_train_07_levy_model_set_2022_part2
from employer-engagement.training.sql.generic.generic_sql_functions import generic_01_tpr
from employer-engagement.training.sql.generic.generic_sql_functions import generic_02_sic

# Set up config of workspace and datastore

aml_workspace = Run.get_context().experiment.workspace
datastore = Datastore.get(aml_workspace, datastore_name='datamgmtdb')

#prevent SettingWithCopyWarning message from appearing
pd.options.mode.chained_assignment = None

# Create model build data into dataframe via processing in SQL

# Create df with all accounts and early adopter flag
levy_model_accounts=levy_train_01_accounts(20)

print (levy_model_accounts)

account_list = levy_model_accounts['A3'].tolist()
print(account_list)
