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
 
# Set up config of workspace and datastore
aml_workspace = Run.get_context().experiment.workspace
datastore = Datastore.get(aml_workspace, datastore_name='datamgmtdb')
run = Run.get_context()
run.log('levy_model_train','levy_model_train')

#prevent SettingWithCopyWarning message from appearing
pd.options.mode.chained_assignment = None

#try:
# Create model build data into dataframe
# Create df with all accounts and early adopter flag
levy_model_accounts=levy_train_functions.levy_train_01_accounts(2)
run.log('Success 01','Accounts Success')
#except Exception:
run.log('EXCEPTION 01','Accounts Exception')

#try:
# select account_ids into list 
account_list = levy_model_accounts['A3'].tolist()

#Remove brackets from list
sql_account_list=str(account_list)[1:-1]
print(sql_account_list)

# Select all accounts data for three time periods in model build

levy_model_accounts_2020 = levy_model_accounts[(levy_model_accounts.A2 <'2020-04-01')]
levy_model_accounts_2020['cohort']='2020'
# months since apprenticeship account sign-up
levy_model_accounts_2020["months_since_sign_up"] = (pd.Timestamp(2020,4,1) - pd.to_datetime(levy_model_accounts_2020["A2"]))/ np.timedelta64(1, "M")

levy_model_accounts_2019 = levy_model_accounts[(levy_model_accounts.A2 <'2019-04-01')]
levy_model_accounts_2019['cohort']='2019'
levy_model_accounts_2019["months_since_sign_up"] = (pd.Timestamp(2019,4,1) - pd.to_datetime(levy_model_accounts_2019["A2"]))/ np.timedelta64(1, "M")

levy_model_accounts_2022 = levy_model_accounts[(levy_model_accounts.A2 <'2022-01-01')]
levy_model_accounts_2022['cohort']='2022'
levy_model_accounts_2022["months_since_sign_up"] = (pd.Timestamp(2022,1,1) - pd.to_datetime(levy_model_accounts_2022["A2"]))/ np.timedelta64(1, "M")


# Add all sets of accounts data into one
levy_model_set=pd.concat([levy_model_accounts_2022,levy_model_accounts_2020,levy_model_accounts_2019])

# make the months since sign-up discrete for analysis purposes
levy_model_set["months_since_sign_up2"] =levy_model_set["months_since_sign_up"].apply(np.floor)
run.log('Success 01a','Accounts Success')
#except Exception:
run.log('EXCEPTION 01a','Accounts Exception')

#try:
# 2018/2019 cohort Part 1
levy_model_set_2018_2019_part1=levy_train_functions.levy_train_02_levy_model_set_2018_2019_part1(sql_account_list)
run.log('Success 02','Commitments 2018/19 Part 1 Success')
#except Exception:
run.log('EXCEPTION 02','Commitments 2018/19 Part 1 Exception')

#try:
# 2018/2019 cohort Part 2
levy_model_set_2018_2019_part2=levy_train_functions.levy_train_03_levy_model_set_2018_2019_part2(sql_account_list)
run.log('Success 03','Commitments 2018/19 Part 2 Success')
#except Exception:
run.log('EXCEPTION 03','Commitments 2018/19 Part 2 Exception')

#try:
# 2019/2020 cohort Part 1
levy_model_set_2019_2020_part1=levy_train_functions.levy_train_04_levy_model_set_2019_2020_part1(sql_account_list)
run.log('Success 04','Commitments 2019/20 Part 1 Success')
#except Exception:
run.log('EXCEPTION 04','Commitments 2019/20 Part 1 Exception')

#try:
# 2018/2019 cohort Part 2
levy_model_set_2019_2020_part2=levy_train_functions.levy_train_05_levy_model_set_2019_2020_part2(sql_account_list)
run.log('Success 05','Commitments 2019/20 Part 2 Success')
#except Exception:
run.log('EXCEPTION 05','Commitments 2019/20 Part 2 Exception')

#try:
# 2022 cohort Part 1
levy_model_set_2022_part1=levy_train_functions.levy_train_06_levy_model_set_2022_part1(sql_account_list)
run.log('Success 06','Commitments 2022 Part 1 Success')
#except Exception:
run.log('EXCEPTION 06','Commitments 2022 Part 1 Exception')

# 2022 cohort Part 2
levy_model_set_2022_part2=levy_train_functions.levy_train_07_levy_model_set_2022_part2(sql_account_list)
#run.log('Success 07','Commitments 2022 Part 2 Success')
#except Exception:
#run.log('EXCEPTION 07','Commitments 2022 Part 2 Exception')

#try:
# join both parts together for all time periods
universe_commitments_2019 = pd.merge(levy_model_set_2018_2019_part1, \
                  levy_model_set_2018_2019_part2, \
                  left_on=['A3'], \
                  right_on=['A3'], \
                  how='left')

universe_commitments_2020 = pd.merge(levy_model_set_2019_2020_part1, \
                  levy_model_set_2019_2020_part2, \
                  left_on=['A3'], \
                  right_on=['A3'], \
                  how='left')

universe_commitments_2022 = pd.merge(levy_model_set_2022_part1, \
                  levy_model_set_2022_part2, \
                  left_on=['A3'], \
                  right_on=['A3'], \
                  how='left')



# Add all sets of accounts data into one
universe_commitments_all=pd.concat([universe_commitments_2022,universe_commitments_2020,universe_commitments_2019])

# add commitment data to accounts
levy_model_set = pd.merge(levy_model_set, \
                  universe_commitments_all, \
                  left_on=['A3','cohort'], \
                  right_on=['A3','cohort'], \
                  how='left')

# Fill commitments with 0 if missing
levy_model_set = levy_model_set.fillna(0)
print(levy_model_set)
#run.log('Success 08','Account manipulation Success')
#except Exception:
#run.log('EXCEPTION 08','Account manipulation Exception')

#try:
# TPR Data
levy_tpr_aggregated=generic_train_functions.generic_01_tpr(sql_account_list)
#run.log('Success 09','TPR Success')
#except Exception:
#run.log('EXCEPTION 09','TPR Exception')

#try:
# Join TPR data to model set
levy_model_set = pd.merge(levy_model_set, \
                  levy_tpr_aggregated, \
                  left_on='A3', \
                  right_on='A3', \
                  how='left')

# Create dummy variables for company type
company_type=pd.get_dummies(levy_model_set['company_type'],prefix='comp_type')
levy_model_set = levy_model_set.merge(company_type, left_index=True, right_index=True)

print(levy_model_set)

# Create year account created variable
#levy_model_set['cohort'] = levy_model_set['account_created'].dt.year
        
# Alter tpr_scheme_start_year to years_since_tpr_signup
levy_model_set['years_since_tpr_signup']=levy_model_set['cohort'].astype(int)-levy_model_set['scheme_start_year']

# Function for new company flag

def fn_new_company(row):
    if row['months_since_sign_up2']<=6 :
        val=1
    else:
        val=0
    return val

levy_model_set['new_company']=levy_model_set.apply(fn_new_company,axis=1)

#run.log('Success 10','TPR manipulation Success')
#except Exception:
#run.log('EXCEPTION 10','TPR manipulation Exception')

try:
    # SIC Data
    sic_aggregated=generic_train_functions.generic_02_sic(sql_account_list)
    run.log('Success 11','SIC Success')
    #except Exception:
    #run.log('EXCEPTION 11','SIC Exception')

    #try:
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

    #Calculate the log(commitments)
    levy_model_set['log_adjusted_commitments'] = np.log2(levy_model_set['total_commitments'] + 1)

    levy_model_set2 = levy_model_set[(levy_model_set.employees <=20000) & (levy_model_set.tpr_match ==1) & (levy_model_set.company_status ==3)]

    levy_model_set3 = levy_model_set2
    run.log('Success 12','SIC manipulation Success')
    #except Exception:
    #run.log('EXCEPTION 12','SIC manipulation Exception')

    #try:
    levy_model_set3.rename(columns = {'A1':'levy_non_levy', 'A3':'account_id', 'months_since_sign_up2':'as_months_since_sign_up'}, inplace = True)

    #Drop any unnecessary rows
    levy_model_set3 = levy_model_set3.drop(['new_sic_code', 'company_status', 'tpr_match', 'company_type', \
    'employees', 'cohort', 'sic_division', 'sic_section', 'total_commitments','d15','A2','months_since_sign_up', \
    'scheme_start_year'], axis=1)

    #Drop rows with only a single unique value
    levy_model_set3 = levy_model_set3.drop([col for col in levy_model_set3 if len(levy_model_set3[col].unique()) == 1], axis=1)

    #Drop rows with any null values
    levy_model_set4 = levy_model_set3.dropna() 
    #run.log('Success 13','Model data prep Success')
except Exception:

    levy_model_set['log_employees'] = np.log2(levy_model_set['employees']+1)

    #Calculate the log(commitments)
    levy_model_set['log_adjusted_commitments'] = np.log2(levy_model_set['total_commitments'] + 1)

    levy_model_set2 = levy_model_set[(levy_model_set.employees <=20000) & (levy_model_set.tpr_match ==1)] 

    levy_model_set3 = levy_model_set2

    levy_model_set3.rename(columns = {'A1':'levy_non_levy', 'A3':'account_id', 'months_since_sign_up2':'as_months_since_sign_up'}, inplace = True)
    
    #Drop any unnecessary rows
    levy_model_set3 = levy_model_set3.drop(['company_status', 'tpr_match', 'company_type', \
    'employees', 'cohort', 'total_commitments','A2','months_since_sign_up', \
    'scheme_start_year'], axis=1)

    #Drop rows with only a single unique value
    levy_model_set3 = levy_model_set3.drop([col for col in levy_model_set3 if len(levy_model_set3[col].unique()) == 1], axis=1)

    #Drop rows with any null values
    levy_model_set4 = levy_model_set3.dropna() 

    print(levy_model_set4)
    run.log('EXCEPTION 13','Model data prep Exception')

    
#try:
#Split dataset into independent and dependent variables
y = levy_model_set4['log_adjusted_commitments']
X = levy_model_set4.drop(['log_adjusted_commitments'], axis=1)

#Split our dataset into train and test sets
X_train_acc, X_test_acc, y_train, y_test = train_test_split(X, y, test_size = 0.25)
X_train = X_train_acc.drop(['account_id'], axis=1)
X_test = X_test_acc.drop(['account_id'], axis=1)
print(X_train)
print(y_train)

#run.log('Success 14','Model train test Success')
#except Exception:
#run.log('EXCEPTION 14','Model train test Exception')

#try:
#Set parameters of xgboost regressor
xg_reg = xgb.XGBRegressor(objective ='reg:squarederror')
#, colsample_bytree = 0.3, learning_rate = 0.05,
#max_depth = 5, alpha = 10, n_estimators = 700, verbosity = 2, eval_metric='rmse')
#run.log('Success 15','Model build Success')
#except Exception:
#run.log('EXCEPTION 15','Model build Exception')

#try:
#Train our model using the training dataset
xg_reg.fit(X_train,y_train)

#Predict the target variable using the test dataset on our model
#preds = xg_reg.predict(X_test)

#R^2, Rreduced R^2 and RMSE
#reduced_r2 = 1-(1-r2_score(y_test, preds))*((len(X_test)-1)/(len(X_test)-X_test.shape[1]-1))
#r2='R^2 = {:.5f}'.format(r2_score(y_test, preds))
#reduced_r2_output='Reduced_R^2 = {:.5f}'.format(reduced_r2)
#rmse='RMSE = {:.5f}'.format(mean_squared_error(y_test, preds, squared=False))


#Shap Values
explainer = shap.TreeExplainer(xg_reg)
#shap_values = explainer.shap_values(X_train)
#plt.clf()
#shap.summary_plot(shap_values, X_train, feature_names=X.drop(['account_id'], axis=1).columns, plot_type="bar", show=True)
#run.log('Success 16','Model stats Success')
#run.log('R2',r2)
#run.log('Reduced R2',reduced_r2_output)
#run.log('RMSE',rmse)
#run.log_image('Shap Plot', plot=plt)
#except Exception:
#run.log('EXCEPTION 16','Model stats Exception')
# Logging Stats

#try:
# Save the trained model in the outputs folder
os.makedirs('outputs', exist_ok=True)
model_file = os.path.join('outputs', 'levy_model.pkl')
joblib.dump(value=xg_reg, filename=model_file)

# Register the model to pick up in scoring
Model.register(workspace=aml_workspace, \
               model_path = model_file, \
               model_name = 'levy_model')
#run.log('Success 17','Model save Success')
#except Exception:
#run.log('EXCEPTION 17','Model save Exception')

#try:
run.get_all_logs(destination='outputs')
run.log('Success 18','Output Logs')
#except Exception:
run.log('EXCEPTION 18','Output Logs')
