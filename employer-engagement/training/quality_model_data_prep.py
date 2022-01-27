import pickle
import os
import pandas as pd
import numpy as np
import pyarrow.parquet as pq
import joblib
import datetime
import shap
import xgboost as xgb
from azureml.core.run import Run
from azureml.core import Dataset, Datastore, Model
from azureml.data.datapath import DataPath
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold


# Set up config of workspace and datastore

aml_workspace = Run.get_context().experiment.workspace
datastore = Datastore.get(aml_workspace, datastore_name='datamgmtdb')

#prevent SettingWithCopyWarning message from appearing
pd.options.mode.chained_assignment = None

# Create model build data into dataframe


# Create commitments plus proportion in each occupation in SQL for each cohort
# Need to split into 6 monthly time periods due to run problems

# 2018 H1

query_2018_H1 = DataPath(datastore, """SELECT C.A3 AS account_id \
, C.levy_split \
, C.account_created \
, C.completed_commitment \
, C.apprenticeship_id \
, C.commitment_date \
, SUM(CASE WHEN DATEDIFF(day,D.B2, C.commitment_date)<=365 and DATEDIFF(day,D.B2, C.commitment_date)>0 THEN 1 ELSE 0 END) AS previous_12mon_commitments \
FROM (SELECT A.A3 \
, A.levy_split \
, A.A2 AS account_created \
, B.B15 AS completed_commitment \
, B.B1 as apprenticeship_id \
, B.commitment_date \
FROM \
(SELECT A3, A1 as levy_split, A2 FROM PDS_AI.PT_A WHERE A2<cast('2018-07-01' as date)) A \
INNER JOIN \
(SELECT B10, B15, B3, CAST(B2 AS DATE) AS commitment_date, B1 \
FROM PDS_AI.PT_B \
WHERE CAST(B2 AS DATE) >= cast('2018-01-01' as date) AND CAST(B2 AS DATE) < cast('2018-07-01' as date) \
AND B3 IN (2,3,4,5) \
AND (B15=1 OR B16 IS NOT NULL OR B19=1) \
) B \
ON A.A3=B.B10) C \
LEFT JOIN \
(SELECT B10, CAST(B2 AS DATE) as B2 \
FROM PDS_AI.PT_B \
WHERE CAST(B2 AS DATE) < cast('2018-07-01' as date) \
AND B3 IN (2,3,4,5) \
) D \
ON C.A3=D.B10 \
GROUP BY C.A3 \
, C.levy_split \
, C.account_created \
, C.completed_commitment \
, C.apprenticeship_id \
, C.commitment_date \
""") 
tabular_2018_H1 = Dataset.Tabular.from_sql_query(query_2018_H1, query_timeout=1000)
quality_model_set_2018_H1 = tabular_2018_H1.to_pandas_dataframe()




# 2018 H2

query_2018_H2 = DataPath(datastore, """SELECT C.A3 AS account_id \
, C.levy_split \
, C.account_created \
, C.completed_commitment \
, C.apprenticeship_id \
, C.commitment_date \
, SUM(CASE WHEN DATEDIFF(day,D.B2, C.commitment_date)<=365 and DATEDIFF(day,D.B2, C.commitment_date)>0 THEN 1 ELSE 0 END) AS previous_12mon_commitments \
FROM (SELECT A.A3 \
, A.levy_split \
, A.A2 AS account_created \
, B.B15 AS completed_commitment \
, B.B1 as apprenticeship_id \
, B.commitment_date \
FROM \
(SELECT A3, A1 as levy_split, A2 FROM PDS_AI.PT_A WHERE A2<cast('2019-01-01' as date)) A \
INNER JOIN \
(SELECT B10, B15, B3, CAST(B2 AS DATE) AS commitment_date, B1 \
FROM PDS_AI.PT_B \
WHERE CAST(B2 AS DATE) >= cast('2018-07-01' as date) AND CAST(B2 AS DATE) < cast('2019-01-01' as date) \
AND B3 IN (2,3,4,5) \
AND (B15=1 OR B16 IS NOT NULL OR B19=1) \
) B \
ON A.A3=B.B10) C \
LEFT JOIN \
(SELECT B10, CAST(B2 AS DATE) as B2 \
FROM PDS_AI.PT_B \
WHERE CAST(B2 AS DATE) < cast('2019-01-01' as date) AND CAST(B2 AS DATE) >= cast('2017-07-01' as date) \
AND B3 IN (2,3,4,5) \
) D \
ON C.A3=D.B10 \
GROUP BY C.A3 \
, C.levy_split \
, C.account_created \
, C.completed_commitment \
, C.apprenticeship_id \
, C.commitment_date """)
tabular_2018_H2 = Dataset.Tabular.from_sql_query(query_2018_H2, query_timeout=1000)
quality_model_set_2018_H2 = tabular_2018_H2.to_pandas_dataframe()




#print(quality_model_set_2018_H2)





# 2019 H1

query_2019_H1 = DataPath(datastore, """SELECT C.A3 AS account_id \
, C.levy_split \
, C.account_created \
, C.completed_commitment \
, C.apprenticeship_id \
, C.commitment_date \
, SUM(CASE WHEN DATEDIFF(day,D.B2, C.commitment_date)<=365 and DATEDIFF(day,D.B2, C.commitment_date)>0 THEN 1 ELSE 0 END) AS previous_12mon_commitments \
FROM (SELECT A.A3 \
, A.levy_split \
, A.A2 AS account_created \
, B.B15 AS completed_commitment \
, B.B1 as apprenticeship_id \
, B.commitment_date \
FROM \
(SELECT A3, A1 as levy_split, A2 FROM PDS_AI.PT_A WHERE A2<cast('2019-07-01' as date)) A \
INNER JOIN \
(SELECT B10, B15, B3, CAST(B2 AS DATE) AS commitment_date, B1 \
FROM PDS_AI.PT_B \
WHERE CAST(B2 AS DATE) >= cast('2019-01-01' as date) AND CAST(B2 AS DATE) < cast('2019-07-01' as date) \
AND B3 IN (2,3,4,5) \
AND (B15=1 OR B16 IS NOT NULL OR B19=1) \
) B \
ON A.A3=B.B10) C \
LEFT JOIN \
(SELECT B10, CAST(B2 AS DATE) as B2 \
FROM PDS_AI.PT_B \
WHERE CAST(B2 AS DATE) < cast('2019-07-01' as date) AND CAST(B2 AS DATE) >= cast('2018-01-01' as date) \
AND B3 IN (2,3,4,5) \
) D \
ON C.A3=D.B10 \
GROUP BY C.A3 \
, C.levy_split \
, C.account_created \
, C.completed_commitment \
, C.apprenticeship_id \
, C.commitment_date """)
tabular_2019_H1 = Dataset.Tabular.from_sql_query(query_2019_H1, query_timeout=10)
quality_model_set_2019_H1 = tabular_2019_H1.to_pandas_dataframe()





#print(quality_model_set_2019_H1)


############################# Add back in ################################

# 2019 H2

#query_2019_H2 = DataPath(datastore, """SELECT C.A3 AS account_id \
#, C.levy_split \
#, C.account_created \
#, C.completed_commitment \
#, C.apprenticeship_id \
#, C.commitment_date \
#, SUM(CASE WHEN DATEDIFF(day,D.B2, C.commitment_date)<=365 and DATEDIFF(day,D.B2, C.commitment_date)>0 THEN 1 ELSE 0 END) AS previous_12mon_commitments \
#FROM (SELECT A.A3 \
#, A.levy_split \
#, A.A2 AS account_created \
#, B.B15 AS completed_commitment \
#, B.B1 as apprenticeship_id \
#, B.commitment_date \
#FROM \
#(SELECT A3, A1 as levy_split, A2 FROM PDS_AI.PT_A WHERE A2<cast('2019-09-01' as date)) A \
#INNER JOIN \
#(SELECT B10, B15, B3, CAST(B2 AS DATE) AS commitment_date, B1 \
#FROM PDS_AI.PT_B \
#WHERE CAST(B2 AS DATE) >= cast('2019-07-01' as date) AND CAST(B2 AS DATE) < cast('2019-09-01' as date) \
#AND B3 IN (2,3,4,5) \
#AND (B15=1 OR B16 IS NOT NULL OR B19=1) \
#) B \
#ON A.A3=B.B10) C \
#LEFT JOIN \
#(SELECT B10, CAST(B2 AS DATE) as B2 \
#FROM PDS_AI.PT_B \
#WHERE CAST(B2 AS DATE) < cast('2019-09-01' as date) AND CAST(B2 AS DATE) >= cast('2018-07-01' as date) \
#AND B3 IN (2,3,4,5) \
#) D \
#ON C.A3=D.B10 \
#GROUP BY C.A3 \
#, C.levy_split \
#, C.account_created \
#, C.completed_commitment \
#, C.apprenticeship_id \
#, C.commitment_date """)
#tabular_2019_H2 = Dataset.Tabular.from_sql_query(query_2019_H2, query_timeout=10)
#quality_model_set_2019_H2 = tabular_2019_H2.to_pandas_dataframe()




#print(quality_model_set_2019_H2)






# Create commitments plus proportion in each occupation in SQL for each cohort

query_commitment_info_all = DataPath(datastore, """SELECT \
A.B1 as apprenticeship_id \
, A.B3 AS apprenticeship_level \
, A.B13 AS apprentice_age \
, A.B12 AS funded_by_levy_transfer \
, A.B11 AS provider_id \
, CASE WHEN A.B6 = '1' THEN 1 ELSE 0 END AS occupation_1 \
, CASE WHEN A.B6 = '2' THEN 1 ELSE 0 END AS occupation_2 \
, CASE WHEN A.B6 = '3' THEN 1 ELSE 0 END AS occupation_3 \
, CASE WHEN A.B6 = '7' THEN 1 ELSE 0 END AS occupation_7 \
, CASE WHEN A.B6 = '13' THEN 1 ELSE 0 END AS occupation_13 \
, CASE WHEN A.B6 = '14' THEN 1 ELSE 0 END AS occupation_14 \
, CASE WHEN A.B6 = '15' THEN 1 ELSE 0 END AS occupation_15 \
, CASE WHEN A.B6 = '17' THEN 1 ELSE 0 END AS occupation_17 \
, CASE WHEN A.B6 = '20' THEN 1 ELSE 0 END AS occupation_20 \
, CASE WHEN A.B6 = '22' THEN 1 ELSE 0 END AS occupation_22 \
, CASE WHEN A.B6 = '24' THEN 1 ELSE 0 END AS occupation_24 \
FROM PDS_AI.PT_B A \
WHERE CAST(B2 AS DATE) >= cast('2018-01-01' as date) AND CAST(B2 AS DATE) < cast('2020-01-01' as date) \
AND B3 IN (2,3,4,5) \
AND (B15=1 OR B16 IS NOT NULL) """)
tabular_commitment_info_all = Dataset.Tabular.from_sql_query(query_commitment_info_all, query_timeout=10)
quality_commitment_info_all = tabular_commitment_info_all.to_pandas_dataframe()




#print(quality_commitment_info_all)






# Union the four unique date ranges together

############################# Add back in ################################

#quality_model_set_all=pd.concat([quality_model_set_2019_H2,quality_model_set_2019_H1,quality_model_set_2018_H2,quality_model_set_2018_H1])
quality_model_set_all=pd.concat([quality_model_set_2019_H1,quality_model_set_2018_H2,quality_model_set_2018_H1])



#print(quality_model_set_all)




# Add on the commitment info leading up to the commitment in question

quality_model_set = pd.merge(quality_model_set_all, \
                  quality_commitment_info_all, \
                  left_on='apprenticeship_id', \
                  right_on='apprenticeship_id', \
                  how='left')



#print(quality_model_set)




# months since apprenticeship account sign-up
quality_model_set["months_since_sign_up"] = (pd.Timestamp(2019,9,1) - pd.to_datetime(quality_model_set["account_created"]))/ np.timedelta64(1, "M")

# make the months since sign-up discrete for analysis purposes
quality_model_set["months_since_sign_up2"] = quality_model_set["months_since_sign_up"].apply(np.floor)

# Pull data from TPR set

query_tpr_aggregated = DataPath(datastore, """SELECT A3 \
, CASE WHEN employees IS NULL THEN 0 ELSE 1 END AS tpr_match \
, CASE WHEN employees IS NOT NULL THEN employees \
WHEN A1=0 and company_type='C' THEN 16 \
WHEN A1=0 and company_type='F' THEN 25 \
WHEN A1=0 and company_type='I' THEN 8 \
WHEN A1=0 and company_type='L' THEN 34 \
WHEN A1=0 and company_type='P' THEN 28.5 \
WHEN A1=0 and company_type='S' THEN 33 \
WHEN A1=1 and company_type='C' THEN 189 \
WHEN A1=1 and company_type='E' THEN 259.5 \
WHEN A1=1 and company_type='F' THEN 156 \
WHEN A1=1 and company_type='I' THEN 153 \
WHEN A1=1 and company_type='L' THEN 261 \
WHEN A1=1 and company_type='P' THEN 683 \
WHEN A1=1 and company_type='S' THEN 579 \
WHEN A1=1 THEN 200 \
WHEN A1=0 THEN 16 \
ELSE NULL END AS employees \
, CASE WHEN scheme_start_year IS NOT NULL THEN scheme_start_year \
WHEN A1=0 and company_type='P' THEN 2008 \
WHEN A1=1 and company_type='C' THEN 2004 \
WHEN A1=1 and company_type='F' THEN 2009 \
WHEN A1=1 and company_type='E' THEN 2008 \
WHEN A1=0 and company_type='I' THEN 2012 \
WHEN A1=0 and company_type='C' THEN 2012 \
WHEN A1=1 and company_type='P' THEN 1999 \
WHEN A1=0 and company_type='L' THEN 2007 \
WHEN A1=1 and company_type='S' THEN 2003 \
WHEN A1=1 and company_type='I' THEN 2001 \
WHEN A1=0 and company_type='F' THEN 2008 \
WHEN A1=1 and company_type='L' THEN 2010 \
WHEN A1=0 THEN 2011 \
WHEN A1=1 THEN 2004 \
ELSE NULL END AS scheme_start_year \
, COALESCE(company_type,'X') as company_type \
, company_status \
FROM ( \
SELECT A3 \
, A1 \
, MAX(D10) AS employees \
, MIN(YEAR(D6)) AS scheme_start_year \
, MAX(SUBSTRING(D12,1,1)) AS company_type \
, MAX(D8) AS company_status \
FROM \
(SELECT A3 \
, A1 \
FROM PDS_AI.PT_A \
WHERE A2<cast('2019-09-01' as date) \
) A \
LEFT JOIN \
(SELECT D15, D10, D6, D12, D8 \
FROM PDS_AI.PT_D \
) B \
ON A.A3=B.D15 \
GROUP BY A3, A1 \
) c """)
tabular_tpr_aggregated = Dataset.Tabular.from_sql_query(query_tpr_aggregated, query_timeout=10)
quality_tpr_aggregated = tabular_tpr_aggregated.to_pandas_dataframe()

# Join TPR data to model set
quality_model_set = pd.merge(quality_model_set, \
                  quality_tpr_aggregated, \
                  left_on='account_id', \
                  right_on='A3', \
                  how='left')

# Create dummy variables for company type
company_type=pd.get_dummies(quality_model_set['company_type'],prefix='comp_type')
quality_model_set = quality_model_set.merge(company_type, left_index=True, right_index=True)

# Create year account created variable
quality_model_set['cohort'] = quality_model_set['account_created'].dt.year

# Alter tpr_scheme_start_year to years_since_tpr_signup
quality_model_set['years_since_tpr_signup']=quality_model_set['cohort'].astype(int)-quality_model_set['scheme_start_year']

# Function for new company flag

def fn_new_company(row):
    if row['months_since_sign_up2']<=6 :
        val=1
    else:
        val=0
    return val

quality_model_set['new_company']=quality_model_set.apply(fn_new_company,axis=1)

# Function for early adopter flag

def fn_early_adopter(row):
    if row['account_created']<=datetime.datetime(2017,7,1) :
        val=1
    else:
        val=0
    return val
quality_model_set['early_adopter']=quality_model_set.apply(fn_early_adopter,axis=1)




#print(quality_model_set)






# Sample data for modelling

############################# Add back in ################################

# quality_sample_data=quality_model_set.sample(50000)
quality_sample_data=quality_model_set.sample(19)

# Only keep relevant variables and rename accordingly

############################# Add back in ################################

#model_cols_to_keep=['A3','levy_split','completed_commitment','previous_12mon_commitments', \
#                    'apprenticeship_level','apprentice_age','funded_by_levy_transfer', \
#                    'occupation_1','occupation_2','occupation_3','occupation_7', \
#                    'occupation_13','occupation_14','occupation_15','occupation_17','occupation_20','occupation_22', \
#                    'occupation_24','months_since_sign_up2','employees','scheme_start_year','comp_type_C','comp_type_E', \
#                    'comp_type_F','comp_type_I','comp_type_L','comp_type_P','comp_type_S','comp_type_X','tpr_match', \
#                    'new_company','early_adopter','years_since_tpr_signup','company_status','commitment_date']
#quality_sample_data = quality_sample_data[model_cols_to_keep]
#quality_sample_data.columns = ['account_id','levy_non_levy','completed_commitment','previous_12mon_commitments', \
#                     'apprenticeship_level','apprentice_age','funded_by_levy_transfer','occupation_1','occupation_2', \
#                     'occupation_3','occupation_7','occupation_13','occupation_14','occupation_15','occupation_17', \
#                     'occupation_20','occupation_22','occupation_24','as_months_since_sign_up','employees', \
#                     'tpr_scheme_start_year','comp_type_C','comp_type_E','comp_type_F','comp_type_I','comp_type_L', \
#                     'comp_type_P','comp_type_S','comp_type_X','tpr_match','new_company','early_adopter', \
#                     'years_since_tpr_signup','company_status','commitment_date']


model_cols_to_keep=['A3','levy_split','completed_commitment','previous_12mon_commitments', \
                    'apprenticeship_level','apprentice_age','funded_by_levy_transfer', \
                    'occupation_1','occupation_2','occupation_3','occupation_7', \
                    'occupation_13','occupation_14','occupation_15','occupation_17','occupation_20','occupation_22', \
                    'occupation_24','months_since_sign_up2','employees','scheme_start_year','comp_type_C', \
                    'comp_type_I','comp_type_X','tpr_match', \
                    'new_company','early_adopter','years_since_tpr_signup','company_status','commitment_date']
quality_sample_data = quality_sample_data[model_cols_to_keep]
quality_sample_data.columns = ['account_id','levy_non_levy','completed_commitment','previous_12mon_commitments', \
                     'apprenticeship_level','apprentice_age','funded_by_levy_transfer','occupation_1','occupation_2', \
                     'occupation_3','occupation_7','occupation_13','occupation_14','occupation_15','occupation_17', \
                     'occupation_20','occupation_22','occupation_24','as_months_since_sign_up','employees', \
                     'tpr_scheme_start_year','comp_type_C','comp_type_I', \
                     'comp_type_X','tpr_match','new_company','early_adopter', \
                     'years_since_tpr_signup','company_status','commitment_date']

# Take logs to standardise the scale
quality_sample_data['log_employees'] = np.log2(quality_sample_data['employees']+1)

# Remove outliers and non matched tpr data and tpr closed companies
quality_sample_data2 = quality_sample_data[(quality_sample_data.employees <=20000) & (quality_sample_data.tpr_match ==1) & (quality_sample_data.company_status==3)]
#quality_sample_data2.to_parquet('C:/Users/rober/Documents/Atos/data/completion_model_sample_data_subset.parquet')

#datastore = Datastore.get(aml_workspace, 'trainingdata')
#dataset = Dataset.Tabular.register_pandas_dataframe(quality_sample_data2, datastore, "quality_train_model_sample", show_progress=True)


print(quality_sample_data2)

# split the data into target and predictors
y = quality_sample_data2['completed_commitment']
#X = quality_sample_data2[['levy_non_levy','previous_12mon_commitments','apprenticeship_level','apprentice_age','funded_by_levy_transfer',
#                'occupation_1','occupation_2','occupation_3','occupation_7','occupation_13','occupation_14','occupation_15',
#                'occupation_17','occupation_20','occupation_22','occupation_24','as_months_since_sign_up','log_employees',
#                'comp_type_C','comp_type_E','comp_type_F','comp_type_I','comp_type_L','comp_type_P','comp_type_S',
#                'comp_type_X','tpr_match','new_company','early_adopter','years_since_tpr_signup']]

############################# Add back in ################################

X = quality_sample_data2[['levy_non_levy','previous_12mon_commitments','apprenticeship_level','apprentice_age','funded_by_levy_transfer', \
                'occupation_1','occupation_2','occupation_3','occupation_7','occupation_13','occupation_14','occupation_15', \
                'occupation_17','occupation_20','occupation_22','occupation_24','as_months_since_sign_up','log_employees', \
                'comp_type_C','comp_type_I', \
                'comp_type_X','tpr_match','new_company','early_adopter','years_since_tpr_signup']]


# get code from Vemal for train, accuracy and add below and register

# need to group final model score up to account level

# Create train and test sets

############################# Add back in ################################
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.5, random_state=99)

xgb_model = xgb.XGBClassifier()
xgb_model.fit(X_train, y_train)

explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X_train)

shap.summary_plot(shap_values, X_train)


run = Run.get_context()
run.log('quality_model_train_log')

# Save the trained model in the outputs folder
print("Saving model...")
os.makedirs('outputs', exist_ok=True)
model_file = os.path.join('outputs', 'quality_model.pkl')
joblib.dump(value=xgb_model, filename=model_file)

# Register the model to pick up in scoring
print('Registering model...')
Model.register(workspace=aml_workspace, \
               model_path = model_file, \
               model_name = 'quality_model')


run = Run.get_context()
run.log('quality_model_data_prep_log','quality_model_data_prep_log')
