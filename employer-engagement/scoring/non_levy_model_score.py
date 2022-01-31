import pickle
import os
import pandas as pd
import numpy as np
import pyarrow.parquet as pq
import joblib
from azureml.core.run import Run
from azureml.core import Dataset, Datastore, Model
from azureml.data.datapath import DataPath
from sklearn.linear_model import LogisticRegression

# Set up config of workspace and datastore
aml_workspace = Run.get_context().experiment.workspace
datastore = Datastore.get(aml_workspace, datastore_name='datamgmtdb')

#prevent SettingWithCopyWarning message from appearing
pd.options.mode.chained_assignment = None

# Create model build data into dataframe

# Create df with all accounts and early adopter flag

query_non_levy_score_set = DataPath(datastore, """SELECT A1, A2, A3, CASE WHEN CAST(A2 AS DATE)<cast('2017-07-01' as date) THEN 1 ELSE 0 END AS early_adopter FROM PDS_AI.PT_A WHERE A1=0""")
tabular_non_levy_score_set = Dataset.Tabular.from_sql_query(query_non_levy_score_set, query_timeout=10)
non_levy_score_set = tabular_non_levy_score_set.to_pandas_dataframe()


# change to today
# months since apprenticeship account sign-up
non_levy_score_set["months_since_sign_up"] = (pd.Timestamp(2022,01,30) - pd.to_datetime(non_levy_score_set["A2"]))/ np.timedelta64(1, "M")
# make the months since sign-up discrete for analysis purposes
non_levy_score_set["months_since_sign_up2"] =non_levy_score_set["months_since_sign_up"].apply(np.floor)



# 2018/2019 cohort Part 1

query_non_levy_commitments = DataPath(datastore, """SELECT A.A3 \
, C.occupation_1 \
, C.occupation_2 \
, C.occupation_3 \
, C.occupation_7 \
, C.occupation_13 \
, C.occupation_14 \
, C.occupation_15 \
, C.occupation_17 \
, C.occupation_20 \
, C.occupation_22 \
, C.occupation_24 \
, C.occupation_null \
, D.commitments_ending_12m \
, C.prev_12m_new_commitments \
, C.prev_12m_new_levy_transfers \
, A.A7 as levy_sending_company \
, E.current_live_commitments \
FROM \
(SELECT A3, CONCAT(YEAR(A2),'-',month(A2)) as yearmon_created, A1 as levy_split, A2, A7 \
FROM PDS_AI.PT_A \
WHERE A1=0 \
) A \
LEFT JOIN  \
LEFT JOIN \
(SELECT B10 \
, COUNT(*) AS prev_12m_new_commitments \
, SUM(CASE WHEN B12=1 THEN 1 ELSE 0 END) AS prev_12m_new_levy_transfers \
, CAST(SUM(CASE WHEN B6 = '1' THEN 1.000 ELSE 0 END) / COUNT(*) AS DECIMAL(10,3)) AS occupation_1 \
, CAST(SUM(CASE WHEN B6 = '2' THEN 1.000 ELSE 0 END) / COUNT(*) AS DECIMAL(10,3)) AS occupation_2 \
, CAST(SUM(CASE WHEN B6 = '3' THEN 1.000 ELSE 0 END) / COUNT(*) AS DECIMAL(10,3)) AS occupation_3 \
, CAST(SUM(CASE WHEN B6 = '7' THEN 1.000 ELSE 0 END) / COUNT(*) AS DECIMAL(10,3)) AS occupation_7 \
, CAST(SUM(CASE WHEN B6 = '13' THEN 1.000 ELSE 0 END) / COUNT(*) AS DECIMAL(10,3)) AS occupation_13 \
, CAST(SUM(CASE WHEN B6 = '14' THEN 1.000 ELSE 0 END) / COUNT(*) AS DECIMAL(10,3)) AS occupation_14 \
, CAST(SUM(CASE WHEN B6 = '15' THEN 1.000 ELSE 0 END) / COUNT(*) AS DECIMAL(10,3)) AS occupation_15 \
, CAST(SUM(CASE WHEN B6 = '17' THEN 1.000 ELSE 0 END) / COUNT(*) AS DECIMAL(10,3)) AS occupation_17 \
, CAST(SUM(CASE WHEN B6 = '20' THEN 1.000 ELSE 0 END) / COUNT(*) AS DECIMAL(10,3)) AS occupation_20 \
, CAST(SUM(CASE WHEN B6 = '22' THEN 1.000 ELSE 0 END) / COUNT(*) AS DECIMAL(10,3)) AS occupation_22 \
, CAST(SUM(CASE WHEN B6 = '24' THEN 1.000 ELSE 0 END) / COUNT(*) AS DECIMAL(10,3)) AS occupation_24 \
, CAST(SUM(CASE WHEN B6 = NULL THEN 1.000 ELSE 0 END) / COUNT(*) AS DECIMAL(10,3)) AS occupation_null \
FROM PDS_AI.PT_B \
WHERE cast(B2 as date) < DATEADD(day,-365,CAST(GETDATE() AS Date)) AND CAST(B2 AS DATE)>=DATEADD(day,-730,CAST(GETDATE() AS Date)) \
GROUP BY B10 \
) C \
ON A.A3=C.B10 \
LEFT JOIN \
(SELECT B10 \
, COUNT(*) AS commitments_ending_12m \
FROM PDS_AI.PT_B \
WHERE cast(B17 as date) < CAST(GETDATE() AS Date) AND CAST(B17 AS DATE)>=DATEADD(day,-365,CAST(GETDATE() AS Date)) \
AND (CAST(B20 AS DATE) >= DATEADD(day,-365,CAST(GETDATE() AS Date)) OR B20 IS NULL) \
AND (CAST(B16 AS DATE) >= DATEADD(day,-365,CAST(GETDATE() AS Date)) OR B16 IS NULL) \
GROUP BY B10 \
) D \
ON A.A3=D.B10 \
LEFT JOIN \
(SELECT B10 \
, COUNT(*) AS current_live_commitments \
FROM PDS_AI.PT_B \
WHERE cast(B2 AS DATE) < DATEADD(day,-365,CAST(GETDATE() AS Date)) AND \
(B20 IS NULL OR CAST(B20 AS DATE)>=DATEADD(day,-365,CAST(GETDATE() AS Date))) AND \
(B16 IS NULL OR CAST(B16 AS DATE)>=DATEADD(day,-365,CAST(GETDATE() AS Date))) \
GROUP BY B10 \
) E \
ON A.A3=E.B10 """)
tabular_non_levy_commitments = Dataset.Tabular.from_sql_query(query_non_levy_commitments, query_timeout=10)
non_levy_commitments = tabular_non_levy_commitments.to_pandas_dataframe()

# add commitment data onto account selection
non_levy_score_set = pd.merge(non_levy_score_set, \ 
                  non_levy_commitments, \
                  left_on=['A3'], \
                  right_on=['A3'], \
                  how='left')

# Fill commitments with 0 if missing
non_levy_score_set = non_levy_score_set.fillna(0)

# TPR data
query_tpr_aggregated = DataPath(datastore, """SELECT A3 \
, CASE WHEN employees IS NULL THEN 0 ELSE 1 END AS tpr_match \
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
, COALESCE(company_type,'X') as company_type \
, company_status \
FROM ( \
SELECT A3 \
, A1 \
, MAX(D10) AS employees \
, MIN(YEAR(D6)) AS scheme_start_year \
, MAX(SUBSTRING(D12,1,1)) AS company_type \
, MAX(D8) as company_status \
FROM \
(SELECT A3 \
, A1 \
FROM PDS_AI.PT_A \
) A \
LEFT JOIN \
(SELECT D15, D10, D6, D12, D8 \
FROM PDS_AI.PT_D \
) B \
ON A.A3=B.D15 \
GROUP BY A3, A1 \
) c """)
tabular_tpr_aggregated = Dataset.Tabular.from_sql_query(query_tpr_aggregated, query_timeout=10)
levy_score_set_tpr_aggregated = tabular_tpr_aggregated.to_pandas_dataframe()


# Join TPR data to model set
non_levy_score_set = pd.merge(non_levy_score_set, \
                  levy_tpr_aggregated, \
                  left_on='account_id', \
                  right_on='A3', \
                  how='left')

# Create dummy variables for company type
company_type=pd.get_dummies(non_levy_score_set['company_type'],prefix='comp_type')
non_levy_score_set = non_levy_score_set.merge(company_type, left_index=True, right_index=True)

# Create year account created variable
#levy_score_set['cohort'] = levy_score_set['account_created'].dt.year

##########################Change current year########################
# Alter tpr_scheme_start_year to years_since_tpr_signup
non_levy_score_set['years_since_tpr_signup']=2022-non_levy_score_set['scheme_start_year']

# Function for new company flag

def fn_new_company(row):
    if row['months_since_sign_up2']<=6 :
        val=1
    else:
        val=0
    return val

non_levy_score_set['new_company']=non_levy_score_set.apply(fn_new_company,axis=1)

# Only keep relevant variables and rename accordingly
model_cols_to_keep=['A1','A3','months_since_sign_up2','occupation_1', \
                    'occupation_2','occupation_3','occupation_7','occupation_13','occupation_14','occupation_15', \
                    'occupation_17','occupation_20','occupation_22','occupation_24','occupation_null','employees', \
                    'years_since_tpr_signup','comp_type_C','comp_type_E','comp_type_F','comp_type_I','comp_type_L', \
                    'comp_type_P','comp_type_S','comp_type_X','tpr_match','new_company','early_adopter', \
                    'commitments_ending_12m','prev_12m_new_commitments','prev_12m_new_levy_transfers', \
                    'levy_sending_company','current_live_commitments','company_status']
non_levy_score_set = non_levy_score_set[model_cols_to_keep]
non_levy_score_set.columns = ['levy_non_levy','account_id','as_months_since_sign_up','occupation_1', \
                     'occupation_2','occupation_3','occupation_7','occupation_13','occupation_14','occupation_15', \
                     'occupation_17','occupation_20','occupation_22','occupation_24','occupation_null','employees', \
                     'years_since_tpr_signup','comp_type_C','comp_type_E','comp_type_F','comp_type_I','comp_type_L', \
                     'comp_type_P','comp_type_S','comp_type_X','tpr_match','new_company','early_adopter', \
                     'commitments_ending_12m','prev_12m_new_commitments','prev_12m_new_levy_transfers', \
                     'levy_sending_company','current_live_commitments','company_status']

# Take logs to standardise the scale
non_levy_score_set['log_adjusted_commitments'] = np.log2(non_levy_score_set['adjusted_commitments']+1)
non_levy_score_set['log_employees'] = np.log2(non_levy_score_set['employees']+1)


print(non_levy_score_set)


# Select model variables only

#X = non_levy_score_set[['levy_non_levy','as_months_since_sign_up','occupation_1', \
#                     'occupation_2','occupation_3','occupation_7','occupation_13','occupation_14','occupation_15', \
#                     'occupation_17','occupation_20','occupation_22','occupation_24','occupation_null','employees', \
#                     'years_since_tpr_signup','comp_type_C','comp_type_E','comp_type_F','comp_type_I','comp_type_L', \
#                     'comp_type_P','comp_type_S','comp_type_X','tpr_match','new_company','early_adopter', \
#                     'commitments_ending_12m','prev_12m_new_commitments','prev_12m_new_levy_transfers', \
#                     'levy_sending_company','current_live_commitments','company_status']]

############################# Add back in ################################


# Take logs to standardise the scale
non_levy_score_set['log_employees'] = np.log2(non_levy_score_set['employees']+1)

X = non_levy_score_set[['levy_non_levy','as_months_since_sign_up','occupation_1', \
                     'occupation_2','occupation_3','occupation_7','occupation_13','occupation_14','occupation_15', \
                     'occupation_17','occupation_20','occupation_22','occupation_24','occupation_null','employees', \
                     'years_since_tpr_signup','comp_type_C','comp_type_I', \
                     'comp_type_X','tpr_match','new_company','early_adopter', \
                     'commitments_ending_12m','prev_12m_new_commitments','prev_12m_new_levy_transfers', \
                     'levy_sending_company','current_live_commitments','company_status']]


# load registered model 
global loaded_model

model_path = Model.get_model_path('non_levy_model')
loaded_model = joblib.load(model_path)

#score dataframe using saved model onto the base
scored=loaded_model.predict_proba(X)
quality_df_scored['non_levy_model_prediction']=scored[:,1]

run = Run.get_context()
run.log('non_levy_model_score_log', 'non_levy_model_score_log')

#write out scored file to parquet
non_levy_df_scored.to_parquet('./outputs/non_levy_model_scored.parquet')
