import pickle
import os
import pandas as pd
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

query_levy_score_set = DataPath(datastore, """SELECT A1, A2, A3, CASE WHEN CAST(A2 AS DATE)<'2017-07-01' THEN 1 ELSE 0 END AS early_adopter FROM PDS_AI.PT_A where A1=1""")
tabular_levy_score_set = Dataset.Tabular.from_sql_query(query_levy_score_set, query_timeout=10)
levy_score_set = tabular_levy_score_set.to_pandas_dataframe()


# change to today
# months since apprenticeship account sign-up
levy_score_set["months_since_sign_up"] = (pd.Timestamp(2022,01,30) - pd.to_datetime(levy_score_set["A2"]))/ np.timedelta64(1, "M")
# make the months since sign-up discrete for analysis purposes
levy_score_set["months_since_sign_up2"] =levy_score_set["months_since_sign_up"].apply(np.floor)



# current cohort Part 1

query_levy_commitments_p1 = DataPath(datastore, """SELECT A3 \
, total_commitments \
, CASE \
WHEN levy_split=1 AND yearmon_created = '2021-12' THEN total_commitments * 11.78 \
WHEN levy_split=1 AND yearmon_created = '2021-11' THEN total_commitments * 5.81 \
WHEN levy_split=1 AND yearmon_created = '2021-10' THEN total_commitments * 3.85 \
WHEN levy_split=1 AND yearmon_created = '2021-9' THEN total_commitments * 2.89 \
WHEN levy_split=1 AND yearmon_created = '2021-8' THEN total_commitments * 2.36 \
WHEN levy_split=1 AND yearmon_created = '2021-7' THEN total_commitments * 1.98 \
WHEN levy_split=1 AND yearmon_created = '2021-6' THEN total_commitments * 1.71 \
WHEN levy_split=1 AND yearmon_created = '2021-5' THEN total_commitments * 1.51 \
WHEN levy_split=1 AND yearmon_created = '2021-4' THEN total_commitments * 1.34 \
WHEN levy_split=1 AND yearmon_created = '2021-3' THEN total_commitments * 1.21 \
WHEN levy_split=1 AND yearmon_created = '2021-2' THEN total_commitments * 1.09 \
ELSE total_commitments END AS adjusted_commitments \
, occupation_1 \
, occupation_2 \
, occupation_3 \
, occupation_7 \
, occupation_13 \
, occupation_14 \
, occupation_15 \
, occupation_17 \
, occupation_20 \
, occupation_22 \
, occupation_24 \
, occupation_null \
, prev_12m_new_commitments \
, prev_12m_new_levy_transfers \
, A7 as levy_sending_company \
FROM \
(SELECT A3, CONCAT(YEAR(A2),'-',month(A2)) as yearmon_created, A1 as levy_split, A2, A7 \
FROM PDS_AI.PT_A \
WHERE CAST(A2 AS DATE)<DATEADD(day,-365,CAST(GETDATE() AS Date)) AND A1=1 \
) A \
LEFT JOIN \
(SELECT B10, count(*) AS total_commitments \
FROM PDS_AI.PT_B \
WHERE cast(B2 as date) >= DATEADD(day,-365,CAST(GETDATE() AS Date)) \
GROUP BY B10 \
) B \
ON A.A3=B.B10 \
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
WHERE cast(B2 as date) >= DATEADD(day,-730,CAST(GETDATE() AS Date)) AND cast(B2 as date) < DATEADD(day,-365,CAST(GETDATE() AS Date)) \
GROUP BY B10 \
) C \
ON A.A3=C.B10 """)
tabular_levy_commitments_p1 = Dataset.Tabular.from_sql_query(query_levy_commitments_p1, query_timeout=10)
levy_commitments_p1 = tabular_levy_commitments_p1.to_pandas_dataframe()

# part 2

query_levy_commitments_p2 = DataPath(datastore, """SELECT A3 \
, commitments_ending_12m \
, current_live_commitments \
FROM \
(SELECT A3, CONCAT(YEAR(A2),'-',month(A2)) as yearmon_created, A1 as levy_split, A2, A7 \
FROM PDS_AI.PT_A \
WHERE CAST(A2 AS DATE)<DATEADD(day,-365,CAST(GETDATE() AS Date)) AND A1=1 \
) A \
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
(B20 IS NULL OR CAST(B20 AS DATE)>=CAST(GETDATE() AS Date)) AND \
(B16 IS NULL OR CAST(B16 AS DATE)>=CAST(GETDATE() AS Date)) \
GROUP BY B10 \
) E \
ON A.A3=E.B10""")
tabular_levy_commitments_p2 = Dataset.Tabular.from_sql_query(query_levy_commitments_p2, query_timeout=10)
levy_commitments_p2 = tabular_levy_commitments_p2.to_pandas_dataframe()

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
levy_score_set = pd.merge(levy_score_set, \
                  levy_tpr_aggregated, \
                  left_on='account_id', \
                  right_on='A3', \
                  how='left')

# Create dummy variables for company type
company_type=pd.get_dummies(levy_score_set['company_type'],prefix='comp_type')
levy_score_set = levy_score_set.merge(company_type, left_index=True, right_index=True)

# Create year account created variable
#levy_score_set['cohort'] = levy_score_set['account_created'].dt.year

##########################Change current year########################
# Alter tpr_scheme_start_year to years_since_tpr_signup
levy_score_set['years_since_tpr_signup']=2022-levy_score_set['scheme_start_year']

# Function for new company flag

def fn_new_company(row):
    if row['months_since_sign_up2']<=6 :
        val=1
    else:
        val=0
    return val

levy_score_set['new_company']=levy_score_set.apply(fn_new_company,axis=1)

# Only keep relevant variables and rename accordingly
########################add back in ########################
#scoring_cols_to_keep=['A1','A3','months_since_sign_up2','occupation_1', \
#                    'occupation_2','occupation_3','occupation_7','occupation_13','occupation_14','occupation_15', \
#                    'occupation_17','occupation_20','occupation_22','occupation_24','occupation_null','employees', \
#                    'years_since_tpr_signup','comp_type_C','comp_type_E','comp_type_F','comp_type_I','comp_type_L', \
#                    'comp_type_P','comp_type_S','comp_type_X','tpr_match','new_company','early_adopter', \
#                    'commitments_ending_12m','prev_12m_new_commitments','prev_12m_new_levy_transfers', \
#                    'levy_sending_company','current_live_commitments','company_status']
#scoring_set = scoring_set[scoring_cols_to_keep]
#scoring_set.columns = ['levy_non_levy','account_id','as_months_since_sign_up','occupation_1', \
#                     'occupation_2','occupation_3','occupation_7','occupation_13','occupation_14','occupation_15', \
#                     'occupation_17','occupation_20','occupation_22','occupation_24','occupation_null','employees', \
#                     'years_since_tpr_signup','comp_type_C','comp_type_E','comp_type_F','comp_type_I','comp_type_L', \
#                     'comp_type_P','comp_type_S','comp_type_X','tpr_match','new_company','early_adopter', \
#                     'commitments_ending_12m','prev_12m_new_commitments','prev_12m_new_levy_transfers', \
#                     'levy_sending_company','current_live_commitments','company_status']

scoring_cols_to_keep=['A1','A3','months_since_sign_up2','occupation_1', \
                    'occupation_2','occupation_3','occupation_7','occupation_13','occupation_14','occupation_15', \
                    'occupation_17','occupation_20','occupation_22','occupation_24','occupation_null','employees', \
                    'years_since_tpr_signup','comp_type_C','comp_type_I', \
                    'comp_type_X','tpr_match','new_company','early_adopter', \
                    'commitments_ending_12m','prev_12m_new_commitments','prev_12m_new_levy_transfers', \
                    'levy_sending_company','current_live_commitments','company_status']
scoring_set = scoring_set[scoring_cols_to_keep]
scoring_set.columns = ['levy_non_levy','account_id','as_months_since_sign_up','occupation_1', \
                     'occupation_2','occupation_3','occupation_7','occupation_13','occupation_14','occupation_15', \
                     'occupation_17','occupation_20','occupation_22','occupation_24','occupation_null','employees', \
                     'years_since_tpr_signup','comp_type_C','comp_type_I', \
                     'comp_type_X','tpr_match','new_company','early_adopter', \
                     'commitments_ending_12m','prev_12m_new_commitments','prev_12m_new_levy_transfers', \
                     'levy_sending_company','current_live_commitments','company_status']

# Take logs to standardise the scale
levy_score_set['log_adjusted_commitments'] = np.log2(levy_score_set['adjusted_commitments']+1)
levy_score_set['log_employees'] = np.log2(levy_score_set['employees']+1)

print(levy_score_set)

# Select model variables only

#X = levy_score_set[['levy_non_levy','account_id','as_months_since_sign_up','occupation_1', \
#                     'occupation_2','occupation_3','occupation_7','occupation_13','occupation_14','occupation_15', \
#                     'occupation_17','occupation_20','occupation_22','occupation_24','occupation_null','employees', \
#                     'years_since_tpr_signup','comp_type_C','comp_type_E','comp_type_F','comp_type_I','comp_type_L', \
#                     'comp_type_P','comp_type_S','comp_type_X','tpr_match','new_company','early_adopter', \
#                     'commitments_ending_12m','prev_12m_new_commitments','prev_12m_new_levy_transfers', \
#                     'levy_sending_company','current_live_commitments','company_status']]

############################# Add back in ################################


# Take logs to standardise the scale
levy_score_set['log_employees'] = np.log2(levy_score_set['employees']+1)

X = levy_score_set[['levy_non_levy','account_id','as_months_since_sign_up','occupation_1', \
                     'occupation_2','occupation_3','occupation_7','occupation_13','occupation_14','occupation_15', \
                     'occupation_17','occupation_20','occupation_22','occupation_24','occupation_null','employees', \
                     'years_since_tpr_signup','comp_type_C','comp_type_I', \
                     'comp_type_X','tpr_match','new_company','early_adopter', \
                     'commitments_ending_12m','prev_12m_new_commitments','prev_12m_new_levy_transfers', \
                     'levy_sending_company','current_live_commitments','company_status']]


# load registered model 
global loaded_model

model_path = Model.get_model_path('levy_model')
loaded_model = joblib.load(model_path)

#score dataframe using saved model onto the base
scored=loaded_model.predict_proba(X)
quality_df_scored['levy_model_prediction']=scored[:,1]

run = Run.get_context()
run.log('levy_model_score_log', 'levy_model_score_log')

#write out scored file to parquet
levy_df_scored.to_parquet('./outputs/levy_model_scored.parquet')
