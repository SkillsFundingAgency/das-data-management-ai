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

# Create df with all accounts and early adopter flag

query_levy_accounts = DataPath(datastore, """SELECT A1, A2, A3, CASE WHEN CAST(A2 AS DATE)<'2017-07-01' THEN 1 ELSE 0 END AS early_adopter FROM PDS_AI.PT_A where A1=1""")
tabular_levy_accounts = Dataset.Tabular.from_sql_query(query_levy_accounts, query_timeout=10)
levy_model_accounts = tabular_levy_accounts.to_pandas_dataframe()

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

# 2018/2019 cohort Part 1

query_2018_2019_part1 = DataPath(datastore, """SELECT A3 \
, '2019' as cohort \
, total_commitments \
, CASE \
WHEN levy_split=1 AND yearmon_created = '2019-3' THEN total_commitments * 12.54 \
WHEN levy_split=1 AND yearmon_created = '2019-2' THEN total_commitments * 6.22 \
WHEN levy_split=1 AND yearmon_created = '2019-1' THEN total_commitments * 4.16 \
WHEN levy_split=1 AND yearmon_created = '2018-12' THEN total_commitments * 3.24 \
WHEN levy_split=1 AND yearmon_created = '2018-11' THEN total_commitments * 2.41 \
WHEN levy_split=1 AND yearmon_created = '2018-10' THEN total_commitments * 1.82 \
WHEN levy_split=1 AND yearmon_created = '2018-9' THEN total_commitments * 1.54 \
WHEN levy_split=1 AND yearmon_created = '2018-8' THEN total_commitments * 1.40 \
WHEN levy_split=1 AND yearmon_created = '2018-7' THEN total_commitments * 1.27 \
WHEN levy_split=1 AND yearmon_created = '2018-6' THEN total_commitments * 1.17 \
WHEN levy_split=1 AND yearmon_created = '2018-5' THEN total_commitments * 1.08 \
WHEN levy_split=0 AND yearmon_created = '2019-3' THEN total_commitments * 3.73 \
WHEN levy_split=0 AND yearmon_created = '2019-2' THEN total_commitments * 2.01 \
WHEN levy_split=0 AND yearmon_created = '2019-1' THEN total_commitments * 1.47 \
WHEN levy_split=0 AND yearmon_created = '2018-12' THEN total_commitments * 1.40 \
WHEN levy_split=0 AND yearmon_created = '2018-11' THEN total_commitments * 1.33 \
WHEN levy_split=0 AND yearmon_created = '2018-10' THEN total_commitments * 1.22 \
WHEN levy_split=0 AND yearmon_created = '2018-9' THEN total_commitments * 1.15 \
WHEN levy_split=0 AND yearmon_created = '2018-8' THEN total_commitments * 1.12 \
WHEN levy_split=0 AND yearmon_created = '2018-7' THEN total_commitments * 1.07 \
WHEN levy_split=0 AND yearmon_created = '2018-6' THEN total_commitments * 1.03 \
WHEN levy_split=0 AND yearmon_created = '2018-5' THEN total_commitments * 1.02 \
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
WHERE A2<'2019-04-01' and A1=1 \
) A \
LEFT JOIN \
(SELECT B10, count(*) AS total_commitments \
FROM PDS_AI.PT_B \
WHERE cast(B2 as date) >= '2018-04-01' AND cast(B2 as date) < '2019-04-01' \
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
WHERE cast(B2 as date) >= '2017-04-01' AND cast(B2 as date) < '2018-04-01' \
GROUP BY B10 \
) C \
ON A.A3=C.B10 """)
tabular_2018_2019_part1 = Dataset.Tabular.from_sql_query(query_2018_2019_part1, query_timeout=10)
levy_model_set_2018_2019_part1 = tabular_2018_2019_part1.to_pandas_dataframe()

# 2018/2019 cohort Part 2

query_2018_2019_part2 = DataPath(datastore, """SELECT A3 \
, commitments_ending_12m \
, current_live_commitments \
FROM \
(SELECT A3, CONCAT(YEAR(A2),'-',month(A2)) as yearmon_created, A1 as levy_split, A2, A7 \
FROM PDS_AI.PT_A \
WHERE A2<'2019-04-01' and A1=1 \
) A \
LEFT JOIN \
(SELECT B10 \
, COUNT(*) AS commitments_ending_12m \
FROM PDS_AI.PT_B \
WHERE cast(B17 as date) < '2019-04-01' AND CAST(B17 AS DATE)>='2018-04-01' \
AND (CAST(B20 AS DATE) >= '2018-04-01' OR B20 IS NULL) \
AND (CAST(B16 AS DATE) >= '2018-04-01' OR B16 IS NULL) \
GROUP BY B10 \
) D \
ON A.A3=D.B10 \
LEFT JOIN \
(SELECT B10 \
, COUNT(*) AS current_live_commitments \
FROM PDS_AI.PT_B \
WHERE cast(B2 AS DATE) < '2018-04-01' AND \
(B20 IS NULL OR CAST(B20 AS DATE)>='2019-04-01') AND \
(B16 IS NULL OR CAST(B16 AS DATE)>='2019-04-01') \
GROUP BY B10 \
) E \
ON A.A3=E.B10""")
tabular_2018_2019_part2 = Dataset.Tabular.from_sql_query(query_2018_2019_part2, query_timeout=10)
levy_model_set_2018_2019_part2 = tabular_2018_2019_part2.to_pandas_dataframe()



# 2019/2020 cohort Part 1

query_2019_2020_part1 = DataPath(datastore, """SELECT A3 \
, '2020' as cohort \
, total_commitments \
, CASE \
WHEN levy_split=1 AND yearmon_created = '2020-3' THEN total_commitments * 12.54 \
WHEN levy_split=1 AND yearmon_created = '2020-2' THEN total_commitments * 6.22 \
WHEN levy_split=1 AND yearmon_created = '2020-1' THEN total_commitments * 4.16 \
WHEN levy_split=1 AND yearmon_created = '2019-12' THEN total_commitments * 3.24 \
WHEN levy_split=1 AND yearmon_created = '2019-11' THEN total_commitments * 2.41 \
WHEN levy_split=1 AND yearmon_created = '2019-10' THEN total_commitments * 1.82 \
WHEN levy_split=1 AND yearmon_created = '2019-9' THEN total_commitments * 1.54 \
WHEN levy_split=1 AND yearmon_created = '2019-8' THEN total_commitments * 1.40 \
WHEN levy_split=1 AND yearmon_created = '2019-7' THEN total_commitments * 1.27 \
WHEN levy_split=1 AND yearmon_created = '2019-6' THEN total_commitments * 1.17 \
WHEN levy_split=1 AND yearmon_created = '2019-5' THEN total_commitments * 1.08 \
WHEN levy_split=0 AND yearmon_created = '2020-3' THEN total_commitments * 3.73 \
WHEN levy_split=0 AND yearmon_created = '2020-2' THEN total_commitments * 2.01 \
WHEN levy_split=0 AND yearmon_created = '2020-1' THEN total_commitments * 1.47 \
WHEN levy_split=0 AND yearmon_created = '2019-12' THEN total_commitments * 1.40 \
WHEN levy_split=0 AND yearmon_created = '2019-11' THEN total_commitments * 1.33 \
WHEN levy_split=0 AND yearmon_created = '2019-10' THEN total_commitments * 1.22 \
WHEN levy_split=0 AND yearmon_created = '2019-9' THEN total_commitments * 1.15 \
WHEN levy_split=0 AND yearmon_created = '2019-8' THEN total_commitments * 1.12 \
WHEN levy_split=0 AND yearmon_created = '2019-7' THEN total_commitments * 1.07 \
WHEN levy_split=0 AND yearmon_created = '2019-6' THEN total_commitments * 1.03 \
WHEN levy_split=0 AND yearmon_created = '2019-5' THEN total_commitments * 1.02 \
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
WHERE A2<'2020-04-01' AND A1=1 \
) A \
LEFT JOIN \
(SELECT B10, count(*) AS total_commitments \
FROM PDS_AI.PT_B \
WHERE cast(B2 as date) >= '2019-04-01' AND cast(B2 as date) < '2020-04-01' \
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
WHERE cast(B2 as date) >= '2018-04-01' AND cast(B2 as date) < '2019-04-01' \
GROUP BY B10 \
) C \
ON A.A3=C.B10""")
tabular_2019_2020_part1 = Dataset.Tabular.from_sql_query(query_2019_2020_part1, query_timeout=10)
levy_model_set_2019_2020_part1 = tabular_2019_2020_part1.to_pandas_dataframe()

# 2018/2019 cohort Part 2

query_2019_2020_part2 = DataPath(datastore, """SELECT A3 \
, commitments_ending_12m \
, current_live_commitments \
FROM \
(SELECT A3, CONCAT(YEAR(A2),'-',month(A2)) as yearmon_created, A1 as levy_split, A2, A7 \
FROM PDS_AI.PT_A \
WHERE A2<'2020-04-01' AND A1=1 \
) A \
LEFT JOIN \
(SELECT B10 \
, COUNT(*) AS commitments_ending_12m \
FROM PDS_AI.PT_B \
WHERE cast(B17 as date) < '2020-04-01' AND CAST(B17 AS DATE)>='2019-04-01' \
AND (CAST(B20 AS DATE) >= '2019-04-01' OR B20 IS NULL) \
AND (CAST(B16 AS DATE) >= '2019-04-01' OR B16 IS NULL) \
GROUP BY B10 \
) D \
ON A.A3=D.B10 \
LEFT JOIN \
(SELECT B10 \
, COUNT(*) AS current_live_commitments \
FROM PDS_AI.PT_B \
WHERE cast(B2 AS DATE) < '2019-04-01' AND \
(B20 IS NULL OR CAST(B20 AS DATE)>='2020-04-01') AND \
(B16 IS NULL OR CAST(B16 AS DATE)>='2020-04-01') \
GROUP BY B10 \
) E \
ON A.A3=E.B10""")
tabular_2019_2020_part2 = Dataset.Tabular.from_sql_query(query_2019_2020_part2, query_timeout=10)
levy_model_set_2019_2020_part2 = tabular_2019_2020_part2.to_pandas_dataframe()




# 2022 cohort Part 1

query_2022_part1 = DataPath(datastore, """SELECT A3 \
, '2022' as cohort \
, total_commitments \
, CASE \
WHEN levy_split=1 AND yearmon_created = '2021-12' THEN total_commitments * 12.54 \
WHEN levy_split=1 AND yearmon_created = '2021-11' THEN total_commitments * 6.22 \
WHEN levy_split=1 AND yearmon_created = '2021-10' THEN total_commitments * 4.16 \
WHEN levy_split=1 AND yearmon_created = '2021-9' THEN total_commitments * 3.24 \
WHEN levy_split=1 AND yearmon_created = '2021-8' THEN total_commitments * 2.41 \
WHEN levy_split=1 AND yearmon_created = '2021-7' THEN total_commitments * 1.82 \
WHEN levy_split=1 AND yearmon_created = '2021-6' THEN total_commitments * 1.54 \
WHEN levy_split=1 AND yearmon_created = '2021-5' THEN total_commitments * 1.40 \
WHEN levy_split=1 AND yearmon_created = '2021-4' THEN total_commitments * 1.27 \
WHEN levy_split=1 AND yearmon_created = '2021-3' THEN total_commitments * 1.17 \
WHEN levy_split=1 AND yearmon_created = '2021-2' THEN total_commitments * 1.08 \
WHEN levy_split=0 AND yearmon_created = '2021-12' THEN total_commitments * 3.73 \
WHEN levy_split=0 AND yearmon_created = '2021-11' THEN total_commitments * 2.01 \
WHEN levy_split=0 AND yearmon_created = '2021-10' THEN total_commitments * 1.47 \
WHEN levy_split=0 AND yearmon_created = '2021-9' THEN total_commitments * 1.40 \
WHEN levy_split=0 AND yearmon_created = '2021-8' THEN total_commitments * 1.33 \
WHEN levy_split=0 AND yearmon_created = '2021-7' THEN total_commitments * 1.22 \
WHEN levy_split=0 AND yearmon_created = '2021-6' THEN total_commitments * 1.15 \
WHEN levy_split=0 AND yearmon_created = '2021-5' THEN total_commitments * 1.12 \
WHEN levy_split=0 AND yearmon_created = '2021-4' THEN total_commitments * 1.07 \
WHEN levy_split=0 AND yearmon_created = '2021-3' THEN total_commitments * 1.03 \
WHEN levy_split=0 AND yearmon_created = '2021-2' THEN total_commitments * 1.02 \
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
WHERE A2<'2022-01-01' AND A1=1 \
) A \
LEFT JOIN \
(SELECT B10, count(*) AS total_commitments \
FROM PDS_AI.PT_B \
WHERE cast(B2 as date) >= '2021-01-01' AND cast(B2 as date) < '2022-01-01' \
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
WHERE cast(B2 as date) >= '2020-01-01' AND cast(B2 as date) < '2021-01-01' \
GROUP BY B10 \
) C \
ON A.A3=C.B10""")
tabular_2022_part1 = Dataset.Tabular.from_sql_query(query_2022_part1, query_timeout=10)
levy_model_set_2022_part1 = tabular_2022_part1.to_pandas_dataframe()

# 2022 cohort Part 2

query_2022_part2 = DataPath(datastore, """SELECT A3 \
, commitments_ending_12m \
, current_live_commitments \
FROM \
(SELECT A3, CONCAT(YEAR(A2),'-',month(A2)) as yearmon_created, A1 as levy_split, A2, A7 \
FROM PDS_AI.PT_A \
WHERE A2<'2022-01-01' AND A1=1 \
) A \
LEFT JOIN \
(SELECT B10 \
, COUNT(*) AS commitments_ending_12m \
FROM PDS_AI.PT_B \
WHERE cast(B17 as date) < '2022-01-01' AND CAST(B17 AS DATE)>='2021-01-01' \
AND (CAST(B20 AS DATE) >= '2021-01-01' OR B20 IS NULL) \
AND (CAST(B16 AS DATE) >= '2021-01-01' OR B16 IS NULL) \
GROUP BY B10 \
) D \
ON A.A3=D.B10 \
LEFT JOIN \
(SELECT B10 \
, COUNT(*) AS current_live_commitments \
FROM PDS_AI.PT_B \
WHERE cast(B2 AS DATE) < '2021-01-01' AND \
(B20 IS NULL OR CAST(B20 AS DATE)>='2022-01-01') AND \
(B16 IS NULL OR CAST(B16 AS DATE)>='2022-01-01') \
GROUP BY B10 \
) E \
ON A.A3=E.B10 """)
tabular_2022_part2 = Dataset.Tabular.from_sql_query(query_2022_part2, query_timeout=10)
levy_model_set_2022_part2 = tabular_2022_part2.to_pandas_dataframe()


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

# 2019 H1

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
WHERE A2<'2022-01-01' \
) A \
LEFT JOIN \
(SELECT D15, D10, D6, D12, D8 \
FROM PDS_AI.PT_D \
) B \
ON A.A3=B.D15 \
GROUP BY A3, A1 \
) c """)
tabular_tpr_aggregated = Dataset.Tabular.from_sql_query(query_tpr_aggregated, query_timeout=10)
levy_tpr_aggregated = tabular_tpr_aggregated.to_pandas_dataframe()

print("levy_tpr_aggregated")
print(levy_tpr_aggregated)
print(levy_tpr_aggregated.tpr_match.value_counts()
print(levy_tpr_aggregated.company_status.value_counts()


# Join TPR data to model set
levy_model_set = pd.merge(levy_model_set, \
                  levy_tpr_aggregated, \
                  left_on='A3', \
                  right_on='A3', \
                  how='left')

print("levy_model_set5")
print(levy_model_set)
print(levy_model_set.employees.value_counts())
print(levy_model_set.tpr_match.value_counts())
print(levy_model_set.company_status.value_counts())
print(levy_model_set.A1.value_counts())


# Create dummy variables for company type
company_type=pd.get_dummies(levy_model_set['company_type'],prefix='comp_type')
levy_model_set = levy_model_set.merge(company_type, left_index=True, right_index=True)

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

# Only keep relevant variables and rename accordingly

model_cols_to_keep=['A1','A3','cohort','months_since_sign_up2','adjusted_commitments','occupation_1', \
                    'occupation_2','occupation_3','occupation_7','occupation_13','occupation_14','occupation_15', \
                    'occupation_17','occupation_20','occupation_22','occupation_24','occupation_null','employees', \
                    'years_since_tpr_signup','comp_type_C','comp_type_I', \
                    'comp_type_X','tpr_match','new_company','early_adopter', \
                    'commitments_ending_12m','prev_12m_new_commitments','prev_12m_new_levy_transfers', \
                    'levy_sending_company','current_live_commitments','company_status']
levy_model_set = levy_model_set[model_cols_to_keep]
levy_model_set.columns = ['levy_non_levy','account_id','cohort','as_months_since_sign_up','adjusted_commitments','occupation_1', \
                     'occupation_2','occupation_3','occupation_7','occupation_13','occupation_14','occupation_15', \
                     'occupation_17','occupation_20','occupation_22','occupation_24','occupation_null','employees', \
                     'years_since_tpr_signup','comp_type_C','comp_type_I', \
                     'comp_type_X','tpr_match','new_company','early_adopter', \
                     'commitments_ending_12m','prev_12m_new_commitments','prev_12m_new_levy_transfers', \
                     'levy_sending_company','current_live_commitments','company_status']


# Take logs to standardise the scale
levy_model_set['log_adjusted_commitments'] = np.log2(levy_model_set['adjusted_commitments']+1)
levy_model_set['log_employees'] = np.log2(levy_model_set['employees']+1)

print("levy_model_set6")
print(levy_model_set)
print(levy_model_set.levy_non_levy.value_counts())
print(levy_model_set.employees.value_counts())
print(levy_model_set.tpr_match.value_counts())
print(levy_model_set.company_status.value_counts())


# Remove outliers and non matched tpr data and tpr closed companies
levy_model_set2 = levy_model_set[(levy_model_set.employees <=20000) & (levy_model_set.tpr_match ==1) & (levy_model_set.company_status ==3)]

print("levy model set 2_v1")
print(levy_model_set2)
print(levy_model_set2.levy_non_levy.value_counts())

# split the data into target and predictors
y = levy_model_set2['adjusted_commitments']

X = levy_model_set2[['levy_non_levy','as_months_since_sign_up','adjusted_commitments','occupation_1', \
                     'occupation_2','occupation_3','occupation_7','occupation_13','occupation_14','occupation_15', \
                     'occupation_17','occupation_20','occupation_22','occupation_24','occupation_null', \
                     'years_since_tpr_signup','comp_type_C','comp_type_I', \
                     'comp_type_X','new_company','early_adopter', \
                     'commitments_ending_12m','prev_12m_new_commitments','prev_12m_new_levy_transfers', \
                     'levy_sending_company','current_live_commitments']]

print(X.levy_non_levy.value_counts())

# Create train and test sets
X_train= pd.concat([X,X,X,X,X,X,X,X,X,X,X,X,X],ignore_index=True)
X_test= pd.concat([X,X,X,X,X,X,X,X,X],ignore_index=True)

y_train= pd.concat([y,y,y,y,y,y,y,y,y,y,y,y,y],ignore_index=True)
y_test= pd.concat([y,y,y,y,y,y,y,y,y],ignore_index=True)

print("X_train")
X_train

xgb_model = xgb.XGBRegressor(objective ='reg:squarederror')

xgb_model.fit(X_train, y_train)

explainer = shap.TreeExplainer(xgb_model)

# Save the trained model in the outputs folder
os.makedirs('outputs', exist_ok=True)
model_file = os.path.join('outputs', 'levy_model.pkl')
joblib.dump(value=xgb_model, filename=model_file)

# Register the model to pick up in scoring
Model.register(workspace=aml_workspace, \
               model_path = model_file, \
               model_name = 'levy_model')

run = Run.get_context()
run.log('levy_model_train','levy_model_train')

