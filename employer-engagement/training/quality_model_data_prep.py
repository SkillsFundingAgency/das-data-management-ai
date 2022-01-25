import pickle
import os
import pandas as pd
import pyarrow.parquet as pq
import joblib
import datetime
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
(SELECT A3, A1 as levy_split, A2 FROM PDS_AI.PT_A WHERE A2<cast('2018-07-01' as date)) A  \
INNER JOIN  \
(SELECT B10, B15, B3, CAST(B2 AS DATE) AS commitment_date, B1 \
FROM PDS_AI.PT_B \
WHERE CAST(B2 AS DATE) >= cast('2018-01-01' as date) AND CAST(B2 AS DATE) < cast('2018-07-01' as date) \
AND B3 IN (2,3,4,5) \
AND (B15=1 OR B16 IS NOT NULL OR B19=1) \
) B  \
ON A.A3=B.B10) C \
LEFT JOIN  \
(SELECT B10, CAST(B2 AS DATE) as B2 \
FROM PDS_AI.PT_B  \
WHERE CAST(B2 AS DATE) < cast('2018-07-01' as date) \
AND B3 IN (2,3,4,5)  \
) D  \
ON C.A3=D.B10 \
GROUP BY C.A3  \
, C.levy_split \
, C.account_created \
, C.completed_commitment \
, C.apprenticeship_id \
, C.commitment_date \
""") 
tabular_2018_H1 = Dataset.Tabular.from_sql_query(query_2018_H1, query_timeout=1000)
quality_model_set_2018_H1 = tabular_2018_H1.to_pandas_dataframe()




print(quality_model_set_2018_H1)




run = Run.get_context()
run.log('quality_model_data_prep.log','quality_model_data_prep.log')
