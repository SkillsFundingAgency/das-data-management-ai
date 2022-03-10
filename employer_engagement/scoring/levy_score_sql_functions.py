from azureml.core import Workspace
from azureml.core.compute import ComputeTarget
from azureml.pipeline.steps import PythonScriptStep
from azureml.pipeline.core import Pipeline, PipelineData, StepSequence, PublishedPipeline
from azureml.core.runconfig import RunConfiguration
from azureml.pipeline.core import PipelineEndpoint
import azureml.core
import os
from azureml.data.datapath import DataPath
from azureml.core import Workspace, Datastore, Dataset, ComputeTarget, Experiment, ScriptRunConfig, Environment, Model
from azureml.core.run import Run

# Set up config of workspace and datastore

aml_workspace = Run.get_context().experiment.workspace
datastore = Datastore.get(aml_workspace, datastore_name='datamgmtdb')

def levy_score_01_accounts(top_x: str) :
    query_levy_accounts = DataPath(datastore, "SELECT top {} A1, A2, A3, CASE WHEN CAST(A2 AS DATE)<cast('2017-07-01' as date) THEN 1 ELSE 0 END AS early_adopter FROM PDS_AI.PT_A WHERE A1=1 ORDER BY RAND()".format(top_x))
    tabular_levy_accounts = Dataset.Tabular.from_sql_query(query_levy_accounts, query_timeout=3600)
    levy_score_set = tabular_levy_accounts.to_pandas_dataframe()
    
    return levy_score_set


def levy_score_02_levy_commitments_part1(sql_account_list: str) :
    query_current_part1 = DataPath(datastore, "SELECT A3 \
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
    WHERE CAST(A2 AS DATE)<DATEADD(day,-365,CAST(GETDATE() AS Date)) AND A1=1 AND A3 in ({0}) \
    ) A \
    LEFT JOIN \
    (SELECT B10, count(*) AS total_commitments \
    FROM PDS_AI.PT_B \
    WHERE cast(B2 as date) >= DATEADD(day,-365,CAST(GETDATE() AS Date)) AND B10 in ({0}) \
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
    AND B10 in ({0}) \
    GROUP BY B10 \
    ) C \
    ON A.A3=C.B10".format({sql_account_list}))
    tabular_current_part1 = Dataset.Tabular.from_sql_query(query_current_part1, query_timeout=10)
    levy_commitments_part1 = tabular_current_part1.to_pandas_dataframe()
    
    return levy_commitments_part1

# def levy_score_03_levy_commitments_part2(sql_account_list: str) :
    # query_current_part2 = DataPath(datastore, "SELECT A3 \
    # , commitments_ending_12m \
    # , current_live_commitments \
    # FROM \
    # (SELECT A3, CONCAT(YEAR(A2),'-',month(A2)) as yearmon_created, A1 as levy_split, A2, A7 \
    # FROM PDS_AI.PT_A \
    # WHERE CAST(A2 AS DATE)<DATEADD(day,-365,CAST(GETDATE() AS Date)) AND A1=1 AND A3 in ({0})\
    # ) A \
    # LEFT JOIN \
    # (SELECT B10 \
    # , COUNT(*) AS commitments_ending_12m \
    # FROM PDS_AI.PT_B \
    # WHERE cast(B17 as date) < CAST(GETDATE() AS Date) AND CAST(B17 AS DATE)>=DATEADD(day,-365,CAST(GETDATE() AS Date)) \
    # AND (CAST(B20 AS DATE) >= DATEADD(day,-365,CAST(GETDATE() AS Date)) OR B20 IS NULL) \
    # AND (CAST(B16 AS DATE) >= DATEADD(day,-365,CAST(GETDATE() AS Date)) OR B16 IS NULL) \
    # AND B10 in ({0}) \
    # GROUP BY B10 \
    # ) D \
    # ON A.A3=D.B10 \
    # LEFT JOIN \
    # (SELECT B10 \
    # , COUNT(*) AS current_live_commitments \
    # FROM PDS_AI.PT_B \
    # WHERE cast(B2 AS DATE) < DATEADD(day,-365,CAST(GETDATE() AS Date)) AND \
    # (B20 IS NULL OR CAST(B20 AS DATE)>=CAST(GETDATE() AS Date)) AND \
    # (B16 IS NULL OR CAST(B16 AS DATE)>=CAST(GETDATE() AS Date)) \
    # AND B10 in ({0}) \
    # GROUP BY B10 \
    # ) E \
    # ON A.A3=E.B10".format({sql_account_list}))
    # tabular_current_part2 = Dataset.Tabular.from_sql_query(query_current_part2, query_timeout=10)
    # levy_commitments_part2 = tabular_current_part2.to_pandas_dataframe()

    # return levy_commitments_part2
