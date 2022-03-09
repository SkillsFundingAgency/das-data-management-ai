from azureml.core import Workspace
from azureml.core.compute import ComputeTarget
# from ml_service.util.env_variables import Env
from azureml.pipeline.steps import PythonScriptStep
from azureml.pipeline.core import Pipeline, PipelineData, StepSequence, PublishedPipeline
from azureml.core.runconfig import RunConfiguration
from azureml.pipeline.core import PipelineEndpoint
import azureml.core
import os
from azureml.data.datapath import DataPath
from azureml.core import Workspace, Datastore, Dataset, ComputeTarget, Experiment, ScriptRunConfig, Environment, Model
from azureml.core.run import Run
# from ml_service.util.manage_environment import get_environment

# Set up config of workspace and datastore

aml_workspace = Run.get_context().experiment.workspace
datastore = Datastore.get(aml_workspace, datastore_name='datamgmtdb')

def levy_train_01_accounts() :
    query_levy_accounts = DataPath(datastore, """SELECT top 7 A1, A2, A3, CASE WHEN CAST(A2 AS DATE)<cast('2017-07-01' as date) THEN 1 ELSE 0 END AS early_adopter FROM PDS_AI.PT_A WHERE A1=0 ORDER BY RAND()""")
    tabular_levy_accounts = Dataset.Tabular.from_sql_query(query_levy_accounts, query_timeout=10)
    levy_model_accounts = tabular_levy_accounts.to_pandas_dataframe()
    
    return levy_model_accounts

# def levy_train_01_accounts(top_x: str) :
    # query_levy_accounts = DataPath(datastore, "SELECT top " + str(top_x) + " A1, A2, A3, CASE WHEN CAST(A2 AS DATE)<cast('2017-07-01' as date) THEN 1 ELSE 0 END AS early_adopter FROM PDS_AI.PT_A WHERE A1=0 ORDER BY RAND()")
    # tabular_levy_accounts = Dataset.Tabular.from_sql_query(query_levy_accounts, query_timeout=10)
    # levy_model_accounts = tabular_levy_accounts.to_pandas_dataframe()
    
    # return levy_model_accounts

def levy_train_01_accounts2(top_x: str) :
    query_levy_accounts = DataPath(datastore, "SELECT top {} A1, A2, A3, CASE WHEN CAST(A2 AS DATE)<cast('2017-07-01' as date) THEN 1 ELSE 0 END AS early_adopter FROM PDS_AI.PT_A WHERE A1=0 ORDER BY RAND()".format(top_x))
    tabular_levy_accounts = Dataset.Tabular.from_sql_query(query_levy_accounts, query_timeout=10)
    levy_model_accounts2 = tabular_levy_accounts.to_pandas_dataframe()
    
    return levy_model_accounts2


# def levy_train_02_levy_model_set_2018_2019_part1() :
    # query_2018_2019_part1 = DataPath(datastore, """SELECT A3 \
    # , '2019' as cohort \
    # , total_commitments \
    # , occupation_1 \
    # , occupation_2 \
    # , occupation_3 \
    # , occupation_7 \
    # , occupation_13 \
    # , occupation_14 \
    # , occupation_15 \
    # , occupation_17 \
    # , occupation_20 \
    # , occupation_22 \
    # , occupation_24 \
    # , occupation_null \
    # , prev_12m_new_commitments \
    # , prev_12m_new_levy_transfers \
    # , A7 as levy_sending_company \
    # FROM \
    # (SELECT A3, CONCAT(YEAR(A2),'-',month(A2)) as yearmon_created, A1 as levy_split, A2, A7 \
    # FROM PDS_AI.PT_A \
    # WHERE A2<'2018-04-01' and A1=1 \
    # ) A \
    # LEFT JOIN \
    # (SELECT B10, count(*) AS total_commitments \
    # FROM PDS_AI.PT_B \
    # WHERE cast(B2 as date) >= '2018-04-01' AND cast(B2 as date) < '2019-04-01' \
    # GROUP BY B10 \
    # ) B \
    # ON A.A3=B.B10 \
    # LEFT JOIN \
    # (SELECT B10 \
    # , COUNT(*) AS prev_12m_new_commitments \
    # , SUM(CASE WHEN B12=1 THEN 1 ELSE 0 END) AS prev_12m_new_levy_transfers \
    # , CAST(SUM(CASE WHEN B6 = '1' THEN 1.000 ELSE 0 END) / COUNT(*) AS DECIMAL(10,3)) AS occupation_1 \
    # , CAST(SUM(CASE WHEN B6 = '2' THEN 1.000 ELSE 0 END) / COUNT(*) AS DECIMAL(10,3)) AS occupation_2 \
    # , CAST(SUM(CASE WHEN B6 = '3' THEN 1.000 ELSE 0 END) / COUNT(*) AS DECIMAL(10,3)) AS occupation_3 \
    # , CAST(SUM(CASE WHEN B6 = '7' THEN 1.000 ELSE 0 END) / COUNT(*) AS DECIMAL(10,3)) AS occupation_7 \
    # , CAST(SUM(CASE WHEN B6 = '13' THEN 1.000 ELSE 0 END) / COUNT(*) AS DECIMAL(10,3)) AS occupation_13 \
    # , CAST(SUM(CASE WHEN B6 = '14' THEN 1.000 ELSE 0 END) / COUNT(*) AS DECIMAL(10,3)) AS occupation_14 \
    # , CAST(SUM(CASE WHEN B6 = '15' THEN 1.000 ELSE 0 END) / COUNT(*) AS DECIMAL(10,3)) AS occupation_15 \
    # , CAST(SUM(CASE WHEN B6 = '17' THEN 1.000 ELSE 0 END) / COUNT(*) AS DECIMAL(10,3)) AS occupation_17 \
    # , CAST(SUM(CASE WHEN B6 = '20' THEN 1.000 ELSE 0 END) / COUNT(*) AS DECIMAL(10,3)) AS occupation_20 \
    # , CAST(SUM(CASE WHEN B6 = '22' THEN 1.000 ELSE 0 END) / COUNT(*) AS DECIMAL(10,3)) AS occupation_22 \
    # , CAST(SUM(CASE WHEN B6 = '24' THEN 1.000 ELSE 0 END) / COUNT(*) AS DECIMAL(10,3)) AS occupation_24 \
    # , CAST(SUM(CASE WHEN B6 = NULL THEN 1.000 ELSE 0 END) / COUNT(*) AS DECIMAL(10,3)) AS occupation_null \
    # FROM PDS_AI.PT_B \
    # WHERE cast(B2 as date) >= '2017-04-01' AND cast(B2 as date) < '2018-04-01' \
    # GROUP BY B10 \
    # ) C \
    # ON A.A3=C.B10""")
    # tabular_2018_2019_part1 = Dataset.Tabular.from_sql_query(query_2018_2019_part1, query_timeout=10)
    # levy_model_set_2018_2019_part1 = tabular_2018_2019_part1.to_pandas_dataframe()
    
    # return levy_model_set_2018_2019_part1

# def levy_train_03_levy_model_set_2018_2019_part2() :
    # query_2018_2019_part2 = DataPath(datastore, """SELECT A3 \
    # , commitments_ending_12m \
    # , current_live_commitments \
    # FROM  \
    # (SELECT A3, CONCAT(YEAR(A2),'-',month(A2)) as yearmon_created, A1 as levy_split, A2, A7 \
    # FROM PDS_AI.PT_A \
    # WHERE A2<'2018-04-01' and A1=1 \
    # ) A  \
    # LEFT JOIN  \
    # (SELECT B10 \
    # , COUNT(*) AS commitments_ending_12m \
    # FROM PDS_AI.PT_B \
    # WHERE CAST(B17 AS DATE) < '2019-04-01' AND CAST(B17 AS DATE)>='2018-04-01'  \
    # AND (CAST(B20 AS DATE) >= '2018-04-01' OR B20 IS NULL)  \
    # AND (CAST(B16 AS DATE) >= '2018-04-01' OR B16 IS NULL)  \
    # GROUP BY B10 \
    # ) D  \
    # ON A.A3=D.B10 \
    # LEFT JOIN  \
    # (SELECT B10 \
    # , COUNT(*) AS current_live_commitments \
    # FROM PDS_AI.PT_B \
    # WHERE cast(B2 AS DATE) < '2018-04-01' AND  \
    # (B20 IS NULL OR CAST(B20 AS DATE)>='2018-04-01') AND \
    # (B16 IS NULL OR CAST(B16 AS DATE)>='2018-04-01') \
    # GROUP BY B10 \
    # ) E \
    # ON A.A3=E.B10""")
    # tabular_2018_2019_part2 = Dataset.Tabular.from_sql_query(query_2018_2019_part2, query_timeout=10)
    # levy_model_set_2018_2019_part2 = tabular_2018_2019_part2.to_pandas_dataframe()

    # return levy_model_set_2018_2019_part2

# def levy_train_04_levy_model_set_2019_2020_part1() :
    # query_2019_2020_part1 = DataPath(datastore, """SELECT A3 \
    # , '2020' as cohort \
    # , total_commitments \
    # , occupation_1 \
    # , occupation_2 \
    # , occupation_3 \
    # , occupation_7 \
    # , occupation_13 \
    # , occupation_14 \
    # , occupation_15 \
    # , occupation_17 \
    # , occupation_20 \
    # , occupation_22 \
    # , occupation_24 \
    # , occupation_null \
    # , prev_12m_new_commitments \
    # , prev_12m_new_levy_transfers \
    # , A7 as levy_sending_company \
    # FROM \
    # (SELECT A3, CONCAT(YEAR(A2),'-',month(A2)) as yearmon_created, A1 as levy_split, A2, A7 \
    # FROM PDS_AI.PT_A \
    # WHERE A2<'2019-04-01' AND A1=1 \
    # ) A \
    # LEFT JOIN \
    # (SELECT B10, count(*) AS total_commitments \
    # FROM PDS_AI.PT_B \
    # WHERE cast(B2 as date) >= '2019-04-01' AND cast(B2 as date) < '2020-04-01' \
    # GROUP BY B10 \
    # ) B \
    # ON A.A3=B.B10 \
    # LEFT JOIN \
    # (SELECT B10 \
    # , COUNT(*) AS prev_12m_new_commitments \
    # , SUM(CASE WHEN B12=1 THEN 1 ELSE 0 END) AS prev_12m_new_levy_transfers \
    # , CAST(SUM(CASE WHEN B6 = '1' THEN 1.000 ELSE 0 END) / COUNT(*) AS DECIMAL(10,3)) AS occupation_1 \
    # , CAST(SUM(CASE WHEN B6 = '2' THEN 1.000 ELSE 0 END) / COUNT(*) AS DECIMAL(10,3)) AS occupation_2 \
    # , CAST(SUM(CASE WHEN B6 = '3' THEN 1.000 ELSE 0 END) / COUNT(*) AS DECIMAL(10,3)) AS occupation_3 \
    # , CAST(SUM(CASE WHEN B6 = '7' THEN 1.000 ELSE 0 END) / COUNT(*) AS DECIMAL(10,3)) AS occupation_7 \
    # , CAST(SUM(CASE WHEN B6 = '13' THEN 1.000 ELSE 0 END) / COUNT(*) AS DECIMAL(10,3)) AS occupation_13 \
    # , CAST(SUM(CASE WHEN B6 = '14' THEN 1.000 ELSE 0 END) / COUNT(*) AS DECIMAL(10,3)) AS occupation_14 \
    # , CAST(SUM(CASE WHEN B6 = '15' THEN 1.000 ELSE 0 END) / COUNT(*) AS DECIMAL(10,3)) AS occupation_15 \
    # , CAST(SUM(CASE WHEN B6 = '17' THEN 1.000 ELSE 0 END) / COUNT(*) AS DECIMAL(10,3)) AS occupation_17 \
    # , CAST(SUM(CASE WHEN B6 = '20' THEN 1.000 ELSE 0 END) / COUNT(*) AS DECIMAL(10,3)) AS occupation_20 \
    # , CAST(SUM(CASE WHEN B6 = '22' THEN 1.000 ELSE 0 END) / COUNT(*) AS DECIMAL(10,3)) AS occupation_22 \
    # , CAST(SUM(CASE WHEN B6 = '24' THEN 1.000 ELSE 0 END) / COUNT(*) AS DECIMAL(10,3)) AS occupation_24 \
    # , CAST(SUM(CASE WHEN B6 = NULL THEN 1.000 ELSE 0 END) / COUNT(*) AS DECIMAL(10,3)) AS occupation_null \
    # FROM PDS_AI.PT_B \
    # WHERE cast(B2 as date) >= '2018-04-01' AND cast(B2 as date) < '2019-04-01' \
    # GROUP BY B10 \
    # ) C \
    # ON A.A3=C.B10""")
    # tabular_2019_2020_part1 = Dataset.Tabular.from_sql_query(query_2019_2020_part1, query_timeout=10)
    # levy_model_set_2019_2020_part1 = tabular_2019_2020_part1.to_pandas_dataframe()
    
    # return levy_model_set_2019_2020_part1

# def levy_train_05_levy_model_set_2019_2020_part2() :
    # query_2019_2020_part2 = DataPath(datastore, """SELECT A3 \
    # , commitments_ending_12m \
    # , current_live_commitments \
    # FROM  \
    # (SELECT A3, CONCAT(YEAR(A2),'-',month(A2)) as yearmon_created, A1 as levy_split, A2, A7 \
    # FROM PDS_AI.PT_A \
    # WHERE A2<'2019-04-01' AND A1=1 \
    # ) A  \
    # LEFT JOIN  \
    # (SELECT B10 \
    # , COUNT(*) AS commitments_ending_12m \
    # FROM PDS_AI.PT_B \
    # WHERE CAST(B17 AS DATE) < '2020-04-01' AND CAST(B17 AS DATE)>='2019-04-01'  \
    # AND (CAST(B20 AS DATE) >= '2019-04-01' OR B20 IS NULL)  \
    # AND (CAST(B16 AS DATE) >= '2019-04-01' OR B16 IS NULL)  \
    # GROUP BY B10 \
    # ) D  \
    # ON A.A3=D.B10 \
    # LEFT JOIN  \
    # (SELECT B10 \
    # , COUNT(*) AS current_live_commitments \
    # FROM PDS_AI.PT_B \
    # WHERE cast(B2 AS DATE) < '2019-04-01' AND  \
    # (B20 IS NULL OR CAST(B20 AS DATE)>='2019-04-01') AND \
    # (B16 IS NULL OR CAST(B16 AS DATE)>='2019-04-01') \
    # GROUP BY B10 \
    # ) E \
    # ON A.A3=E.B10""")
    # tabular_2019_2020_part2 = Dataset.Tabular.from_sql_query(query_2019_2020_part2, query_timeout=10)
    # levy_model_set_2019_2020_part2 = tabular_2019_2020_part2.to_pandas_dataframe()

    # return levy_model_set_2019_2020_part2

# def levy_train_06_levy_model_set_2022_part1() :
    # query_2022_part1 = DataPath(datastore, """SELECT A3 \
    # , '2022' as cohort \
    # , total_commitments \
    # , occupation_1 \
    # , occupation_2 \
    # , occupation_3 \
    # , occupation_7 \
    # , occupation_13 \
    # , occupation_14 \
    # , occupation_15 \
    # , occupation_17 \
    # , occupation_20 \
    # , occupation_22 \
    # , occupation_24 \
    # , occupation_null \
    # , prev_12m_new_commitments \
    # , prev_12m_new_levy_transfers \
    # , A7 as levy_sending_company \
    # FROM \
    # (SELECT A3, CONCAT(YEAR(A2),'-',month(A2)) as yearmon_created, A1 as levy_split, A2, A7 \
    # FROM PDS_AI.PT_A \
    # WHERE A2<'2021-01-01' AND A1=1 \
    # ) A \
    # LEFT JOIN \
    # (SELECT B10, count(*) AS total_commitments \
    # FROM PDS_AI.PT_B \
    # WHERE cast(B2 as date) >= '2021-01-01' AND cast(B2 as date) < '2022-01-01' \
    # GROUP BY B10 \
    # ) B \
    # ON A.A3=B.B10 \
    # LEFT JOIN \
    # (SELECT B10 \
    # , COUNT(*) AS prev_12m_new_commitments \
    # , SUM(CASE WHEN B12=1 THEN 1 ELSE 0 END) AS prev_12m_new_levy_transfers \
    # , CAST(SUM(CASE WHEN B6 = '1' THEN 1.000 ELSE 0 END) / COUNT(*) AS DECIMAL(10,3)) AS occupation_1 \
    # , CAST(SUM(CASE WHEN B6 = '2' THEN 1.000 ELSE 0 END) / COUNT(*) AS DECIMAL(10,3)) AS occupation_2 \
    # , CAST(SUM(CASE WHEN B6 = '3' THEN 1.000 ELSE 0 END) / COUNT(*) AS DECIMAL(10,3)) AS occupation_3 \
    # , CAST(SUM(CASE WHEN B6 = '7' THEN 1.000 ELSE 0 END) / COUNT(*) AS DECIMAL(10,3)) AS occupation_7 \
    # , CAST(SUM(CASE WHEN B6 = '13' THEN 1.000 ELSE 0 END) / COUNT(*) AS DECIMAL(10,3)) AS occupation_13 \
    # , CAST(SUM(CASE WHEN B6 = '14' THEN 1.000 ELSE 0 END) / COUNT(*) AS DECIMAL(10,3)) AS occupation_14 \
    # , CAST(SUM(CASE WHEN B6 = '15' THEN 1.000 ELSE 0 END) / COUNT(*) AS DECIMAL(10,3)) AS occupation_15 \
    # , CAST(SUM(CASE WHEN B6 = '17' THEN 1.000 ELSE 0 END) / COUNT(*) AS DECIMAL(10,3)) AS occupation_17 \
    # , CAST(SUM(CASE WHEN B6 = '20' THEN 1.000 ELSE 0 END) / COUNT(*) AS DECIMAL(10,3)) AS occupation_20 \
    # , CAST(SUM(CASE WHEN B6 = '22' THEN 1.000 ELSE 0 END) / COUNT(*) AS DECIMAL(10,3)) AS occupation_22 \
    # , CAST(SUM(CASE WHEN B6 = '24' THEN 1.000 ELSE 0 END) / COUNT(*) AS DECIMAL(10,3)) AS occupation_24 \
    # , CAST(SUM(CASE WHEN B6 = NULL THEN 1.000 ELSE 0 END) / COUNT(*) AS DECIMAL(10,3)) AS occupation_null \
    # FROM PDS_AI.PT_B \
    # WHERE cast(B2 as date) >= '2020-01-01' AND cast(B2 as date) < '2021-01-01' \
    # GROUP BY B10 \
    # ) C \
    # ON A.A3=C.B10""")
    # tabular_2022_part1 = Dataset.Tabular.from_sql_query(query_2022_part1, query_timeout=10)
    # levy_model_set_2022_part1 = tabular_2022_part1.to_pandas_dataframe()
    
    # return levy_model_set_2022_part1

# def levy_train_07_levy_model_set_2022_part2() :
    # query_2022_part2 = DataPath(datastore, """SELECT A3 \
    # , commitments_ending_12m \
    # , current_live_commitments \
    # FROM  \
    # (SELECT A3, CONCAT(YEAR(A2),'-',month(A2)) as yearmon_created, A1 as levy_split, A2, A7 \
    # FROM PDS_AI.PT_A \
    # WHERE A2<'2021-01-01' AND A1=1 \
    # ) A  \
    # LEFT JOIN  \
    # (SELECT B10 \
    # , COUNT(*) AS commitments_ending_12m \
    # FROM PDS_AI.PT_B \
    # WHERE cast(B17 as date) < '2022-01-01' AND CAST(B17 AS DATE)>='2021-01-01'  \
    # AND (CAST(B20 AS DATE) >= '2021-01-01' OR B20 IS NULL)  \
    # AND (CAST(B16 AS DATE) >= '2021-01-01' OR B16 IS NULL)  \
    # GROUP BY B10 \
    # ) D  \
    # ON A.A3=D.B10 \
    # LEFT JOIN  \
    # (SELECT B10 \
    # , COUNT(*) AS current_live_commitments \
    # FROM PDS_AI.PT_B \
    # WHERE cast(B2 AS DATE) < '2021-01-01' AND  \
    # (B20 IS NULL OR CAST(B20 AS DATE)>='2021-01-01') AND \
    # (B16 IS NULL OR CAST(B16 AS DATE)>='2021-01-01') \
    # GROUP BY B10 \
    # ) E \
    # ON A.A3=E.B10""")
    # tabular_2022_part2 = Dataset.Tabular.from_sql_query(query_2022_part2, query_timeout=10)
    # levy_model_set_2022_part2 = tabular_2022_part2.to_pandas_dataframe()

    # return levy_model_set_2022_part2
