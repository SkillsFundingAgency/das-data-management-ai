from azureml.core import Workspace
from azureml.core.compute import ComputeTarget
from ml_service.util.env_variables import Env
from azureml.pipeline.steps import PythonScriptStep
from azureml.pipeline.core import Pipeline, PipelineData, StepSequence, PublishedPipeline
from azureml.core.runconfig import RunConfiguration
from azureml.pipeline.core import PipelineEndpoint
import azureml.core
import os
from azureml.core import Workspace, Datastore, Dataset, ComputeTarget, Experiment, ScriptRunConfig, Environment, Model
from azureml.core.run import Run
from ml_service.util.manage_environment import get_environment

# def levy_train_01_accounts() :
    # query_levy_accounts = DataPath(datastore, """SELECT A1, A2, A3, CASE WHEN CAST(A2 AS DATE)<'2017-07-01' THEN 1 ELSE 0 END AS early_adopter FROM PDS_AI.PT_A where A1=1""")
    # tabular_levy_accounts = Dataset.Tabular.from_sql_query(query_levy_accounts, query_timeout=10)
    # levy_model_accounts = tabular_levy_accounts.to_pandas_dataframe()
    
    # return levy_model_accounts

def levy_train_01_accounts(aml_workspace: Workspace, aml_compute: str, pipeline_run_config: str, experiment: str, top_x: str) :
    query_levy_accounts = DataPath(datastore, "SELECT top " + str(top_x) + "A1, A2, A3, CASE WHEN CAST(A2 AS DATE)<cast('2017-07-01' as date) THEN 1 ELSE 0 END AS early_adopter FROM PDS_AI.PT_A WHERE A1=0 ORDER BY RAND()"
    tabular_levy_accounts = Dataset.Tabular.from_sql_query(query_levy_accounts, query_timeout=10)
    levy_model_accounts = tabular_levy_accounts.to_pandas_dataframe()
    
    return levy_model_accounts



