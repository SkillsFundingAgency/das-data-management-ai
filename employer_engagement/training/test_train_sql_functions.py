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

def test_train_sql_exec(top_x: str) :
    query_test_train = DataPath(datastore, "SELECT TOP({}) * FROM [Pds_AI].[PT_Test]".format(top_x))
    tabular_test_train = Dataset.Tabular.from_sql_query(query_test_train, query_timeout=3600)
    test_train_df = tabular_test_train.to_pandas_dataframe()
    
    return test_train_df


