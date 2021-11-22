import azureml.core
import os
from azureml.core import Workspace, Datastore, Dataset, ComputeTarget, Experiment, ScriptRunConfig, Environment
from azureml.core.run import Run
from azureml.core.runconfig import RunConfiguration
from azureml.pipeline.steps import PythonScriptStep
from azureml.pipeline.core import Pipeline, PipelineData
from azureml.pipeline.core import PublishedPipeline
from ml_service.util.attach_compute import get_compute
from ml_service.util.env_variables import Env
from ml_service.util.manage_environment import get_environment
import os

e = Env()

subscription_id = os.environ.get("SUBSCRIPTION_ID")
workspace_name = os.environ.get("WORKSPACE_NAME")
resource_group = os.environ.get("RESOURCE_GROUP")

# Get Azure machine learning workspace
aml_workspace = Workspace.get(
name=e.workspace_name,
subscription_id=e.subscription_id,
resource_group=e.resource_group,
)
print("get_workspace:")
print(aml_workspace)

experiment = Experiment(aml_workspace, "employer-engagement")

# aml_compute = get_compute(aml_workspace, e.compute_name, e.vm_size)
# if aml_compute is not None:
    # print("aml_compute:")
    # print(aml_compute)

aml_compute = aml_workspace.compute_targets["cpucluster"]

experiment_folder = 'employer-engagement'

# Create a Python environment for the experiment (from a .yml file)
environment = Environment.from_conda_specification("environment", experiment_folder + "/conda_dependencies.yml")
# Register the environment 
environment.register(workspace=aml_workspace)
registered_env = Environment.get(aml_workspace, 'environment')
# Create a new runconfig object for the pipeline
pipeline_run_config = RunConfiguration()
# Use the compute you created above. 
pipeline_run_config.target = aml_compute
# Assign the environment to the run configuration
pipeline_run_config.environment = registered_env
print ("Run configuration created.")

# environment = get_environment(
    # aml_workspace,
    # e.aml_env_name,
    # conda_dependencies_file=aml_env_train_conda_dep_file,
    # create_new=e.rebuild_env,
# )  
# run_config = RunConfiguration()
# run_config.environment = environment


score_source_dir="/scoring"
score_step = PythonScriptStep(
    name='scoring',
    script_name="score.py",
    compute_target=aml_compute,
    runconfig=pipeline_run_config,
    source_directory=score_source_dir)


steps = [score_step]
# Create pipeline
pipeline = Pipeline(workspace=aml_workspace, steps=steps)
pipeline.validate()

# Publish pipeline to AzureML
published_pipeline = pipeline.publish('model-scoring-pipeline')


