import azureml.core
import os
from azureml.core import Workspace, Datastore, Dataset, ComputeTarget, Experiment, ScriptRunConfig, Environment, Model
from azureml.core.run import Run
from azureml.core.runconfig import RunConfiguration
from azureml.pipeline.steps import PythonScriptStep
from azureml.pipeline.core import PipelineParameter
from azureml.pipeline.core import Pipeline, PipelineData, StepSequence, PublishedPipeline
from ml_service.util.env_variables import Env
from ml_service.util.manage_environment import get_environment
from ml_service.pipelines.attach_levy_train import get_levy_train
from ml_service.pipelines.attach_levy_score import get_levy_score

# Set up all environment details
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

# Create experiment if it doesn't exist
experiment = Experiment(aml_workspace, "employer-engagement")

# Add compute information to variable
aml_compute = aml_workspace.compute_targets["cpucluster"]

# Set up experiment folder
experiment_folder = 'employer_engagement'
 
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

# Adding a PipelineParameter to a step
pipeline_param = PipelineParameter(name="pipeline_arg", default_value="6")
   

#Create pipelines for levy models
get_levy_train(aml_workspace, aml_compute, pipeline_run_config, experiment,pipeline_param) 
get_levy_score(aml_workspace, aml_compute, pipeline_run_config, experiment)
