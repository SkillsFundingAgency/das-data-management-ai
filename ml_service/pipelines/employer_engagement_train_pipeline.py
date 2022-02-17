import os
from azureml.core import Workspace, Experiment, Environment
from azureml.core.runconfig import RunConfiguration
from ml_service.util.env_variables import Env
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
experiment_folder = 'employer-engagement'
 
environment = Environment.get(workspace=aml_workspace, name="AzureML-xgboost-0.9-ubuntu18.04-py37-cpu-inference")

# Create a new runconfig object for the pipeline
pipeline_run_config = RunConfiguration()
# Use the compute you created above. 
pipeline_run_config.target = aml_compute
# Assign the environment to the run configuration
pipeline_run_config.environment = environment

#Create pipelines for levy models
get_levy_train(aml_workspace, aml_compute, pipeline_run_config, experiment) 
get_levy_score(aml_workspace, aml_compute, pipeline_run_config, experiment)
