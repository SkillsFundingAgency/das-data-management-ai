from azureml.core import Workspace
from azureml.core.compute import ComputeTarget
from ml_service.util.env_variables import Env
from azureml.pipeline.steps import PythonScriptStep
from azureml.pipeline.core import Pipeline, PipelineData, StepSequence, PublishedPipeline
from azureml.core.runconfig import RunConfiguration

import azureml.core
import os
from azureml.core import Workspace, Datastore, Dataset, ComputeTarget, Experiment, ScriptRunConfig, Environment, Model
from azureml.core.run import Run
from ml_service.util.manage_environment import get_environment


# Set up all environment details
# e = Env()
# subscription_id = os.environ.get("SUBSCRIPTION_ID")
# workspace_name = os.environ.get("WORKSPACE_NAME")
# resource_group = os.environ.get("RESOURCE_GROUP")

# Get Azure machine learning workspace
# aml_workspace = Workspace.get(
# name=e.workspace_name,
# subscription_id=e.subscription_id,
# resource_group=e.resource_group,
# )
# print("get_workspace:")
# print(aml_workspace)

# Create experiment if it doesn't exist
# experiment = Experiment(aml_workspace, "employer-engagement")

# Add compute information to variable
# aml_compute = aml_workspace.compute_targets["cpucluster"]

# Set up experiment folder
# experiment_folder = 'employer-engagement'

# Create a Python environment for the experiment (from a .yml file)
# environment = Environment.from_conda_specification("environment", experiment_folder + "/conda_dependencies.yml")
# Register the environment 
# environment.register(workspace=aml_workspace)
# registered_env = Environment.get(aml_workspace, 'environment')
# Create a new runconfig object for the pipeline
# pipeline_run_config = RunConfiguration()
# Use the compute you created above. 
# pipeline_run_config.target = aml_compute
# Assign the environment to the run configuration
# pipeline_run_config.environment = registered_env
# print ("Run configuration created.")


def get_quality_train(aml_workspace: Workspace, aml_compute: str) :
# Use train step to build and model
    train_source_dir="./employer-engagement/training"
    train_step = PythonScriptStep(
        name='model_build',
        script_name="train.py",
        compute_target=aml_compute,
        runconfig=pipeline_run_config,
        source_directory=train_source_dir)

    # Create sequence of steps for model train
    train_step_sequence = StepSequence(steps = [train_step])

    # Create pipeline
    train_pipeline = Pipeline(workspace=aml_workspace, steps=train_step_sequence)
    train_pipeline.validate()

    train_pipeline_run = experiment.submit(train_pipeline,regenerate_outputs=True)

    # RunDetails(pipeline_run).show()
    train_pipeline_run.wait_for_completion()

    # Publish pipeline to AzureML
    train_published_pipeline = train_pipeline.publish('train-pipeline')
    
    return
