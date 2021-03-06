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

def get_levy_train(aml_workspace: Workspace, aml_compute: str, pipeline_run_config: str, experiment: str) :
    # levy_model train step
    train_source_dir="./employer_engagement/training"
    levy_model_train_step = PythonScriptStep(
        name='levy_model_train',
        script_name="levy_model_train.py",
        compute_target=aml_compute,
        runconfig=pipeline_run_config,
        source_directory=train_source_dir,
        allow_reuse=False)


    # Create sequence of steps for model train
    levy_model_train_step_sequence = StepSequence(steps = [levy_model_train_step])

    # Create pipeline
    levy_model_train_pipeline = Pipeline(workspace=aml_workspace, steps=levy_model_train_step_sequence)
    levy_model_train_pipeline.validate()

    # Publish pipeline to AzureML
    levy_model_train_published_pipeline = levy_model_train_pipeline.publish('levy-model-train-pipeline')

    # create Pipeline Endpoint if not already exists , if exists add pipeline to the endpoint
    
    try:
          pipeline_endpoint = PipelineEndpoint.get(workspace=aml_workspace, name="levy_train_model_endpoint")
          pipeline_endpoint.add_default(levy_model_train_published_pipeline)
    except Exception:
          pipeline_endpoint = PipelineEndpoint.publish(workspace=aml_workspace,
                                                       name="levy_train_model_endpoint", 
                                                       pipeline=levy_model_train_published_pipeline,
                                                       description="Endpoint to Levy Train pipeline",
                                                       )
    
    return
