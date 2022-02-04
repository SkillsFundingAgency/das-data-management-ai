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

def get_non_levy_train(aml_workspace: Workspace, aml_compute: str, pipeline_run_config: str, experiment: str) :
    # non_levy_model train step
    train_source_dir="./employer-engagement/training"
    non_levy_model_train_step = PythonScriptStep(
        name='non_levy_model_train',
        script_name="non_levy_model_train.py",
        compute_target=aml_compute,
        runconfig=pipeline_run_config,
        source_directory=train_source_dir)


    # Create sequence of steps for model train
    non_levy_model_train_step_sequence = StepSequence(steps = [non_levy_model_train_step])

    # Create pipeline
    non_levy_model_train_pipeline = Pipeline(workspace=aml_workspace, steps=non_levy_model_train_step_sequence)
    non_levy_model_train_pipeline.validate()

    non_levy_model_train_pipeline_run = experiment.submit(non_levy_model_train_pipeline,regenerate_outputs=True)

    # RunDetails(pipeline_run).show()
    non_levy_model_train_pipeline_run.wait_for_completion()

    # Publish pipeline to AzureML
    non_levy_model_train_published_pipeline = non_levy_model_train_pipeline.publish('non_levy-model-train-pipeline')
    
    return
