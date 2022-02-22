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

def get_non_levy_score(aml_workspace: Workspace, aml_compute: str, pipeline_run_config: str, experiment: str) :
    # non_levy model score step
    score_source_dir="./employer-engagement/scoring"
    non_levy_model_score_step = PythonScriptStep(
        name='non_levy_model_score',
        script_name="non_levy_model_score.py",
        compute_target=aml_compute,
        runconfig=pipeline_run_config,
        source_directory=score_source_dir)

    # Create sequence of steps
    non_levy_model_score_step_sequence = StepSequence(steps = [non_levy_model_score_step])

    # Create pipeline
    non_levy_model_score_pipeline = Pipeline(workspace=aml_workspace, steps=non_levy_model_score_step_sequence)
    non_levy_model_score_pipeline.validate()

    non_levy_model_score_pipeline_run = experiment.submit(non_levy_model_score_pipeline,regenerate_outputs=True)

    # RunDetails(pipeline_run).show()
    non_levy_model_score_pipeline_run.wait_for_completion()

    # Publish pipeline to AzureML
    non_levy_model_score_published_pipeline = non_levy_model_score_pipeline.publish('non_levy-model-score-pipeline')

    return