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

def get_levy_score(aml_workspace: Workspace, aml_compute: str, pipeline_run_config: str, experiment: str) :
    # levy model score step
    score_source_dir="./employer-engagement/scoring"
    levy_model_score_step = PythonScriptStep(
        name='levy_model_score',
        script_name="levy_model_score.py",
        compute_target=aml_compute,
        runconfig=pipeline_run_config,
        source_directory=score_source_dir,
        allow_reuse=False)

    # Create sequence of steps
    levy_model_score_step_sequence = StepSequence(steps = [levy_model_score_step])

    # Create pipeline
    levy_model_score_pipeline = Pipeline(workspace=aml_workspace, steps=levy_model_score_step_sequence)
    levy_model_score_pipeline.validate()

    pipeline_endpoint = PipelineEndpoint.get(workspace=aml_workspace, name="levy-model-score-pipeline")
    pipeline_run= experiment.submit(pipeline_endpoint,pipeline_version="0")

    #levy_model_score_pipeline_run = experiment.submit(pipeline_endpoint, levy_model_score_pipeline,regenerate_outputs=True)
    levy_model_score_pipeline_run = experiment.submit(levy_model_score_pipeline,regenerate_outputs=True)

    # RunDetails(pipeline_run).show()
    # levy_model_score_pipeline_run.wait_for_completion()

    # Publish pipeline to AzureML
    levy_model_score_published_pipeline = levy_model_score_pipeline.publish('levy-model-score-pipeline')

    return