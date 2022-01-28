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

def get_quality_score(aml_workspace: Workspace, aml_compute: str, pipeline_run_config: str, experiment: str) :
    # quality model score step
    score_source_dir="./employer-engagement/scoring"
    quality_model_score_step = PythonScriptStep(
        name='quality_model_score',
        script_name="quality_model_score.py",
        compute_target=aml_compute,
        runconfig=pipeline_run_config,
        source_directory=score_source_dir)
        #allow_reuse=False)

    # Create sequence of steps
    quality_model_score_step_sequence = StepSequence(steps = [quality_model_score_step])

    # Create pipeline
    quality_model_score_pipeline = Pipeline(workspace=aml_workspace, steps=quality_model_score_step_sequence)
    quality_model_score_pipeline.validate()

    quality_model_score_pipeline_run = experiment.submit(quality_model_score_pipeline,regenerate_outputs=True)

    # RunDetails(pipeline_run).show()
    quality_model_score_pipeline_run.wait_for_completion()

    # Publish pipeline to AzureML
    quality_model_score_published_pipeline = quality_model_score_pipeline.publish('quality-model-score-pipeline')

    return