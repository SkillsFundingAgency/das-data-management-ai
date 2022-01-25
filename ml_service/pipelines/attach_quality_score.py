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

    score_source_dir="./employer-engagement/scoring"
    score_step = PythonScriptStep(
        name='scoring',
        script_name="score.py",
        compute_target=aml_compute,
        runconfig=pipeline_run_config,
        source_directory=score_source_dir,
        allow_reuse=False)

    # Create sequence of steps
    score_step_sequence = StepSequence(steps = [score_step])

    # Create pipeline
    score_pipeline = Pipeline(workspace=aml_workspace, steps=score_step_sequence)
    score_pipeline.validate()

    score_pipeline_run = experiment.submit(score_pipeline,regenerate_outputs=True)

    # RunDetails(pipeline_run).show()
    score_pipeline_run.wait_for_completion()

    # Publish pipeline to AzureML
    score_published_pipeline = score_pipeline.publish('scoring-pipeline')

    return