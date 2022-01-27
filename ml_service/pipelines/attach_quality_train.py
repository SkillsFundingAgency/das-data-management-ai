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


def get_quality_train(aml_workspace: Workspace, aml_compute: str, pipeline_run_config: str, experiment: str) :
    # quality_model data prep step
    train_source_dir="./employer-engagement/training"
    quality_model_data_prep_step = PythonScriptStep(
        name='quality_model_data_prep',
        script_name="quality_model_data_prep.py",
        compute_target=aml_compute,
        runconfig=pipeline_run_config,
        source_directory=train_source_dir)

#    # quality model train step
#    quality_model_train_step = PythonScriptStep(
#        name='quality_model_train',
#        script_name="quality_model_train.py",
#        compute_target=aml_compute,
#        runconfig=pipeline_run_config,
#        source_directory=train_source_dir)

#    # quality model accuracy and register step
#    quality_model_accuracy_register_step = PythonScriptStep(
#        name='quality_model_accuracy_register',
#        script_name="quality_model_accuracy_register.py",
#        compute_target=aml_compute,
#        runconfig=pipeline_run_config,
#        source_directory=train_source_dir)

    # Create sequence of steps for model train
#    quality_model_train_step_sequence = StepSequence(steps = [quality_model_data_prep_step, quality_model_train_step, quality_model_accuracy_register_step])
    quality_model_train_step_sequence = StepSequence(steps = [quality_model_data_prep_step, quality_model_train_step])

    # Create pipeline
    quality_model_train_pipeline = Pipeline(workspace=aml_workspace, steps=quality_model_train_step_sequence)
    quality_model_train_pipeline.validate()

    quality_model_train_pipeline_run = experiment.submit(quality_model_train_pipeline,regenerate_outputs=True)

    # RunDetails(pipeline_run).show()
    quality_model_train_pipeline_run.wait_for_completion()

    # Publish pipeline to AzureML
    quality_model_train_published_pipeline = quality_model_train_pipeline.publish('quality-model-train-pipeline')
    
    return
