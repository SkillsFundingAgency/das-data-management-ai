from azureml.core import Workspace
from azureml.core.compute import AmlCompute
from azureml.core.compute import ComputeTarget
from azureml.exceptions import ComputeTargetException
from ml_service.util.env_variables import Env
from azureml.pipeline.steps import PythonScriptStep
from azureml.pipeline.core import Pipeline, PipelineData, StepSequence, PublishedPipeline
from azureml.core.runconfig import RunConfiguration

def get_quality_train() :
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
