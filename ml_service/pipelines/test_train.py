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

def test_train(aml_workspace: Workspace, aml_compute: str, pipeline_run_config: str, experiment: str) :
   
    train_source_dir="./employer_engagement/training"
    test_train_step = PythonScriptStep(
        name='test_train',
        script_name="test_train.py",
        compute_target=aml_compute,
        runconfig=pipeline_run_config,
        source_directory=train_source_dir,
        allow_reuse=False)


    # Create sequence of steps for model train
    test_train_sequence = StepSequence(steps = [test_train_step])

    # Create pipeline
    test_train_pipeline = Pipeline(workspace=aml_workspace, steps=test_train_sequence)
    test_train_pipeline.validate()


    # Publish pipeline to AzureML
    test_train_published_pipeline = test_train_pipeline.publish('test-train-pipeline')

    # create Pipeline Endpoint if not already exists , if exists add pipeline to the endpoint
    
    try:
          pipeline_endpoint = PipelineEndpoint.get(workspace=aml_workspace, name="test_train_endpoint")
          pipeline_endpoint.add_default(test_train_published_pipeline)
    except Exception:
          pipeline_endpoint = PipelineEndpoint.publish(workspace=aml_workspace,
                                                       name="test_train_endpoint", 
                                                       pipeline=test_train_published_pipeline,
                                                       description="Endpoint to test Train pipeline",
                                                       )
    try:
        print("pipeline_endpoint name={}".format(str(pipeline_endpoint.name)))
        print("pipeline endpoint ID: {}".format(str(pipeline_endpoint.id)))
        print("pipeline URL: {}".format(str(pipeline_endpoint.endpoint)))
    except Exception as e:
        print("Exception: {}".format(e))
        pass
    return
