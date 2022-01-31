import azureml.core
import os
from azureml.core import Workspace, Datastore, Dataset, ComputeTarget, Experiment, ScriptRunConfig, Environment, Model
from azureml.core.run import Run
from azureml.core.runconfig import RunConfiguration
from azureml.pipeline.steps import PythonScriptStep
from azureml.pipeline.core import Pipeline, PipelineData, StepSequence, PublishedPipeline
# from ml_service.util.attach_compute import get_compute
from ml_service.util.env_variables import Env
from ml_service.util.manage_environment import get_environment
import os
from ml_service.pipelines.attach_quality_train import get_quality_train
from ml_service.pipelines.attach_quality_score import get_quality_score

# Set up all environment details
e = Env()
subscription_id = os.environ.get("SUBSCRIPTION_ID")
workspace_name = os.environ.get("WORKSPACE_NAME")
resource_group = os.environ.get("RESOURCE_GROUP")

# Get Azure machine learning workspace
aml_workspace = Workspace.get(
name=e.workspace_name,
subscription_id=e.subscription_id,
resource_group=e.resource_group,
)
print("get_workspace:")
print(aml_workspace)

# Create experiment if it doesn't exist
experiment = Experiment(aml_workspace, "employer-engagement")

# Add compute information to variable
aml_compute = aml_workspace.compute_targets["cpucluster"]

# Set up experiment folder
experiment_folder = 'employer-engagement'

# Create a Python environment for the experiment (from a .yml file)
environment = Environment.from_conda_specification("environment", experiment_folder + "/conda_dependencies.yml")
# Register the environment 
environment.register(workspace=aml_workspace)
registered_env = Environment.get(aml_workspace, 'environment')
# Create a new runconfig object for the pipeline
pipeline_run_config = RunConfiguration()
# Use the compute you created above. 
pipeline_run_config.target = aml_compute
# Assign the environment to the run configuration
pipeline_run_config.environment = registered_env
print ("Run configuration created.")

############### Import Quality Train Model Pipeline ##################

# Use train step to build and model
# quality_train_source_dir="./employer-engagement/training"
# quality_train_step = PythonScriptStep(
#     name='quality_model_build',
#     script_name="train.py",
#     compute_target=aml_compute,
#     runconfig=pipeline_run_config,
#     source_directory=train_source_dir)

# Create sequence of steps for model train
# quality_train_step_sequence = StepSequence(steps = [quality_train_step])

# Create pipeline
# quality_train_pipeline = Pipeline(workspace=aml_workspace, steps=quality_train_step_sequence)
# quality_train_pipeline.validate()

# quality_train_pipeline_run = experiment.submit(quality_train_pipeline,regenerate_outputs=True)

# RunDetails(pipeline_run).show()
# quality_train_pipeline_run.wait_for_completion()

# Publish pipeline to AzureML
# quality_train_published_pipeline = quality_train_pipeline.publish('quality-train-pipeline')


############### Scoring Model ###################

# quality_score_source_dir="./employer-engagement/scoring"
# quality_score_step = PythonScriptStep(
#     name='quality_scoring',
#     script_name="score.py",
#     compute_target=aml_compute,
#     runconfig=pipeline_run_config,
#     source_directory=score_source_dir,
#     allow_reuse=False)



# Create sequence of steps for model score
# quality_score_step_sequence = StepSequence(steps = [quality_score_step])

# Create pipeline
# quality_score_pipeline = Pipeline(workspace=aml_workspace, steps=quality_score_step_sequence)
# quality_score_pipeline.validate()

# quality_score_pipeline_run = experiment.submit(quality_score_pipeline,regenerate_outputs=True)

# RunDetails(pipeline_run).show()
# quality_score_pipeline_run.wait_for_completion()

# Publish pipeline to AzureML
# quality_score_published_pipeline = quality_score_pipeline.publish('quality-scoring-pipeline')

# get_compute(aml_workspace, e.compute_name, e.vm_size)

#get_quality_train(aml_workspace, aml_compute, pipeline_run_config, experiment) # successful
#get_quality_score(aml_workspace, aml_compute, pipeline_run_config, experiment) # bad default data 

get_levy_train(aml_workspace, aml_compute, pipeline_run_config, experiment) 
#get_levy_score(aml_workspace, aml_compute, pipeline_run_config, experiment)

#get_non_levy_train(aml_workspace, aml_compute, pipeline_run_config, experiment)
#get_non_levy_score(aml_workspace, aml_compute, pipeline_run_config, experiment)
