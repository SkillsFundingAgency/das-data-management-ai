import azureml.core
import os
from azureml.core import Workspace, Datastore, Dataset, ComputeTarget, Experiment, ScriptRunConfig, Environment, Model
from azureml.core.run import Run
from azureml.core.runconfig import RunConfiguration
from azureml.pipeline.core import Pipeline, PipelineData, StepSequence, PublishedPipeline
from azureml.pipeline.steps import PythonScriptStep
from azureml.core import Workspace
from azureml.core.compute import ComputeTarget
from azureml.core.model import Model

from ml_service.util.env_variables import Env
from ml_service.util.manage_environment import get_environment


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

# Create experiment if it doesn't exist
experiment = Experiment(aml_workspace, "WithdrawalAI")

# Add compute information to variable
aml_compute = aml_workspace.compute_targets["cpucluster"]

# Set up experiment folder
experiment_folder = 'WithdrawalAI'
 
# Create a Python environment for the experiment (from a .yml file)
environment = Environment.from_conda_specification("environment", experiment_folder + "/conda_env_MVS.yml")
# Register the environment 
environment.register(workspace=aml_workspace)
registered_env = Environment.get(aml_workspace, 'environment')


#Register model - even though its inference we need to tell AzureML that there's an AI model involved otherwise things start crashing
Model.register(workspace=aml_workspace,
               model_path=experiment_folder +"/Inference/dummy_model.pkl",
               model_name="dummy_model",
               tags={'pretrained':'inception'},
               description='Withdrawal AI model'
               )
# Create a new runconfig object for the pipeline
pipeline_run_config = RunConfiguration()
# Use the compute you created above. 
pipeline_run_config.target = aml_compute
# Assign the environment to the run configuration
pipeline_run_config.environment = registered_env



print("**************************************")
print("START: Check if a model is registered correctly prior to job start")
models=Model.list(workspace=aml_workspace)
print("Number of models registered: {}".format(len(models)))
if(len(models)>0):
    modctr=0
    for mod in models:
        print("Model: {}".format(modctr))
        try:
            print(f"name :{mod.name}")
        except:
            pass
        try:
            print(f'model ID: {mod.id}')
        except:
            pass
        try:
            print(f'Path: {mod.serialize()}')
        except:
            pass
        print(mod)
        print("\n")
        modctr+=1
print("*************************")


def get_WithdrawalAIModel(aml_workspace: Workspace, aml_compute: str, pipeline_run_config: str, experiment: str) :
    # non_levy_model train step
    train_source_dir="./WithdrawalAI/Inference/" # path is a dud
    WithdrawalAIModel_step = PythonScriptStep(
        name='WithdrawalAIModel',
        script_name="testML.py",
        compute_target=aml_compute,
        runconfig=pipeline_run_config,
        source_directory=train_source_dir)


    # Create sequence of steps for model train
    WithdrawalAIModel_step_sequence = StepSequence(steps = [WithdrawalAIModel_step])

    # Create pipeline
    WithdrawalAIModel_pipeline = Pipeline(workspace=aml_workspace, steps=WithdrawalAIModel_step_sequence)
    WithdrawalAIModel_pipeline.validate()







    print("**************************************")
    print("Runtime: Check if a model is registered correctly prior to job start")
    models=Model.list(workspace=aml_workspace)
    print("Number of models registered: {}".format(len(models)))
    if(len(models)>0):
        modctr=0
        for mod in models:
            print("Model: {}".format(modctr))
            try:
                print(f"name :{mod.name}")
            except:
                pass
            try:
                print(f'model ID: {mod.id}')
            except:
                pass
            try:
                print(f'Path: {mod.serialize()}')
            except:
                pass
            print(mod)
            print("\n")
            modctr+=1
    ###the validate() method returns a list, so lets see what it spits out! Might be useful
    validation_obj=WithdrawalAIModel_pipeline.validate()
    print("validation object")
    print(validation_obj)
    print("*************************")



    ###the offending buggy line - this is where the code will crash
    WithdrawalAIModel_pipeline_run = experiment.submit(WithdrawalAIModel_step_sequence,regenerate_outputs=True)

    # RunDetails(pipeline_run).show()
    WithdrawalAIModel_pipeline_run.wait_for_completion()

    # Publish pipeline to AzureML
    WithdrawalAIModel_published_pipeline = WithdrawalAIModel_pipeline.publish('WithdrawalAI-model-train-pipeline')
    
    return


#Create pipelines for withdrawal rate model
get_WithdrawalAIModel(aml_workspace,aml_compute,pipeline_run_config,experiment)