import azureml.core
import os
from azureml.core import Workspace, Datastore, Dataset, ComputeTarget, Experiment, ScriptRunConfig, Environment, Model
from azureml.core.run import Run
from azureml.core.runconfig import RunConfiguration
from azureml.pipeline.steps import PythonScriptStep
from azureml.pipeline.core import Pipeline, PipelineData, StepSequence, PublishedPipeline
from ml_service.util.env_variables import Env
from ml_service.util.manage_environment import get_environment
from ml_service.pipelines.test_train import test_train


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
print("DEVOPS: Starting AUTH BLOCK")
#from azureml.core.authentication import MsiAuthentication
#aml_workspace._auth=MsiAuthentication()
print(aml_workspace._auth)
print(aml_workspace._auth_object)
print("DEVOPS: AzureML MI credentials retrieved")

from azure.identity import ManagedIdentityCredential
print("DEVOPS: Now trying to get the blob")
az_cred_blob=ManagedIdentityCredential()
print("DEVOPS: Now set a credential, we hope")

#try:
#    import azure.storage.blob
#    print("DevOps: Storage blob libs installed")
#except Exception as e:
#    print("DevOps: storage blob libs not installed")
#    print("Exception: {}".format(e))
#try:
#    from azure.storage.blob import BlobServiceClient
#except Exception as e:
#    print("DevOps: storage blob service client not installed")
#    print("Exception: {}".format(e))#

#try:
#    from azure.storage.blob import ContainerClient
#    print("DevOps: container client installed")
#except:
#    print("DevOps: container client not installed")
#    pass


#url="https://{}.blob.core.windows.net".format(os.environ.get('DATA_STORAGE_ACCOUNT_NAME'))
#containername=os.environ.get("DATA_STORAGE_CONTAINER_NAME")
#blobservice=BlobServiceClient(url,credential=az_cred_blob)
#container_client=ContainerClient(account_url=url,credential=az_cred_blob,container_name=containername)
#print("DevOps: Got blob clients")
#list_blobs=container_client.list_blobs()
#print("DevOps: Got list of blobs: {} items".format(len(list_blobs)))
#for blob in list_blobs:
#    print(blob.name)
#    current_file_dir=os.path.dirname(os.path.abspath(__file__))
#    basepath=current_file_dir+"../../employer_engagement/training/ML_Models/Download_Manifest/"
#    fullpath=basepath+"/"+blob.name
#    if("/" in blob.name): # virtual directory - need to make it on local cloud disk location
#        fname=blob.name.split("/")[-1]
#        parentpath=basepath+blob.name.replace(fname,"")
#        try:
#            os.makedirs(parentpath)
#        except:
#            pass
#    try:
#        blob_client=blobservice.get_blob_client(container=blob.container,blob=blob.name)
#        strm=blob_client.download_blob()
#        with open(fullpath,'wb+') as wf:
#            wf.write(strm.readall())

#        print("written to file")
#    except Exception as e:
#        print(f"Exception: {e}")

    




# Create experiment if it doesn't exist
experiment = Experiment(aml_workspace, "employer-engagement")

# Add compute information to variable
aml_compute = aml_workspace.compute_targets["cpucluster"]

# Set up experiment folder
experiment_folder = 'employer_engagement'
 
# Create a Python environment for the experiment (from a .yml file)
environment = Environment.from_conda_specification("environment", experiment_folder + "/conda_dependencies.yml")
# Register the environment 
environment.register(workspace=aml_workspace)
registered_env = Environment.get(aml_workspace, 'environment')
# Create a new runconfig object for the pipeline
pipeline_run_config = RunConfiguration()
# Use the compute you created above. 
pipeline_run_config.amlcompute=aml_compute
pipeline_run_config.target = aml_compute
# Assign the environment to the run configuration
pipeline_run_config.environment = registered_env

#Create pipelines for levy models
test_train(aml_workspace, aml_compute, pipeline_run_config, experiment) 

