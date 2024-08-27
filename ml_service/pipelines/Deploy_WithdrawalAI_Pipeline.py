# %%
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential#,ActiveDirectoryInteractive
from azure.ai.ml.entities import Data,Model

from azure.ai.ml.constants import AssetTypes

# authenticate
credential =DefaultAzureCredential()
import os
SUBSCRIPTION = os.environ.get('SUBSCRIPTION_ID')
RESOURCE_GROUP = os.environ.get('RESOURCE_GROUP')
WS_NAME = os.environ.get('WORKSPACE_NAME')
# Get a handle to the workspace
ml_client = MLClient(
    credential=credential,
    subscription_id=SUBSCRIPTION,
    resource_group_name=RESOURCE_GROUP,
    workspace_name=WS_NAME,
)


# %%
from azure.storage.blob import BlobServiceClient,ContainerClient

url_base="{}"
url_extra="./mlmodels/dummy_model.pkl" # dir on path on blob
full_url=f"{url_base}/mlmodels/dummy_model.pkl" #full dirpath

try:
    blobservice=BlobServiceClient(account_url=f"{url_base}",credential=credential) # GET HTTPS REQUEST

    account_url=f"{url_base}/"
    container_name="mlmodels/"
    full_url=f"{url_base}"
    container_client=ContainerClient(account_url=full_url,credential=credential,container_name=container_name)

    lblobs=container_client.list_blobs()
    for itm in lblobs:
        print("*******************************")
        print(itm)        
        try:
            blob_client=blobservice.get_blob_client(container=itm.container,blob=itm.name)
            strm=blob_client.download_blob()
            #if(".txt" in itm.name):
            #    print(strm.readall().decode())
            with open("./WithdrawalAI/Inference/"+"Prod_"+itm.name,'wb+') as f:
                f.write(strm.readall())
            
        except Exception as e:
            print("EXCEPTION: {}".format(e))
except Exception as P:
    print("blob download exception: {}".format(P))

# %%
if(os.path.exists('./WithdrawalAI/Inference/Prod_dummy_model.pkl')):
    model_path="./WithdrawalAI/Inference/Prod_dummy_model.pkl"
else:
    print("Using fallback model as cloud download probably failed")
    model_path="./WithdrawalAI/Inference/dummy_model.pkl"
v1='initial'
my_model=Model(
    path=model_path,
    name='dummymodel_TEST_PROD',
    type=AssetTypes.CUSTOM_MODEL,
    description='Model created from cloud path'
)
ml_client.models.create_or_update(my_model)

for itm in ml_client.models.list():
    print(itm)

# %%


# %%
from azure.ai.ml.entities import Environment
image="mcr.microsoft.com/mlops/python:latest"
pipeline_job_env=Environment(
    name="TESTENV",
    description="Test pipeline env",
    conda_file="./WithdrawalAI/dependencies/conda.yaml",
    image=image    
)

pipeline_job_env=ml_client.environments.create_or_update(pipeline_job_env)

# %%


# %%


# %%
from azure.ai.ml import MLClient
from azure.ai.ml.dsl import pipeline
from azure.ai.ml import load_component

# %%
test_infer_component=load_component(source="./WithdrawalAI/Inference/Comp_Inference.yml")


# %%

# Retrieve an already attached Azure Machine Learning Compute.

cpu_compute_target =  "cpu-cluster" #by default, but I called it sth different
print(ml_client.compute.get(cpu_compute_target))

# %%
@pipeline(
    default_compute=cpu_compute_target,
)
def TestAI():
    inf_node=test_infer_component()
    inf_node.compute=cpu_compute_target
pipeline_job=TestAI()

# %%
pipeline_job_out = ml_client.jobs.create_or_update(
    pipeline_job, experiment_name="pipeline_samples"
)


print(pipeline_job_out)