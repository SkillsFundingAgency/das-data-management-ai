import pandas as pd

import pyodbc 
import sqlalchemy
import azure.identity

from azure.identity import DefaultAzureCredential


import argparse
import json


import json

import argparse
parser=argparse.ArgumentParser('Dummy_Routine.py')
parser.add_argument("--creds",action='store',dest='creds',help='Credentials (JSON) required to call Azure SQL',default="../../AUTH_TOKEN/AUTH.json")

import time
import datetime
currtime=datetime.datetime.now()
print(currtime)
DD=currtime.day
MM=currtime.month
YYYY=currtime.year

hh=currtime.hour
mm=currtime.minute

ts_formatted=f"WithdrawalRateAIOutput_{mm}_{hh}_{DD}_{MM}_{YYYY}.csv"

print(ts_formatted)

date=currtime.date()
ttime=currtime.time()


args=parser.parse_args()
#jf=open(args.creds)
#creds=json.load(jf)

#SERVER=creds['server']
#USERNAME=creds['user']

#DATABASE=creds['database']

#AUTHENTICATION=creds['auth']

from azureml.core.run import Run
from azureml.core import Dataset, Datastore, Model
from azureml.data.datapath import DataPath
from azureml.core.experiment import Experiment
from azureml.core.workspace import Workspace



driver="ODBC Driver 18 for SQL Server"

from azureml.core import Workspace
from azureml.core.compute import ComputeTarget
from azureml.pipeline.steps import PythonScriptStep
from azureml.pipeline.core import Pipeline, PipelineData, StepSequence, PublishedPipeline
from azureml.core.runconfig import RunConfiguration
from azureml.pipeline.core import PipelineEndpoint
import azureml.core
import os
from azureml.data.datapath import DataPath
from azureml.core import Workspace, Datastore, Dataset, ComputeTarget, Experiment, ScriptRunConfig, Environment, Model
from azureml.core.run import Run
 
# Set up config of workspace and datastore


aml_workspace = Run.get_context().experiment.workspace
datastore = Datastore.get(aml_workspace, datastore_name='datamgmtdb')
run=Run.get_context()
run.log("HELLO THERE, STARTING JOB")

#print(run)


def QueryDB(dummyvar=1000):
    df_pandas=pd.DataFrame()
    try:
     query=DataPath(datastore,"SELECT TOP({}) * FROM ASData_PL.Comt_Apprenticeship".format(1000)) # just to show how it works with the data
     tabular_obj=Dataset.Tabular.from_sql_query(query,query_timeout=3600)
     df_pandas=tabular_obj.to_pandas_dataframe()
    except Exception as e:
     run.log("Exception: {}".format(e))
    return df_pandas.copy(deep=True)

df_in=QueryDB()
run.log("Database queried successfully")
run.log("Number of rows: {}".format(len(df_in)))


df_BDT_ALLOC=df_in.sample(frac=0.5,replace=False,random_state=42)


df_sample_1=df_in.sample(frac=0.4,replace=False,random_state=42)
df_sample_not1=df_in.drop(index=df_sample_1.index)
df_sample_3=df_sample_not1.sample(frac=0.2,random_state=42)
df_sample_2=df_sample_not1.drop(index=df_sample_3.index)

df_out=df_in.copy(deep=True)
df_out['Predicted_Withdrawal']=0
df_out['Predicted_Withdrawal'].iloc[df_BDT_ALLOC.index]=1
df_out['EmailClassification']="DUD"
df_out['EmailClassification'].iloc[df_sample_1.index]="Email A"
df_out['EmailClassification'].iloc[df_sample_2.index]='Email B'
df_out['EmailClassification'].iloc[df_sample_3.index]="Early Years"
print(df_out['EmailClassification'].value_counts())

df_out['ApprenticeshipId']=df_out['Id']
df_out=df_out[['ApprenticeshipId','EmailClassification','Predicted_Withdrawal']]
run.log("Generated an output dataframe")
df_out.to_csv("./{}.csv".format(ts_formatted))

exit(0)
