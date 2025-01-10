from azureml.core import Workspace
from azureml.core.compute import ComputeTarget
from azureml.pipeline.steps import PythonScriptStep
from azureml.pipeline.core import Pipeline, PipelineData, StepSequence, PublishedPipeline
from azureml.core.runconfig import RunConfiguration
from azureml.pipeline.core import PipelineEndpoint
import azureml.core
import os
import pandas as pd
from azureml.data.datapath import DataPath
from azureml.core import Workspace, Datastore, Dataset, ComputeTarget, Experiment, ScriptRunConfig, Environment, Model
from azureml.core.run import Run

# Set up config of workspace and datastore

aml_workspace = Run.get_context().experiment.workspace
datastore = Datastore.get(aml_workspace, datastore_name='datamgmtdb')

def TestQuery(top_x: str) :
    query_test = DataPath(datastore,"SELECT TOP({}) * FROM ASData_PL.Comt_Apprenticeship".format(top_x))
    tabular_test = Dataset.Tabular.from_sql_query(query_test, query_timeout=3600)
    test_df = tabular_test.to_pandas_dataframe()
    
    return test_df


def Validate_Columns(df_in):
    #Expect number of columns to be validated explicitly
    print("VALIDATION REPORT")
    lencheck=False
    if(len(df_in.columns==39)):
        print("SQL dataframe has correct number of rows")
        lencheck=True
    if((len(df_in.columns)!=39) & (len(df_in.columns)!=35): # flag only if the table is incorrect somehow.
        print("WARNING: SQL dataframe has {} columns, the downstream might spectacularly explode".format(len(df_in.columns)))

    
    expected_column_names=['CommitmentId', 'Id', 'CreatedOn', 'ULN', 'StandardUId', 'StartDate', 'EndDate', 'StopDate', 'LearnStartDate', 'PlannedEndDate', 'LearnActEndDate', 'UKPRN', 'DelLoc_Pst_Lower_Layer_SOA', 'CompletionStatus', 'IsTransfer', 'StandardCode', 'Level', 'SectorSubjectAreaTier1', 'SectorSubjectAreaTier2', 'SectorSubjectAreaTier2_Desc', 'SectorSubjectAreaTier1_Desc',  'LARSCODE', 'FLAG_AGGREGATED_LOWRATING', 'weighted_average_annual_minwage', 'weighted_average_annual_maxwage', 'ProviderUkprn', 'FrameworkOrStandardLarsCode', 'EmployerAccountId', 'Employer type', 'Employer sector estimate', 'Employee_size_estimate', 'CURR_STAMP', 'YESTERDAY', 'LASTWEEK', 'CreatedRecordDate']
    missing_columns=[]
    for col in expected_column_names:
        if(not any(x==col for x in df_in.columns)):
            print("ERROR: DATAFRAME IS MISSING: {}".format(col))
    validation_report={'Length OK?':lencheck,'Missing Columns':missing_columns}
    return validation_report
    

def ExtractView(environment=""):
    if(environment=="" or environment==None):
        print("EXTRACTVIEW: NO ENVIRONMENT SPECIFIED")
        return pd.DataFrame(),{}
    print("Extract View: Environment:{}")
    if(environment=="PP" or environment=="PROD"):
        
        query=DataPath(datastore,'SELECT * FROM [PDS_AI].[MLOpsData]')
    elif(environment=="AT" or environment=="TEST"):
        query=DataPath(datastore,'SELECT * FROM [PDS_AI].[Dev_MLOpsData]')
    else:
        print("EXTRACTVIEW ERROR: ENVIRONMENT {} SPECIFIED BUT NOT IN LIST".format(environment))
        return pd.DataFrame(),{}

    tabular=Dataset.Tabular.from_sql_query(query,query_timeout=3600)
    outdf=tabular.to_pandas_dataframe()
    validation_report=Validate_Columns(outdf)
    #validation_report="" # force a temporary validation report to skip
    return outdf,validation_report



