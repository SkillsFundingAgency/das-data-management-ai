import pickle
import os
import glob
import pandas as pd
import numpy as np
import pyarrow.parquet as pq
import joblib
import datetime
import shap
import math
import xgboost as xgb
import matplotlib.pyplot as plt
from azureml.core.run import Run
from azureml.core import Dataset, Datastore, Model
from azureml.data.datapath import DataPath
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn import datasets, linear_model
from scipy import stats
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

import levy_train_sql_functions as levy_train_functions
import generic_sql_functions as generic_train_functions
import sql_interface 
import pickle 
import time
# Set up config of workspace and datastore
aml_workspace = Run.get_context().experiment.workspace
#datastore = Datastore.get(aml_workspace, datastore_name='datamgmtdb')
run = Run.get_context()
#run.log('levy_model_train','levy_model_train')

#prevent SettingWithCopyWarning message from appearing
#pd.options.mode.chained_assignment = None
os.environ['CUDA_VISIBLE_DEVICES']='-1'

### Get XGBoost version -pkl load might be a bit funky if XGB is different to the model saved
try:
    run.log("INFO 1","INFO: XGB VERSION {}".format(xgb.__version__))
except:
    pass

try:
    models=Model.list(aml_workspace)
    run.log("INFO 2","INFO: Number of models registered: {}".format(len(models)))
    run.log("INFO 3","INFO : List of models: {}".format(str([x.name for x in models])))
    run.log("INFO 4","INFO:  MODEL PATHS: {}".format([Model.get_model_path(x.name) for x in models]))
    #Get latest model
    #model=models[-1]
    #model.download(target_dir=".",exist_ok=False) # We don't need this model
    #run.log("INFO 5", "MODEL DOWNLOADED SUCCESSFULLY")
except Exception as e:
    run.log("EXCEPTION 1","MODEL REGISTRY ERROR: {}".format(e))
    



try:
    run.log("INFO 6","BLOB DOWNLOAD  CHECK")
    run.log("INFO 7", "BLOB AUTH: {}".format(aml_workspace._auth))
    blob=Datastore.get(aml_workspace,'workspaceblobstore')
    
    dataset = Dataset.File.from_files((blob, 'Dummy_AE/'))
    file_paths = dataset.to_path()
    ctr=0
    for path in file_paths:
        print(path)
        run.log("INFO 8 A{}",format(ctr),'FILES OBTAINED: {}'.format(str(path)))
        ctr+=1
    dataset = Dataset.File.from_files((blob, 'ONSData/'))
    file_paths = dataset.to_path()   
    ctr=0
    for path in file_paths:
        print(path)
        run.log("INFO 8 B{}".format(ctr),'FILES OBTAINED: {}'.format(str(path)))
        ctr+=1
except Exception as E:
    run.log("EXCEPTION 2",f'Exception: {E}')

try:
    df_in_AIMod,validation_report=sql_interface.ExtractView()
    run.log('INFO 9","Queried Datamart OK')
    run.log("INFO 10","Extraction verification report: {}".format(str(validation_report)))
except Exception as E:
    run.log("EXCEPTION 3",f'Exception: {E}')
    pass


# Make some directories on the AzureML build machines
try:
    os.makedirs("./ML_Models/Download_Manifest/Dummy_Autoencoder/")
except Exception as e:
    run.log("EXCEPTION 4",f'{e}')
    pass

try:
    os.makedirs("./ML_Models/Download_Manifest/ONSData/")
except Exception as e:
    run.log("EXCEPTION 5",f'{e}')
    pass
    
try:
    run.log('INFO 11','Now attempting real download of the autoencoder')
    blob=Datastore.get(aml_workspace,'workspaceblobstore')
    dataset = Dataset.File.from_files((blob, 'Dummy_AE/'))
    run.log('INFO 12A',str(blob))
    
    dataset.download(target_path="./ML_Models/Download_Manifest/Dummy_Autoencoder/",overwrite=True)
    ld=glob.glob("./ML_Models/Download_Manifest/Dummy_Autoencoder/*")
    run.log('INFO 13',f'List of Files: {str(ld)}')
    run.log('INFO 13 A-1',f'BLOBNAME {str(blob.name)}')
    run.log('INFO 13 A-2',f'BLOBKEY {str(blob.account_name)}')
    run.log('INFO 13 A-3',f'BLOBKEY {str(blob.container_name)}')
    run.log('INFO 13 A-4',f'BLOB CLIENT ID: {str(blob.client_id)}')
            
except Exception as E:
    run.log('EXCEPTION 6',f'{E}')

run.log("INFO 14 A0","Now trying the download of the autoencoder files via a loop of files")
try:
    blob=Datastore.get(aml_workspace,'workspaceblobstore')    
    dataset = Dataset.File.from_files((blob, 'Dummy_AE/'))
    file_paths = dataset.to_path()
    download_path="./ML_Models/Download_Manifest/Dummy_Autoencoder/"
    ctr=1
    for path in file_paths:
        print(path)
        run.log("INFO 14 A{}".format(ctr),'FILES OBTAINED: {}'.format(str(path)))
        individual_dataset=Dataset.File.from_files((blob,"Dummy_AE"+path))
        individual_dataset.download(target_path=download_path,overwrite=True,ignore_not_found=True)
        run.log("INFO 14 B{}".format(ctr),'FILES OBTAINED: {}'.format(str(path)))
        ctr+=1
    ld=glob.glob(download_path+"*")
    ld_hidden=glob.glob(download_path+".*")
    run.log('INFO 14 C','List of files: {}, hidden files: {}'.format(ld,ld_hidden))
except Exception as E:
    run.log('EXCEPTION 7',f'{E}')


try:
    run.log('INFO 15','Now attempting download of the ONS data')
    blob=Datastore.get(aml_workspace,'workspaceblobstore')
    dataset = Dataset.File.from_files((blob, 'ONSData/'))
    
    dataset.download(target_path="./ML_Models/Download_Manifest/ONSData/",overwrite=True,ignore_not_found=True)
    ld=glob.glob("./ML_Models/Download_Manifest/ONSData/*")
    run.log('INFO 16', f'List of Files: {str(ld)}')


    run.log('INFO 16 A-1',f'BLOBNAME {str(blob.name)}')
    run.log('INFO 16 A-2',f'BLOBKEY {str(blob.account_name)}')
    run.log('INFO 16 A-3',f'BLOBKEY {str(blob.container_name)}')
    run.log('INFO 16 A-4',f'BLOB CLIENT ID: {str(blob.client_id)}')
except Exception as E:
    run.log('EXCEPTION 8',f'{E}')


run.log("INFO 17 A0","Now trying the download of the ONS files via a loop of files")
try:
    blob=Datastore.get(aml_workspace,'workspaceblobstore')    
    dataset = Dataset.File.from_files((blob, 'ONSData/'))
    file_paths = dataset.to_path()
    download_path="./ML_Models/Download_Manifest/ONSData/"
    ctr=1
    for path in file_paths:
        print(path)
        run.log("INFO 17 A{}".format(ctr),'FILES OBTAINED: {}'.format(str(path)))
        individual_dataset=Dataset.File.from_files((blob,'ONSData'+path))
        individual_dataset.download(target_path=download_path,overwrite=True)
        run.log("INFO 17 B{}".format(ctr),'FILES OBTAINED: {}'.format(str(path)))
        ctr+=1
    ld=glob.glob(download_path+"*")
    run.log('INFO 17 C','List of files: {}'.format(ld))
except Exception as E:
    run.log('EXCEPTION 9',f'{E}')


# MOVE FILES INTO APPROPRIATE DIRECTORIES
try:
    os.makedirs("./ML_Models/Models/Dummy_AE/")
except:
    pass
try:
    os.makedirs("./ML_Models/ONSData")
except:
    pass

try:
    os.system('cp -r -a -v ./ML_Models/Download_Manifest/ONSData/* ./ML_Models/ONSData/')
    run.log('JOB COPY PROCESS 0','ONS data copied')    
except:
    pass    

try:
    print("TRYING TO MOVE FILES TO THE AUTOENCODER DIRECTORY")
    os.system("ls -ltra ./ML_Models/Download_Manifest/Dummy_Autoencoder/")
    
    os.system('cp -r -a -v ./ML_Models/Download_Manifest/Dummy_Autoencoder/* ./ML_Models/Models/Dummy_AE/')
    print("Autoencoder: COPIED FILES")
    os.system('cp -r -a -v ./ML_Models/Download_Manifest/Dummy_Autoencoder/.* ./ML_Models/Models/Dummy_AE/') # additional block to move hidden files
    
    print("ASSIGNING ADDITIONAL PERMS TO AE BINARIES")
    os.system("rm -r -v ./ML_Models/Models/Dummy_AE/ONSData") # for some reason the ONSData is being downloaded into the Autoencoder directory - remove it
    os.system('chmod 755 ./ML_Models/Models/Dummy_AE/* ') # permissions on the Autoencoder binaries
    os.system('chmod 755 ./ML_Models/Models/Dummy_AE/.* ') # permissions on the autoencoder binaries' hidden files
    os.system('chmod 755 -R ./ML_Models/Models/Dummy_AE') # permissions on everything in this directory
    print("Autoencoder: COPIED HIDDEN FILES")
    print("GETTING LIST OF FILES IN AE DIR")
    os.system("ls -ltra ./ML_Models/Models/Dummy_AE/")
    print("\n\n")
except:
    pass
# temp download of fake dataset (CSV)
try:
    os.system('cp -r -a -v ./ML_Models/Download_Manifest/ONSData/Fake_Dataframe_SQLOutput.csv  ./ML_Models/')
except:
    pass

try:
    os.system('cp -r -a -v ./ML_Models/Download_Manifest/ONSData/ScalerSetup*.json ./ML_Models/Models/')
except:
    pass
print("NOW COPYING THE BDT STEP FILES - CURRENTLY STORED IN ONS DATA DIRECTORY")
try:
    os.system('mkdir ./ML_Models/Models/BDTStepConfig/')
except:
    pass
try:    
    os.system('cp -r -a -v ./ML_Models/Download_Manifest/ONSData/Model_BDT*.model ./ML_Models/Models/BDTStepConfig/')
    os.system('cp -r -a -v ./ML_Models/Download_Manifest/ONSData/PCA*.pkl ./ML_Models/Models/BDTStepConfig/')
    print("CONFIRMING BDT MODELS DOWNLOADED IN MANIFEST")
    ldbdt=glob.glob("./ML_Models/Models/BDTStepConfig/*.*")
    
    print("BDT CONFIG FILES CHECK: {}".format(ldbdt))
    print("BDT CHECK COMPLETE")
except:
    pass
run.log('JOB START INFO 0',"JOB START")

df_in=pd.DataFrame()
try:
    df_in=pd.read_csv('./ML_Models/Fake_Dataframe_SQLOutput.csv',index_col=0)
except Exception as E:
    # major exception
    run.log('DATA LOAD EXECUTION ERROR: ',f'{str(E)}')
df_autoencoded=None
try:
    import DataPreprocessing_Step
    df_out=DataPreprocessing_Step.Preprocess_Data(df_in,run)
    df_autoencoded=DataPreprocessing_Step.AE_CPIH_STEP(df_out,run)
    
    print("AUTOENCODED")
except Exception as E:
    run.log('DATA PREPROCESS EXECUTION ERROR: ',f'{str(E)}')
    print("PREPROCESSING ERROR: {}".format(E))

try:
    import Generate_BDT_Predictions_NOLocation as BDTCode
    currtime=datetime.datetime.now()
    hh=currtime.hour
    mm=currtime.minute
    DD=currtime.day
    MM=currtime.month
    YYYY=currtime.year
    fname_AImod=f'WithdrawalRateAIPrediction_{YYYY}{MM}{DD}{hh}{mm}00'
    BDTCode.RunBDTModel(infile="./ML_Models/Fake_Dataframe_SQLOutput.csv",plots=False,outfile="./outputs/{}.csv".format(fname_AImod),PandasInput=df_autoencoded.copy())
    run.log('BDT EVAL','OK')
except Exception as E:
    print("BDT EVALUATION FAILURE")
    print("EXCEPTION {}".format(E))

print("PREPROCESSING COMPLETE")
run.log("INFO19","JOB FINISH STATUS OK")
print("*****************************")
print("END OF JOB")
print("METRICS:")
#print(run.get_metrics())
metrics=run.get_metrics()
for key in metrics.keys():
    print(key,metrics[key])
print("***************************")
print("PRESENTING LOG:")
#print(run.get_all_logs())


#try:
#    # Create model build data into dataframe
#    # Create df with all accounts and early adopter flag
   
run.log('Success 01','Finished Job')
