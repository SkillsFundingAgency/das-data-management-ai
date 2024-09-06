import pandas as pd
import numpy as np
global isAzure

try:
    from azureml.data.datapath import DataPath
    from azureml.core import Workspace, Datastore, Dataset, ComputeTarget, Experiment, ScriptRunConfig, Environment, Model
    from azureml.core.run import Run
except:
    print("No AML libs installed, can still run but will crash on AML")
    pass
import logging 
infoctr=0
errorctr=0
    
class ErrorHandler:
    """Dual Use Error Handler that can be used with/without AzureML Run in a transparent way once set"""
    isAzure=False # is AzureML in this case
    logfunction="Data Preprocessing" #
    logger=None
    logctr=0
    run=None
    def __init__(self,isAzure,logstep,run=None):
        self.isAzure=isAzure
        if(not isAzure):
            self.logger=logging.getLogger(logstep)
            logging.basicConfig(level=logging.DEBUG)
        else:
            self.run=run
        self.logctr=0
    def log(self,logtype="INFO",msg=""):
        metricstring=f'{logtype}:{self.logfunction} {self.logctr}:'
        if self.isAzure:
            
            # Since we can get the context repeatedly, no harm in getting it again
            run = Run.get_context()
            run.log(str(metricstring),str(msg))
        else:
            logstring=f'{metricstring} {msg}'
            if("error" in logtype.lower()):                
                self.logger.error(logstring)
            elif("warning" in logtype.lower()):
                self.logger.warning(logstring)
            else:
                self.logger.info(logstring)
        self.logctr+=1



    




def Preprocess_Data(df_in=pd.DataFrame()) : 
    ###master block
    df_out=df_in.copy()

    isAzure=False
    logger=None
    run=None
    try:
        aml_workspace = Run.get_context().experiment.workspace
        #datastore = Datastore.get(aml_workspace, datastore_name='datamgmtdb')
        run = Run.get_context()
        isAzure=True
    except Exception as e:
        print("No AML workspace detected - now using logger logs")  
        print("AML ERROR: {}".format(e))
        pass      
    logger=ErrorHandler(isAzure,logstep="Preprocessing",run=run)

    logger.log('INFO',"Hello there")
    
    
    

    return df_out

if __name__=="__main__":
    import argparse
    parser=argparse.ArgumentParser("DocumentProcessor")
    parser.add_argument("--i", dest='input',required=True,help="Input CSV file")
    args=parser.parse_args()

    
    NRows=10000
    v1=np.random.randint(low=0,high=500,size=NRows)
    v2=np.random.randint(low=0,high=5000,size=NRows)
    v3=np.random.randint(low=0,high=5100,size=NRows)
    df_t=pd.DataFrame({
        'v1':v1,
        'v2':v2,
        'v3':v3
    })

    df_out=Preprocess_Data(df_t)
    
    print("*********FINISHED SIMPLE TEST*************\n\n\n")

    df_in=pd.read_csv(args.input,index_col=0)
    df_out=Preprocess_Data(df_in)
    print("REAL OUTPUT:{}".format(df_out.columns))
