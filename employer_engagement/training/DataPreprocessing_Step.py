import pandas as pd
import numpy as np

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
            logging.basicConfig(level=logging.INFO)
        else:
            self.run=run
        self.logfunction=logstep
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
            elif("debug" in logtype.lower()):
                self.logger.debug(logstring)
            else:
                self.logger.info(logstring)
        self.logctr+=1



    

def Preprocess_Data(df_in=pd.DataFrame(),run=None) : 
    """
    STAGE 1 for the preprocessing: Taking the dataframe from Az SQL, doing the first set of JOINs to ONS data so we have all the data in one super table.
    """
    ###master block
    df_out=df_in.copy()

    isAzure=False
    logger=None
    if(run==None):
        try:
            aml_workspace = Run.get_context().experiment.workspace
            #datastore = Datastore.get(aml_workspace, datastore_name='datamgmtdb')
            run = Run.get_context()
            isAzure=True
        except Exception as e:
            print("No AML workspace detected - now using logger logs")  
            print("AML ERROR: {}".format(e))
            pass      
    else:
        isAzure=True
    logger=ErrorHandler(isAzure,logstep="Preprocessing",run=run)

    logger.log('INFO',"Hello there")
    
    

    df_econ=pd.read_csv("./ML_Models/ONSData/calculated_features.csv")
    logger.log('INFO', 'Loaded economic data from the ONS')
    economic_relevant_fields=[
        'date',
        'gdp',
        'agriculture_gdp',
        'construction_gdp',
        'production_gdp',
        'services_gdp',
        'production',
        'cpih_inflation',
        'house_price_index',
        'household_spending',
        'household_income',
        'unemployment',
        'cci',
        'bci',
        'cli',
        'undergraduate_students',
        'postgraduate_students',
        'hpi_england',
        'avg_house_price_england',
        'median_income_england',
        'median_salary_all',
        'median_salary_16-17',
        'median_salary_18-21',
        'median_salary_22-29',
        'median_salary_30-39',
        'median_salary_40-49',
        'median_salary_50-59'
        ]
    df_econ=df_econ[economic_relevant_fields]
    df_econ['date']=pd.to_datetime(df_econ['date'])

    # %%
    print(df_in.columns)

    df_as_bq_social_nonull=df_in[df_in['StartDate'].notna()].copy()

    print(list(df_as_bq_social_nonull.columns))
    for c in df_as_bq_social_nonull:
        if(c=="EndDate" or c=="ActualStartDate"):
            print(c)
            df_as_bq_social_nonull[c+"_corr"]=pd.to_datetime(df_as_bq_social_nonull[c],errors='coerce')
        if(c=="StartDate"):
            print("*{}*".format(c))
            print(df_as_bq_social_nonull)
            df_as_bq_social_nonull[c+"_corr"]=pd.to_datetime(df_as_bq_social_nonull[c],errors='coerce')


    covid_timestamp=pd.Timestamp('2020-03-23')
    df_as_bq_social_nonull['CovidRef']=covid_timestamp
    df_as_bq_social_nonull['unixtimediff_exp_end']=(df_as_bq_social_nonull['EndDate_corr'].dt.to_period('M').astype('int64') - df_as_bq_social_nonull['CovidRef'].dt.to_period('M').astype('int64'))

    df_as_bq_social_nonull['unixtimediff_start']=(df_as_bq_social_nonull['StartDate_corr'].dt.to_period('M').astype('int64') - df_as_bq_social_nonull['CovidRef'].dt.to_period('M').astype('int64'))

    #logger.log('RUN')
    from pandas.tseries.offsets import MonthBegin
    def MonthShift(timedf,field_to_shift='StartDate_corr'):
        # function to shift the date to the nearest 1st of the month, so I can join to the Economic data.
        cpydf=timedf.copy(deep=True)
        
        cpydf[field_to_shift+" (1st of Month)"]=cpydf[field_to_shift]+MonthBegin(1)
        #cpydf['TimeDiff']=abs((cpydf[field_to_shift]-cpydf[field_to_shift+" (1st of Month)"]).dt.days)

        #logging.debug("Diff")
        #logging.debug(cpydf[(cpydf[field_to_shift]<pd.Timestamp("01-01-2023"))*(cpydf['TimeDiff']>5)].head(40))
                            
        return cpydf

    df_as_bq_social_nonull=MonthShift(df_as_bq_social_nonull,field_to_shift='StartDate_corr')
    
    

    df_as_bq_social_nonull['06MoBeforeDate']=pd.to_datetime(df_as_bq_social_nonull['StartDate_corr (1st of Month)']-pd.DateOffset(months=6))
    df_as_bq_social_nonull['12MoBeforeDate']=pd.to_datetime(df_as_bq_social_nonull['StartDate_corr (1st of Month)']-pd.DateOffset(months=12))
    df_as_bq_social_nonull["18MoBeforeDate"]=pd.to_datetime(df_as_bq_social_nonull['StartDate_corr (1st of Month)']-pd.DateOffset(months=18))
    df_as_bq_social_nonull['24MoBeforeDate']=pd.to_datetime(df_as_bq_social_nonull['StartDate_corr (1st of Month)']-pd.DateOffset(months=24))

    logger.log("INFO","Now completed Time transforms")

    def DoBigEconomicsMerge(df_AS,df_EconomicsData):
        time_fields_economicdata='date'
        time_fields_ILR=['06MoBeforeDate','12MoBeforeDate','18MoBeforeDate','24MoBeforeDate']
                        #'_06MoAfterDate','_12MoAfterDate','_18MoAfterDate','_24MoAfterDate','_36MoAfterDate','_48MoAfterDate',
                        #'onequarterdate_after','halfwaydate_after','threequarterdate_after'
                        #]
        df_ILR_cpy=df_AS.copy()
        
        econ_cols=[]
        for tf in time_fields_ILR:
            #rename all the cols to have the stamp on them
            df_economics_tmp=df_EconomicsData.copy() 
            df_economics_tmp.columns=[x+"_"+tf for x in df_economics_tmp.columns]
            datefield=time_fields_economicdata+"_"+tf
            df_ILR_cpy=pd.merge(df_ILR_cpy,df_economics_tmp,how="left",left_on=[tf],right_on=[datefield])

            if("Before" in datefield):
                cols=list(df_economics_tmp.columns)
                cols.remove(datefield)
                econ_cols=econ_cols+cols
            

        return df_ILR_cpy,econ_cols   
    df_FullEconVariables,econ_cols_before=DoBigEconomicsMerge(df_as_bq_social_nonull,df_econ)
    logger.log("INFO","Now completed merge with economic data")
    
    #logger.log("DEBUG",df_FullEconVariables.head(30))


    # %%
    # for c in df_FullEconVariables.columns:
    #     if "." in c:
    #         print(c)
    #         try:
    #             nbins=5
    #             col_red=c.replace(".1","")

    #             binmax=df_FullEconVariables[col_red].max()
    #             binmin=df_FullEconVariables[col_red].min()
    #             binwidth=(binmax-binmin)/nbins
    #             binspec=np.arange(binmin,binmax+binwidth,binwidth)
    #             plt.close()
    #             plt.hist(df_FullEconVariables[c],histtype='step',label='DUPVAR',bins=binspec)
    #             plt.hist(df_FullEconVariables[col_red],histtype='step',label='BASEVAR',bins=binspec)
    #             plt.legend()
    #             plt.xlabel(col_red)
    #             plt.ylabel('FREQ')
    #             plt.show()
    #         except Exception as E:
    #             pass
    df_FullEconVariables_Reduced=df_FullEconVariables[[x for x in df_FullEconVariables if "." not in x]].copy()


    # %% [markdown]
    # ## ADD THE IOD 2019 RESULTS
    # 

    # %%
    df_social=pd.read_csv("./ML_Models/ONSData/iod_2019.csv")
    #filter out duplicate indices (only keep scores)

    key='LSOA code (2011)'
    list_reduced=[key]


    for x in df_social.columns:
        if("Score" in x):
            list_reduced.append(x)
    df_social=df_social[list_reduced]
    

    df_fullvars=pd.merge(left=df_FullEconVariables_Reduced,
                        right=df_social,
                        left_on=['DelLoc_Pst_Lower_Layer_SOA'],
                        right_on=['LSOA code (2011)'],
                        how='left'
                        )
    logger.log("INFO","Completed LSOA link")
    

    # %%
    df_out=df_fullvars.copy(deep=True)
    #for p in list(df_out.columns):
    #    logger.log('INFO',p)

    logger.log("INFO",'End of Data Preproc')
    print("FINISHED PREPROC")
    return df_out

def AE_CPIH_STEP(df_in,run=None):
    ## STAGE 2 OF THE CODE:
    #IMPUTE MISSING VARIABLES
  
    df_out=df_in.copy()


    isAzure=False
    logger=None
    if(run==None):
        try:
            aml_workspace = Run.get_context().experiment.workspace
            #datastore = Datastore.get(aml_workspace, datastore_name='datamgmtdb')
            run = Run.get_context()
            isAzure=True
        except Exception as e:
            print("No AML workspace detected - now using logger logs")  
            print("AML ERROR: {}".format(e))
            pass     
    else:# in case we handed a run object to it from the parent
        isAzure=True
    logger=ErrorHandler(isAzure,'Autoencoder_Step',run)
    logger.log('INFO','\n')
    logger.log('INFO','Hello from inside step')
    
    try:
        from DataPreprocessingFunctions import Process_AE 

    except Exception as E:
        logger.log('ERROR',"Autoencoder libraries don't work, this is probably an install problem with Python")
        logger.log('ERROR','Exception: {}'.format(E))
        logger.log("ERROR",'SKIPPING AUTOENCODER')
        print("AUTOENCODER IMPORT FAILURE: {}".format(E))
        return df_out
    logger.log('INFO','Autoencoder import OK')
    try:
        df_CPIH_AE=Process_AE.Process_AE_INPUT(df_in,False,-1,logger)
        df_out=df_CPIH_AE.copy()
    except Exception as e:
        logger.log('ERROR',"Autoencoder runtime doesn't work")
        logger.log(f"Exception: {e}")
        logger.log("ERROR",'SKIPPING AUTOENCODER')
        print("AE Runtime Error:{}".format(e))
        return df_out
    logger.log("INFO","Completed_Autoencoder step")
    return df_out

if __name__=="__main__":
    import argparse
    parser=argparse.ArgumentParser("DocumentProcessor")
    parser.add_argument("--i", dest='input',required=True,help="Input CSV file")
    parser.add_argument("--o",dest='stgoutput',required=True,help='OUTPUT CSV (without imputation)')
    parser.add_argument("--f",dest='outf',required=False,default='',help='Final Autoencoder output')
    args=parser.parse_args()

    df_in=pd.read_csv(args.input,index_col=0)
    df_preproc=Preprocess_Data(df_in)
    cpy_proc=df_preproc.copy()
    cpy_proc.to_csv(args.stgoutput)
    print("REAL OUTPUT:{}".format(df_preproc.head(3)))

    df_AE=AE_CPIH_STEP(df_preproc)

    if(args.outf!=""):
        df_AE.to_csv(args.outf)
    print()

# %%
