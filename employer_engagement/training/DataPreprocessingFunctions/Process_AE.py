# %%

import time




#import argparse 
#parser=argparse.ArgumentParser()
#parser.add_argument('--outf',action='store',dest='ofile',required=True,help="Output File (CSV)",type=str)
#parser.add_argument('--inf',action='store',dest='infile',required=True,help='Input File (CSV)',type=str)
#parser.add_argument("--AEOnly",action='store_true',default='False',help="Skip CPIH correction")
#parser.add_argument("--n",action='store',dest='n',help="NRows",default=-1,type=int)
#args=parser.parse_args()
#ofile=args.ofile
import sys
#sys.path.append(sys.path.__dir__)
import pandas as pd

from DataPreprocessingFunctions.ECON_CPIH_Correction import ProcessSalaryData

#ofile=args.ofile#"./CSV/INPUT_CSV_2205/IMPUTATION_OUTPUT_FULL_3105.csv"
#cpi_ofile=args.ofile.replace(".csv","_CPIH.csv")  #"./CSV/INPUT_CSV_2205/CPIH_CORR_3105.csv"
#outfile=ofile
#ofile_CPIHCorr=cpi_ofile

#"./CSV/INPUT_CSV_2205/IMPUTATION_INPUT_FULL_3105.csv"

def Process_AE_INPUT(df_in=pd.DataFrame(),aeonlyflag=False,nRows=-1,logger=None):

    # %%
    dstring="""
    Flag to disable AutoEncoder: {}
    nRows: {}
    logger_step: {}
    """.format(aeonlyflag,nRows,logger.logfunction)

    logger.log('INFO',dstring)

    import time
    start=time.time()
    logger.log('INFO',"**********")
    # RUN CPIH CORR

    if(not aeonlyflag):
        df_input=df_in.copy()
        #import DataPreprocessingFunctions.PCA_Processing as PCA

        logger.log("INFO","****** RUN PCA ON SOCIAL FIELDS")
        key='LSOA code (2011)'
        list_reduced=[key]

        df_social_columns=['LSOA code (2011)',
                            'LSOA name (2011)', 
                            'Local Authority District code (2019)', 
                            'Local Authority District name (2019)', 
                            'Index of Multiple Deprivation (IMD) Score', 
                            'Index of Multiple Deprivation (IMD) Rank (where 1 is most deprived)', 
                            'Index of Multiple Deprivation (IMD) Decile (where 1 is most deprived 10% of LSOAs)', 
                            'Income Score (rate)', 
                            'Income Rank (where 1 is most deprived)', 
                            'Income Decile (where 1 is most deprived 10% of LSOAs)', 
                            'Employment Score (rate)', 
                            'Employment Rank (where 1 is most deprived)', 
                            'Employment Decile (where 1 is most deprived 10% of LSOAs)', 
                            'Education, Skills and Training Score', 
                            'Education, Skills and Training Rank (where 1 is most deprived)', 
                            'Education, Skills and Training Decile (where 1 is most deprived 10% of LSOAs)', 'Health Deprivation and Disability Score', 'Health Deprivation and Disability Rank (where 1 is most deprived)', 'Health Deprivation and Disability Decile (where 1 is most deprived 10% of LSOAs)', 'Crime Score', 'Crime Rank (where 1 is most deprived)', 'Crime Decile (where 1 is most deprived 10% of LSOAs)', 'Barriers to Housing and Services Score', 'Barriers to Housing and Services Rank (where 1 is most deprived)', 'Barriers to Housing and Services Decile (where 1 is most deprived 10% of LSOAs)', 'Living Environment Score', 'Living Environment Rank (where 1 is most deprived)', 'Living Environment Decile (where 1 is most deprived 10% of LSOAs)', 'Income Deprivation Affecting Children Index (IDACI) Score (rate)', 'Income Deprivation Affecting Children Index (IDACI) Rank (where 1 is most deprived)', 'Income Deprivation Affecting Children Index (IDACI) Decile (where 1 is most deprived 10% of LSOAs)', 'Income Deprivation Affecting Older People (IDAOPI) Score (rate)', 'Income Deprivation Affecting Older People (IDAOPI) Rank (where 1 is most deprived)', 'Income Deprivation Affecting Older People (IDAOPI) Decile (where 1 is most deprived 10% of LSOAs)', 'Children and Young People Sub-domain Score', 'Children and Young People Sub-domain Rank (where 1 is most deprived)', 'Children and Young People Sub-domain Decile (where 1 is most deprived 10% of LSOAs)', 'Adult Skills Sub-domain Score', 'Adult Skills Sub-domain Rank (where 1 is most deprived)', 'Adult Skills Sub-domain Decile (where 1 is most deprived 10% of LSOAs)', 'Geographical Barriers Sub-domain Score', 'Geographical Barriers Sub-domain Rank (where 1 is most deprived)', 'Geographical Barriers Sub-domain Decile (where 1 is most deprived 10% of LSOAs)', 'Wider Barriers Sub-domain Score', 'Wider Barriers Sub-domain Rank (where 1 is most deprived)', 'Wider Barriers Sub-domain Decile (where 1 is most deprived 10% of LSOAs)', 'Indoors Sub-domain Score', 'Indoors Sub-domain Rank (where 1 is most deprived)', 'Indoors Sub-domain Decile (where 1 is most deprived 10% of LSOAs)', 'Outdoors Sub-domain Score', 'Outdoors Sub-domain Rank (where 1 is most deprived)', 'Outdoors Sub-domain Decile (where 1 is most deprived 10% of LSOAs)', 'Total population: mid 2015 (excluding prisoners)', 'Dependent Children aged 0-15: mid 2015 (excluding prisoners)', 'Population aged 16-59: mid 2015 (excluding prisoners)', 'Older population aged 60 and over: mid 2015 (excluding prisoners)', 'Working age population 18-59/64: for use with Employment Deprivation Domain (excluding prisoners) ']
        for x in df_social_columns:
            if("Score" in x):
                list_reduced.append(x)
        try:
            list_reduced.remove('LSOA code (2011)')
        except:
            pass
        cols_social=list_reduced
        df_input=df_in
        # PCA variables for the social fields will all be null, so propagate through.
        ctr=0
        for c in cols_social:
            df_input=df_input.rename(columns={c:'PCA_SOCIAL_{}'.format(ctr)})
            ctr+=1

        df_input=df_input.drop('LSOA code (2011)',axis=1)

        if(nRows>0):
            df_input=df_input.head(nRows)
        df_cpih=ProcessSalaryData(df_input,UseASData=True,updated_CPIH_Figs=True,logger=logger)
        df_test=df_cpih.copy(deep=True)
        #df_cpih.to_csv(ofile_CPIHCorr)

        print("**********")
        step1=time.time()
        logger.log('INFO',"CPI STEP:{} SECONDS ELAPSED".format(round(step1-start,3)))

        

    # Next function call
    

    # %%
    import sys
    #try:
    #    sys.path.append("../Python/")
    #except:
    #    pass

    import pandas as pd


    start=time.time()
    #df_test=pd.read_csv("./CSV/INPUT_CSV_2205/CPIH_CORR_3105.csv",index_col=0)
    if(aeonlyflag):
        df_test=df_in
    else:
        pass
        
    for p in df_test.columns:
        print(p)
    PII_VARIABLES=[
            'Age at start group',
            'Ethnicity',
            'Ethnic Minorities Flag',
            'Sex of apprentice',
            'LLDD Indicator',
            'Primary LLDD',
            'Workplace_LSOA', 
            'Learner LSOA'        
    ]

    PII_VARIABLES=[
        'Id',
        'CommitmentId',
        'ApprenticeshipId',
        'ULN',
        'UKPRN',
        'Level',
        'ProviderUkprn',
        'FLAG_AGGREGATED_LOWRATING',
        #'HASHED_ULN',
        'SectorSubjectAreaTier1',
        'SectorSubjectAreaTier2',
        'EmployerAccountId',
        'FrameworkOrStandardLarsCode',
        'LARSCODE',
        'StandardCode',
        'CompletionStatus'
    ]

    PII_VARIABLES=[
        'Id',
        'CommitmentId',
        'ApprenticeshipId',
        'ULN',
        'UKPRN',
        'Level',
        'ProviderUkprn',
        'FLAG_AGGREGATED_LOWRATING',
        #'HASHED_ULN',
        'SectorSubjectAreaTier1',
        'SectorSubjectAreaTier2',
        'EmployerAccountId',
        'FrameworkOrStandardLarsCode',
        'LARSCODE',
        'StandardCode',
        'CompletionStatus',
        #'Unnamed: 0',
        #'weighted_average_annual_minwage',
        #'weighted_average_annual_maxwage'
        'date_06MoBeforeDate',
        'date_12MoBeforeDate',
        'date_18MoBeforeDate',
        #'LSOA code (2011)'
        #'weightedaverage_minwage',
        #'weightedaverage_maxwage'
    ]
    logger.log('INFO','Finished CPIH correction, now running Autoencoder')
    
    from DataPreprocessingFunctions.Imputation_NN_PROD_PCA import ImputeVariables
    try:
        from DataPreprocessingFunctions.Imputation_NN_PROD_PCA import ImputeVariables
    except Exception as e:
        logger.log("ERROR","EXCEPTION: {}".format(e))
        logger.log('ERROR',"AutoEncoder will crash, so skip for moment - don't deploy this code to prod!")
        return df_test    
    
    logger.log('INFO',"MIDAS import OK")
    #return df_test
    try:
        df_test=df_test.drop('CURR_STAMP',axis=1)  
        df_test=df_test.drop('YESTERDAY',axis=1)
        df_test=df_test.drop('LASTWEEK',axis=1)
        df_test=df_test.drop('CreatedRecordDate',axis=1)
        df_test=df_test.drop('LearnStartDate',axis=1)
        df_test=df_test.drop('StopDate',axis=1)
        df_test=df_test.drop("SectorSubjectAreaTier2_Desc",axis=1)
        df_test=df_test.drop('SectorSubjectAreaTier1_Desc',axis=1)
        df_test=df_test.drop('StandardUId',axis=1)
        df_test=df_test.drop('CreatedOn',axis=1)
    except Exception as E:
        logger.log("ERROR","AE Exception: {}".format(E))
        pass
    df_test=df_test.copy(deep=True)#.reset_index()
    # get df of variables that we don't do anything with on the imputation - this is PII info
    df_PII=df_test[PII_VARIABLES].copy(deep=True)


    df_test=df_test.drop(PII_VARIABLES,axis=1)

    df_test_passvariables=df_test.select_dtypes(exclude=['int64','float64'])    


    df_test_floatvars=df_test.select_dtypes(['int64','float64'])
    print("+++++++++++++++++++++++++++++++")
    print(len(df_test_floatvars.columns))
    for col in df_test_floatvars.columns:
        print(col)
    print("++++++++++++++++++++++++++")
    
    logger.log('INFO',"STARTING IMPUTATION STEPS")
    #return df_test
    df_test_floatvars=ImputeVariables(df_test_floatvars,False,True)

    df_t=pd.concat([df_test_passvariables,df_test_floatvars,df_PII],axis=1)
    print(df_t.columns,len(df_t))
    step2=time.time()
    logger.log('INFO',"AE STEP: {} SEC ELAPSED".format(round(step2-start,3)))
    
    print("*******************")
    print("DONE")
    return df_t




# %%


# %%


# %% [markdown]
# 


