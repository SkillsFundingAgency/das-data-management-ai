# %%
import sys
try:
    sys.path.append("../Python/")
except:
    pass

import time




import argparse 
parser=argparse.ArgumentParser()
parser.add_argument('--outf',action='store',dest='ofile',required=True,help="Output File (CSV)",type=str)
parser.add_argument('--inf',action='store',dest='infile',required=True,help='Input File (CSV)',type=str)
parser.add_argument("--AEOnly",action='store_true',default='False',help="Skip CPIH correction")
parser.add_argument("--n",action='store',dest='n',help="NRows",default=-1,type=int)
args=parser.parse_args()
ofile=args.ofile

from ECON_CPIH_Correction import ProcessSalaryData
import pandas as pd
ofile=args.ofile#"./CSV/INPUT_CSV_2205/IMPUTATION_OUTPUT_FULL_3105.csv"
cpi_ofile=args.ofile.replace(".csv","_CPIH.csv")  #"./CSV/INPUT_CSV_2205/CPIH_CORR_3105.csv"
outfile=ofile
ofile_CPIHCorr=cpi_ofile
aeflag=False  #args.aeflag # flag to disable the AE
infile=args.infile#"./CSV/INPUT_CSV_2205/IMPUTATION_INPUT_FULL_3105.csv"




# %%
dstring="""
Infile: {}
Outfile: {}
CPIH_File: {}
Flag to disable AutoEncoder: {}
""".format(infile,outfile,ofile_CPIHCorr,aeflag)

print(dstring)

import time
start=time.time()
print("**********")
# RUN CPIH CORR
import sys
try:
    sys.path.append("../Python/")
except:
    pass
if(not args.AEOnly):
    df_input=pd.read_csv(infile)
    import PCA_Processing as PCA

    print("****** RUN PCA ON SOCIAL FIELDS")

    df_social=pd.read_csv("../CSV/ONS_Data_Input/iod_2019.csv")
    #filter out duplicate indices (only keep scores)

    key='LSOA code (2011)'
    list_reduced=[key]


    for x in df_social.columns:
        if("Score" in x):
            list_reduced.append(x)
    df_social=df_social[list_reduced]
    print(df_social)

    cols_social=list(df_social.columns)
    cols_social.remove('LSOA code (2011)')

    df_input=pd.read_csv(infile)
    # PCA variables for the social fields will all be null, so propagate through.
    ctr=0
    for c in cols_social:
        df_input=df_input.rename(columns={c:'PCA_SOCIAL_{}'.format(ctr)})
        ctr+=1
    df_input=df_input.drop('LSOA code (2011)',axis=1)

    if(args.n>0):
        df_input=df_input.head(args.n)
    df_cpih=ProcessSalaryData(df_input,UseASData=True,updated_CPIH_Figs=True)
    df_test=df_cpih.copy(deep=True)
    df_cpih.to_csv(ofile_CPIHCorr)

    print("**********")
    step1=time.time()
    print("CPI STEP:{} SECONDS ELAPSED".format(round(step1-start,3)))

    

# Next function call
print("NOW STARTING AUTOENCODER")

# %%
import sys
try:
    sys.path.append("../Python/")
except:
    pass

import pandas as pd


start=time.time()
#df_test=pd.read_csv("./CSV/INPUT_CSV_2205/CPIH_CORR_3105.csv",index_col=0)
df_test=pd.read_csv(ofile_CPIHCorr,index_col=0)
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
    'LSOA code (2011)'
    #'weightedaverage_minwage',
    #'weightedaverage_maxwage'
]
df_test=df_test.copy(deep=True)#.reset_index()
# get df of variables that we don't do anything with on the imputation - this is PII info
df_PII=df_test[PII_VARIABLES].copy(deep=True)


df_test=df_test.drop(PII_VARIABLES,axis=1)

df_test_passvariables=df_test.select_dtypes(exclude=['int64','float64'])    


df_test_floatvars=df_test.select_dtypes(['int64','float64'])
print("+++++++++++++++++++++++++++++++")
print(len(df_test_floatvars.columns))
print("++++++++++++++++++++++++++")
from employer_engagement.training.DataPreprocessingFunctions.Imputation_NN_PROD_PCA import ImputeVariables
print("STARTING IMPUTATION STEPS")
df_test_floatvars=ImputeVariables(df_test_floatvars,True,True)

df_t=pd.concat([df_test_passvariables,df_test_floatvars,df_PII],axis=1)
print(df_t.columns,len(df_t))
step2=time.time()
print("AE STEP: {} SEC ELAPSED".format(round(step2-start,3)))
df_t.to_csv(outfile)
print("*******************")
print("DONE")





# %%


# %%


# %% [markdown]
# 


