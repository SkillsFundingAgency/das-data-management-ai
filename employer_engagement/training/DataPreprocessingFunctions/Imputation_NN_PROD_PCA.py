#!/usr/bin/env/python

import pandas as pd
#import sklearn
import os
import tensorflow
import MIDASpy as midas
#from sklearn.preprocessing import MinMaxScaler
from keras.callbacks import EarlyStopping

from matplotlib import pyplot as plt
import numpy as np
np.random.RandomState(42)
pd.set_option('display.max_rows', 200)
pd.set_option('display.max_columns',500)
import sys

def GetNullColumns(indf):
    #print(indf.isna().any())

    nulltable=indf.isna().any().copy()
    #print("************")
    null_list=list(nulltable.index[nulltable].copy())
    #print("NULL LIST",null_list)
    return null_list

def Transform(df_in=pd.DataFrame(),mindict={}):
    df_out=df_in.copy(deep=True)
    for c in mindict.keys():
        try:
            df_out[c]=(df_out[c]-mindict[c]['min'])/mindict[c]['range']
        except:
            print("No variable found")
            pass
        #minval=df_in[c].min()
        #maxval=df_in[c].max()
        #rangeval=df_in[c].max()-df_in[c].min()
        #df_out[c]=(df_in[c]-minval)/rangeval
        #mindict[c]={
        #            'min':minval,
        #            'max':maxval,
        #            'range':rangeval
        #            }

    return df_out,mindict
def InvertTransform(df_scaled=pd.DataFrame(),mindict={}):
    df_inverted=df_scaled.copy(deep=True)
    for c in mindict.keys():
        try:
            minval=mindict[c]['min']
            maxval=mindict[c]['max']
            rangeval=mindict[c]['range']
            df_scaled[c]=df_scaled[c].fillna(0)
            df_inverted[c]=df_scaled[c]*rangeval+minval
        except:
            pass
    return df_inverted
    
def ImputeVariables(indf,diagnostic=False,cache=False):
    import os
    dirname=os.path.dirname(__file__)+"/"
    
    indf=indf.select_dtypes(["int64","float64"]).copy(deep=True)
    numerical_cols=list(indf.columns)

    if(diagnostic):
        cols=numerical_cols
        sum_null_list=[]
        percent_null_list=[]
        n_total=len(indf)
        for c in numerical_cols:
            sum_nulls=indf[c].isna().sum()
            sum_null_list.append(sum_nulls)
            nullpercent=(sum_nulls/n_total)*100
            percent_null_list.append(round(nullpercent,3))
            if(sum_nulls>100):
                print("COL: {} NULLS: {} ({} %)".format(c,sum_nulls,nullpercent))

        df_diagnostic=pd.DataFrame({'Variable':cols,'Number of Null Rows':sum_null_list,'Percent of total rows as Null':percent_null_list})

        df_diagnostic.sort_values(inplace=True,by='Percent of total rows as Null',ascending=False)
        df_diagnostic=df_diagnostic[df_diagnostic['Percent of total rows as Null']>5]
        print(df_diagnostic[df_diagnostic['Percent of total rows as Null']>5])        
        
        #df_diagnostic.to_csv(dirname+"../CSV/Processed_CSVs/ProfiledCSVs/DIAG_NULLROW_CSV_ASDATA_SINGLE.csv")
    


    


    #return outdf

  
    import json
    jf=open(dirname+"../ML_Models/Models/ScalerSetup.json",'r')
    mindict=json.load(jf)
    jf.close()
    
    scaled_df,mindict=Transform(indf,mindict=mindict)

    
    #scaler=MinMaxScaler()
    print("PRE SCALED DF SHAPE: {}".format(indf.to_numpy().shape))
    
    #data_scaled=scaler.fit_transform(indf)

    #scaled_df=pd.DataFrame(data_scaled,columns=indf.columns)
    print("POST SCALED DF SHAPE: {}".format(scaled_df.to_numpy().shape))
    print(scaled_df.head(3))

    imputer=midas.Midas(layer_structure=[256,256],
                        vae_layer=False,
                        seed=42,
                        input_drop=0.50,
                        #savepath=dirname+"../ML_Models/Models/DUMMY_AE/" # dummy data Autoencoder
                        savepath=dirname+r"..\ML_Models\Models\MIDAS_CHECKPOINTS_PROD_PCA\\" # real data autoencoder
                        )
    imputer.build_model(scaled_df,softmax_columns=[])
    if(not cache):
        
        # try out different profiles perhaps :
    
        imputer.train_model(training_epochs=10,verbose=True)    
    #imputer.train_model(training_epochs=10)
    else:
        print("LOADING AE FROM CACHE")
        # MIDAS does not have an explicit loader method, but does restore a model from defaults if previously built







        
    
    nsamples=10
    imputations=imputer.yield_samples(m=nsamples)
    
    analysisdfs=[]
    
    itr=0
    for imputation in imputations:
        df_unscaled=InvertTransform(imputation,mindict)
        df_unscaled=pd.DataFrame(df_unscaled,columns=indf.columns)
        analysisdfs.append(df_unscaled)
        print("imputation: {}".format(itr))
        #df_unscaled.to_csv(dirname+"../CSV/Processed_CSVs/ProfiledCSVs/AE_IMPUTATION_{}_{}_ASDATA_PROF.csv".format(itr,nsamples))
        print("Saved imputation to CSV")
        itr+=1

    d=True
    if(diagnostic and d):
        for c in indf.columns:
            print("PLOTTING {}".format(c))
            plt.close("all")
            ctr=0
            nonulls=indf.copy()
            nbins=20

            binmax=nonulls[c].max()
            binmin=nonulls[c].min()
            
            binwidth=(binmax-binmin)/nbins

            print("GOT BINSPEC")
            plt.hist(nonulls[c],label="Base distribution (no null)",bins=np.linspace(binmin,binmax+binwidth,nbins),histtype='step', color="black",linestyle="--")
            print("FIRSTPLOT")
            for df_impute in analysisdfs:
                print("itr: {}".format(ctr))
                plt.hist(df_impute[c],label="Imputation_{}".format(ctr),bins=np.linspace(binmin,binmax+binwidth,nbins),histtype='step',linestyle=":")
                ctr+=1
            plt.xlabel(c)
            plt.ylabel("Freq.")
            plt.legend()
            plt.title("Imputation distribution checker across the {} imputations".format(nsamples))
            print("SAVEFIG")
            #plt.savefig("C:\\Users\\manthony2\\OneDrive - Department for Education\\Documents\\ESFA_PROJECT\\Workdir\\Plots\\ImputationDiagnostics\\Autoencoder_{}_{}_ASDATA_PROF.png".format(nsamples,c))


    df_tmp=indf.copy(deep=True)
    #add a flag if the plot contains nullrows
    df_tmp['InterpolatedFlag']=df_tmp.notna().all(axis=1).astype(int)
    print("TRYING TO IMPUTE VALUES")

    df_tmp_null=df_tmp[df_tmp['InterpolatedFlag']==0]
    #print(df_tmp_null.head(40),len(df_tmp_null),len(df_tmp))
    
    nullindices=df_tmp[df_tmp['InterpolatedFlag']==0].index
    random_sample=[]
    

  
    for iter in range(0,len(nullindices)):
        randint=np.random.randint(0,nsamples-1)
        random_sample.append(randint)

    

    df_cpy=df_tmp.copy(deep=True)

    # for each row in the dataset, sample N AutoEncoder samples, and pick the corresponding row from the AutoEncoded sample
    autoencoder_dfs=analysisdfs 


    # Randomly sample & replace NANs with the AUTOENCODED equivalent.
    randctr=0
    for i in nullindices.values:
        #print(i)
        #print("Modifying ROW {}".format(i))
        #try:
        df_row=(df_cpy.iloc[i,]).to_frame()
        #except:
        #    print("Positional Indexer out of bounds on index: {}".format(i))
        #    continue
        # Because index of PD series gets put in a vertical fashion rather than horizontal, take transpose
        df_row=df_row.transpose()
        #print("NULLCOL")
        #print(df_row)
        #read off the rand int row
        randsample=random_sample[randctr]
        # get DF associated to that row, and get the same iloc
        #print("RANDNUMBER: {}".format(randsample))
        #print("AE row {}, {}".format(len(autoencoder_dfs[randsample]),i))
        autoencoder_row=autoencoder_dfs[randsample].iloc[i,].to_frame()
        autoencoder_row=autoencoder_row.transpose()

        #identify which columns in the row contain NaNs:
        #print("GOT AE")
        nullcols=GetNullColumns(df_row) #df_row.columns[df_row.isna().any()].tolist() # should only be 1 element
        #print("df_row.columns {}".format(list(df_row.columns)))
        #print("NULLCOLS ",nullcols)
        og_cols=list(df_row.columns)

        for c in nullcols: 
            #print("MODIFYING COLUMN {}".format(c))
            entry=autoencoder_row[c]
            #print("AE entry: {}".format(entry))
            cidx=0
            for x in range(0,len(og_cols)):
                if (og_cols[x]==c):
                    cidx=x
                    break
                else:
                    continue
            df_cpy.iloc[i,cidx]=entry

    


        if(randctr % 500==0):
            print("MODIFIED ROW {}/{}".format(randctr,len(nullindices)))
        randctr+=1
        
    outdf=df_cpy.copy(deep=True)

    return outdf
    


if __name__=="__main__":
    try:
        os.chdir("Python/")
    except:
        pass
    df_test=pd.read_csv("../CSV/Processed_CSVs/TestCSV_Imputation_CPIHCorr.csv",index_col=0)
    
    #print(df_test.iloc[9000])

    row=df_test.iloc[9000]
    print("**********")
    #tf=pd.DataFrame(row,columns=df_test.columns)
    tf=row.to_frame()
    #print(len(tf.columns),tf.columns)
    #print(tf)
    #print(row.to_frame().transpose())
    #print(list(tf.transpose().columns))
    tft=tf.transpose()
    print((list(tft.columns))[0])
    key=(list(tft.columns))[0]
    print(key)
    print(tft[key].values[0])
    

    nullcols=GetNullColumns(tft)
    df_out=df_test.copy(deep=True)
    og_cols=list(df_out.columns)

    for c in nullcols:
        print("*****")
        print(c,tft[c].values[0])
        
        cidx=0
        for p in range(0,len(og_cols)):
            if(c==og_cols[p]):
                cidx=p
                break
        df_out.iloc[9000,cidx]=99
        val=df_out.iloc[9000,cidx]
        
        print(c,val)
    print("++++++++++")
    
    print("***************************************")
    for c in nullcols:
        print(c,df_out[c].iloc[9000])



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
    df_test=df_test.copy(deep=True).reset_index()
    # get df of variables that we don't do anything with on the imputation - this is PII info
    df_PII=df_test[PII_VARIABLES].copy(deep=True)
    
    df_test=df_test.drop(PII_VARIABLES,axis=1)
    df_test_passvariables=df_test.select_dtypes(exclude=['int64','float64'])    


    df_test_floatvars=df_test.select_dtypes(['int64','float64'])

    print(df_test_floatvars.columns)
    print("****")
    print(df_test_passvariables.columns)
    
    df_test_floatvars=ImputeVariables(df_test_floatvars,True,True)
  

    df_t=pd.concat([df_test_passvariables,df_test_floatvars,df_PII],axis=1)
    print(df_t.columns,len(df_t))

    df_t.to_csv("../CSVs/Processed_CSVs/Imputation_AE_OUTPUT_2324DATASET.csv")
    
    