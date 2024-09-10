#!/usr/bin/env python 

# Python module to run the PCA on specific variables
import pandas as pd
from sklearn.decomposition import PCA

from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer,make_column_transformer

import numpy as np
import matplotlib.pyplot as plt
def OptimPCA(arr=None,nDefined=0,nCols=0,label=""):
    #Function to find optimal number of PCA components
    pca_optim=PCA(n_components=nCols)

    pca_optim.fit(arr)
    num_comps=np.arange(1,nCols,1)

    expl_var=np.cumsum(pca_optim.explained_variance_ratio_)[:nCols-1]
    plt.close('all')

    plt.plot(num_comps,expl_var,marker='x')
    plt.xlabel("Number of features")
    plt.ylabel("Explained variance by these features")
    plt.title("Eigenvalue search for optimal number of features ({})".format(label))
    
    plt.axhline(y=0.95,color='r',linestyle="--")
    plt.axhline(y=0.99,color="r",linestyle="--")
    plt.grid()
    
    #import pathlib
    #pth=pathlib.Path(__file__).parent_resolve()
    plt.savefig("../Plots/PCAConfig/PCA_Eigenvalue_scan_{}.png".format(label))
    plt.close("all")
    last_index=0
    ctr=0
    for v in expl_var:
        #print("VAR",v)
        if(v<0.99):
            last_index=ctr
        ctr+=1
    print("Optimization Recommends {} components for {}".format(last_index,label))
    return last_index


def run_pca(indf=pd.DataFrame(),
            cols=[],
            n_components=3,
            label="ECON",
            optim=False,
            cache=False,
            verbose=False,
            DumpAllVars=False,
            ProdMode=False
            ):

    df_og=indf.copy(deep=True)
    df_pca=df_og.copy(deep=True)

    if(len(cols)==0):
        cols_economicfields=[x for x in list(df_og.columns) if("BeforeDate" in x)]
    else:
        cols_economicfields=cols
    #print(cols_economicfields)
 
        

    print("NUMBER OF INPUT VARIABLES {}".format(len(cols_economicfields)))
    #for c in cols_economicfields:
    #    print("SUM NULL ROWS: {}  out of {} in entry {}".format(df_og[c].isna().sum(),len(df_og),c))

    #df_normalized=(df_og-df_og.mean())/df_og.std()

    #pca=PCA(n_components=len(df_normalized.columns))

    #pca.fit(df_normalized[cols_economicfields])
    if(verbose):
        for var_in in cols_economicfields:
            n_null=len(df_og[df_og[var_in].isna()])
            perc_null=(n_null/len(df_og))*100
            print("COL: {}, Number of entries: {}, Number missing : {} ({})".format(var_in,len(df_og[var_in]),n_null,round(perc_null,3)))
    
    imp=SimpleImputer(missing_values=np.nan,strategy='constant',fill_value=0.0)

    varlist=[]
    itr=0
    for c in cols_economicfields:
        varlist.append((c,imp,[itr]))
        itr+=1

    #for v in varlist:
    #    print(v)
    column_trans=ColumnTransformer(
        varlist,
        remainder='passthrough'
    )    

    #df_imputed=pd.DataFrame(column_trans.fit_transform(df_og[cols_economicfields].astype("float64").to_numpy()),columns=df_og[cols_economicfields].columns)
    df_imputed=df_og[cols_economicfields].fillna(value=0.0).astype('float64')#.to_numpy()

    df_imputed=df_imputed[cols_economicfields].copy(deep=True)


    if(not ProdMode):
        #productio
        df_normalized=(df_imputed-df_imputed.mean())/df_imputed.std()
    if(ProdMode):

        df_normalized=df_imputed.copy()
        import json
        import os
        dirname=os.path.dirname(__file__)+"/"
        jft=open(dirname+"../ML_Models/Models/ScalerSetup.json")
        jf=json.load(jft)
        for c in jf.keys():
            try:
                df_normalized[c]=(df_imputed[c]-jf[c]['mean'])/jf[c]['std']
            except:
                pass

    #print(df_normalized)
  
    nBefore=len(df_normalized)
    print("BEFORE ",nBefore)
    #for c in df_normalized.columns:
    #    print(c)
        #print(df_normalized[c])
    df_normalized=df_normalized.dropna().copy()
    nAfter=len(df_normalized)
    print("AFTER: {}, LOST: {}".format(nAfter,nBefore-nAfter))

    if(optim):
        print("GENERATING OPTIMIZATION PLOT FOR PCA")
        n_components=OptimPCA(df_normalized.to_numpy(),n_components,len(df_normalized.columns),label)

    if(cache):
        print("LOADING PCA FROM CACHE")
        fname=dirname+"../ML_Models/Models/PCA_{}.pkl".format(label)
        import pickle
        try:
            pca=pickle.load(open(fname,'rb'))
            arr_pca=pca.transform(X=df_normalized.to_numpy())

        except Exception as e:
            print("EXCEPTION {}".format(e))
            return pd.DataFrame()
    else:
        pca=PCA(n_components=n_components)
        arr_pca=pca.fit_transform(X=df_normalized.to_numpy())

    #print(arr_pca)
    indices=df_normalized.index
    # As arbitrary number of new PCA variables set by user, choose a set with new names
    list_new_columns=[
        "PCA_{}_{}".format(label,x) for x in range(0,n_components)
    ]
    df_pca=pd.DataFrame(arr_pca,columns=list_new_columns)
    #print(df_pca)

    cols_nopca=list(indf.columns)
    for c in cols_economicfields:
        if(DumpAllVars):
            continue
        try:
            cols_nopca.remove(c)
        except:
            pass
    
    outdf=indf[cols_nopca].copy(deep=True)
    for newc in list_new_columns:
        outdf[newc]=df_pca[newc].copy()


    #outdf=pd.concat([outdf_tmp,df_pca[list_new_columns]],axis=1)
    print("OUTPUT OF PCA: ",len(outdf),len(df_pca),round((len(outdf)-len(df_pca)/len(outdf))*100,3))
    if(not cache):
        import pickle as pk
        print("DUMP PCA MODEL CONFIG")
        pk.dump(pca,open("../models/PCA_{}.pkl".format(label),'wb'))
    
    if(verbose):
        print("DIAGNOSTIC")
        for p in cols_nopca:
            print(p)
    if(verbose):
        print("END OF JOB")
        for var_in in outdf.columns:
            n_null=len(outdf[outdf[var_in].isna()])
            perc_null=(n_null/len(outdf))*100
            print("COL: {}, Number of entries: {}, Number missing : {} ({})".format(var_in,len(outdf[var_in]),n_null,round(perc_null,3)))
    if(verbose):
        explained_variances_ratio=pca.explained_variance_ratio_
        print("\n\n\n\n***********************************************************")
        print("EXPLAINED VARIANCE RATIO")
        print(explained_variances_ratio)
        print("\n\n\n\n\n\n\n\n\n\n\n\n")
        print(pca.components_[0:5])
        np.save('../Plots/pcacomp_{}.bin'.format(label),pca.components_[0:5])
        np.save('../Plots/pcaexplained_varratio_{}.bin'.format(label),explained_variances_ratio)
        np.save('../Plots/pca_explained_variance_{}.bin'.format(label),pca.explained_variance_)
        import json
        fp=open('../Plots/l_cols_lookup_{}.json'.format(label),'w+')
        json.dump(list(df_normalized.columns),fp,indent=4)
        fp.close()
    return outdf




