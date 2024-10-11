# %%
import sys
import pandas as pd
import numpy as np
import os


try:
    from azureml.data.datapath import DataPath
    from azureml.core import Workspace, Datastore, Dataset, ComputeTarget, Experiment, ScriptRunConfig, Environment, Model
    from azureml.core.run import Run
except:
    print('INFO',"No AML libs installed, can still run but will crash on AML")
    pass
import logging 




class ErrorHandler:
    """Dual Use Error Handler that can be used with/without AzureML Run in a transparent way once set"""
    isAzure=False # is AzureML in this case
    logfunction="AI model" #
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
            elif("debug" in logtype.lower()):
                self.logger.debug(logstring)
            else:
                self.logger.info(logstring)
        self.logctr+=1

#filter out duplicate indices (only keep scores)
def RunBDTModel(infile="",outfile="",plots=False,PandasInput=pd.DataFrame(),RunMemCheck=False):
    memtracker=None
    if(RunMemCheck):
        from pympler import tracker
        from pympler import muppy
        memtracker=tracker.SummaryTracker()
    isAzure=False
    logger=None
    run=None
    try:
        aml_workspace = Run.get_context().experiment.workspace
        #datastore = Datastore.get(aml_workspace, datastore_name='datamgmtdb')
        run = Run.get_context()
        isAzure=False # force logs to be written to disk
    except Exception as e:
        print("No AML workspace detected - now using logger logs")  
        print("AML ERROR: {}".format(e))
        pass      
    logger=ErrorHandler(isAzure,logstep="Preprocessing",run=run)


    key='LSOA code (2011)'
    list_reduced=[key]

    df_social=pd.read_csv("./ML_Models/ONSData/iod_2019.csv")
    for x in df_social.columns:
        if("Score" in x):
            list_reduced.append(x)
    df_social=df_social[list_reduced]
    logger.log('INFO',df_social)


    if(len(PandasInput)>0):
        df_FullEconVariables_input=PandasInput.copy()
    else: # local file load
        df_FullEconVariables_input=pd.read_csv(infile,index_col=0)#"./CSV/INPUT_CSV_2205/BDT_INPUT_STARTS2324.csv",index_col=0)
    #df_FullEconVariables_input=df_FullEconVariables_input[pd.to_datetime(df_FullEconVariables_input['EndDate'],errors='coerce')<pd.Timestamp('2023-08-01')]
    #df_FullEconVariables_input=df_FullEconVariables_input[pd.to_datetime(df_FullEconVariables_input['EndDate'],errors='coerce')<pd.Timestamp('2023-08-01')]

    import DataPreprocessingFunctions.PCA_Processing as PCA

    df_FullEconVariables_red=df_FullEconVariables_input.drop('SectorSubjectAreaTier2',axis=1)
    for c in df_FullEconVariables_red.columns:
        if("date_" in c):
            df_FullEconVariables_red.drop(c,inplace=True,axis=1)
        if(c=='06MoBeforeDate' or c=='12MoBeforeDate' or c=='18MoBeforeDate' or c=='24MoBeforeDate'):
            df_FullEconVariables_red.drop(c,inplace=True,axis=1)
        if(c=='StartDate'):
            df_FullEconVariables_red['StartDate']=pd.to_datetime(df_FullEconVariables_red['StartDate'],errors='coerce')
        if(c=='EndDate'):
            df_FullEconVariables_red['EndDate']=pd.to_datetime(df_FullEconVariables_red['EndDate'],errors='coerce')
        if(c=='PlannedEndDate'):
            df_FullEconVariables_red['PlannedEndDate']=pd.to_datetime(df_FullEconVariables_red['PlannedEndDate'],errors='coerce')

        if('date' in c.lower()):
            logger.log('INFO',"COL_READ : {}".format(c))
    logger.log('INFO',df_FullEconVariables_red[['PlannedEndDate','LearnActEndDate','StartDate']])
    df_FullEconVariables_red['Course_Duration']=(pd.to_datetime(df_FullEconVariables_red['PlannedEndDate'],errors='coerce').dt.to_period('M')-pd.to_datetime(df_FullEconVariables_red['StartDate'],errors='coerce').dt.to_period('M')).apply(lambda x: x.n if pd.notnull(x) else 0)
    df_FullEconVariables_red['ActualDuration']=(pd.to_datetime(df_FullEconVariables_red['LearnActEndDate'],errors='coerce').dt.to_period('M')-df_FullEconVariables_red['StartDate'].dt.to_period('M')).apply(lambda x: x.n if pd.notnull(x) else 0)

    cols_social=list(df_social.columns)
    cols_social.remove('LSOA code (2011)')

    logger.log('INFO',cols_social)

    #df_opt=PCA.run_pca(
    #    df_FullEconVariables_red.copy().fillna(0),
    #    n_components=len(cols_social),
    #    cols=cols_social,
    #    label='SOCIAL',
    #    verbose=True,
    #    ProdMode=True
    #    )
    df_opt=df_FullEconVariables_red.copy()
    logger.log('INFO',list(df_opt.columns))
    l_econfields=[x for x in list(df_opt.columns) if "BeforeDate" in x]
    logger.log('INFO',l_econfields)
    #df_post_pca=df_opt.copy()
    df_post_pca=PCA.run_pca(
        df_opt.copy().fillna(0),
        n_components=len(l_econfields),
        cols=l_econfields,
        label='ECON_DUMMY',
        verbose=False,
        ProdMode=True
        ,cache=True
    )

    logger.log('INFO',df_post_pca)

    for c in df_post_pca.columns:
        logger.log('INFO',c)

    import string
    def NumericallyEncodeSectorEstimate(x,splitchar=":"):
        if(splitchar not in x):
            return -99
        category=x.split(splitchar)[0]
        category=category.strip() # remove whitespace if at all used
        encoding=int(string.ascii_lowercase.index(category.lower())+1) 
        # Notion of encoding is A=1, B=2, C=3, D=4,...,Z=26
        return encoding

    df_post_pca['Employer_sector_estimate_encoded']=df_post_pca['Employer sector estimate'].apply(lambda x:NumericallyEncodeSectorEstimate(x))
    df_post_pca['Employee_size_estimate_encoded']=df_post_pca['Employee_size_estimate'].apply(lambda x: NumericallyEncodeSectorEstimate(x,splitchar=")"))
    df_post_pca['Employer_levy_status_encoded']=df_post_pca['Employer type'].apply(lambda x: 0 if "non-" in x.lower() else 1)

    # %% [markdown]
    # # Input diagnostics

    # %%
    cols_input=list(df_FullEconVariables_input.columns)

    for c in cols_input:
        logger.log('INFO',c)

    logger.log('INFO',df_FullEconVariables_input[(df_FullEconVariables_input['unixtimediff_exp_end']>200)|(df_FullEconVariables_input['unixtimediff_start']<-200)])

    logger.log('INFO',df_FullEconVariables_input[df_FullEconVariables_input['unixtimediff_start']<-100])

    # -ve course durations

    df_diag=df_FullEconVariables_input.copy()

    df_diag['Duration']=(pd.to_datetime(df_diag['EndDate_corr']).dt.to_period('M')-pd.to_datetime(df_diag['StartDate_corr']).dt.to_period('M')).apply(lambda x: x.n if not pd.isnull(x) else -9999)

        
    logger.log('INFO',df_diag)

    # %%
    df_plot=df_FullEconVariables_red.copy()

    df_plot['Withdraw']=df_plot['CompletionStatus'].apply(lambda x:
                                                        -1 if x==1
                                                        else 0 if x==2
                                                        else 1 if x==3
                                                        else np.nan
                                                        )

    import matplotlib.pyplot as plt
    df_plot_withdrew=df_plot[df_plot['Withdraw']==1].copy()
    df_plot_completed=df_plot[df_plot['Withdraw']==0].copy()

    nbins=40
    for column in df_plot_withdrew.columns:
        if("Actual Withdrawal" in column.lower()):
            continue
        if("ithdraw" in column.lower()):
            continue

        logger.log('INFO',"Processing {}".format(column))
        try:
            binmax=max(df_plot_completed[column].max(),df_plot_withdrew[column].max())
            binmin=min(df_plot_withdrew[column].min(),df_plot_completed[column].min())
            binwidth=(binmax -binmin)/nbins
            if (binwidth<1e-4):
                continue
        except:
            continue
    
        try:
            binspec= np.arange(binmin,binmax+binwidth,binwidth)
        except:
            continue
        plt.close("all")
        plt.hist(df_plot_withdrew[column],color="red",label="Actual Withdraw",bins=np.arange(binmin,binmax+binwidth,binwidth),stacked=True,density=True,histtype='step')
        plt.hist(df_plot_completed[column],color="blue",label="Actual Complete",bins=np.arange(binmin,binmax+binwidth,binwidth),stacked=True,density=True,histtype='step')

        #plt.hist(df_lvl_completed_true[column],color="green",label="True Negative",bins=np.arange(binmin,binmax+binwidth,binwidth),stacked=True,density=True,histtype='step')
        #plt.hist(df_lvl_completed_false[column],color="black",label="False Negative",bins=np.arange(binmin,binmax+binwidth,binwidth),stacked=True,density=True,histtype='step')
    
        plt.ylabel("Freq density")
        
        plt.xlabel(column)
        plt.legend()
        plt.title(column)
        try:
            1==1
            #plt.savefig("./Plots/BDTProfiling/AccuracyMetrics/INPUTDATA/diagnostic_alldata_metrics_{}.png".format(column.lower().replace(" ","_")))
        except:
            pass
        #plt.show()
        plt.close("all")

        # no density plot

        plt.close("all")
        plt.hist(df_plot_withdrew[column],color="red",label="Actual Withdraw",bins=np.arange(binmin,binmax+binwidth,binwidth),histtype='step')
        plt.hist(df_plot_completed[column],color="blue",label="Actual Complete",bins=np.arange(binmin,binmax+binwidth,binwidth),histtype='step')

        #plt.hist(df_lvl_completed_true[column],color="green",label="True Negative",bins=np.arange(binmin,binmax+binwidth,binwidth),histtype='step')
        #plt.hist(df_lvl_completed_false[column],color="black",label="False Negative",bins=np.arange(binmin,binmax+binwidth,binwidth),histtype='step')
    
        plt.ylabel("Frequency")
        
        plt.xlabel(column)
        plt.legend()
        plt.title(column)
        try:
            #plt.savefig("./Plots/BDTProfiling/AccuracyMetrics/INPUTDATA/diagnostic_alldata_metrics_{}_nodensity.png".format(column.lower().replace(" ","_")))
            1==1
        except:
            pass
        #plt.show()
        plt.close("all")



    # %% [markdown]
    # # Profile a JOIN to the BDT model to get the flags
    # 

    # %%
    #df_survey=pd.read_csv(r'..\CSV\OpinionPoll_PatchSQL.csv')


    #df_survey.columns=[x+"_custom" for x in df_survey.columns]
    #logger.log('INFO',df_survey.head(2))
    df_IPT=df_post_pca.copy()
    #df_IPT=df_IPT.drop(['LARSCODE','FLAG_AGGREGATED_LOWRATING'],axis=1)
    #for c in df_IPT.columns:
    #    if(("StandardCode" in c or 'UKPRN' in c) and "HASHED" not in c):
    #        logger.log('INFO',c)



    logger.log('INFO',"**********************")
    #for itr in xrange(0,len(df_IPT.columns)):
    #    logger.log('INFO',list(df_IPT.columns)[itr])
    #    if(itr%5==0):
    #        logger.log('INFO',"\n")
    #df_merge_poll=pd.merge(left=df_IPT,
    #                       right=df_survey,
    #                       left_on=['StandardCode','UKPRN'],
    #                       right_on=['LARSCODE','UKPRN'],
    #                       how='left'
    #                       )
    df_merge_poll=df_IPT.copy()
    logger.log('INFO',(len(df_IPT),len(df_merge_poll)))
    logger.log('INFO',df_merge_poll[df_merge_poll['FLAG_AGGREGATED_LOWRATING'].notna()])
    df_merge_poll['FLAG_AGGREGATED_LOWRATING']=df_merge_poll['FLAG_AGGREGATED_LOWRATING'].fillna(0)


    # %% [markdown]
    # ## Withdrawal rate checks (actual withdrawal)

    # %%
    df_plot_grpby=df_plot.copy()
    df_plot_grpby['Level']=df_plot_grpby['Level'].astype('int64',errors='ignore')
    df_plot_grpby['QMonth']=pd.PeriodIndex(df_plot_grpby['StartDate'],freq='Q').to_timestamp()
    df_plot_grpby['QMonth_End']=pd.PeriodIndex(df_plot_grpby['EndDate'],freq='Q').to_timestamp()
    df_plot_grpby=df_plot_grpby[df_plot_grpby['Withdraw']>=0]
    pvt_table=pd.pivot_table(df_plot_grpby,
        columns='Withdraw',
        index='Level',
        values='ApprenticeshipId',
        aggfunc='count'
    )

    #logger.log('INFO',pvt_table[pvt_table.index>=0].T)

    df_pvt=pvt_table[pvt_table.index>=0].reset_index().rename_axis(None,axis=1)

    #df_pvt['Sum']=df_pvt[0.0]+df_pvt[1.0]
    #df_pvt['% Complete (per level)']=(df_pvt[0.0]*100)/(df_pvt['Sum'])
    #df_pvt['% Withdraw (per level)']=(df_pvt[1.0]*100)/(df_pvt['Sum'])
    #df_pvt=df_pvt.rename(columns={0.0:'N Completed',1.0:'N Withdrawn'})
    #df_pvt.drop('Sum',axis=1,inplace=True)
    logger.log('INFO',df_pvt)


    pvt_table_date=pd.pivot_table(
        df_plot_grpby,
        columns=['Withdraw'],
        index=['Level','QMonth'],
        values='ApprenticeshipId',
        aggfunc='count'
        )
    #
    #logger.log('INFO',pvt_table_date.reset_index().rename_axis(None,axis=1))

    #df_pvt_date=pvt_table_date.reset_index().rename_axis(None,axis=1)
    #df_pvt_date['Sum']=df_pvt_date[0.0]+df_pvt_date[1.0]
    #df_pvt_date['% Complete (per level)']=(df_pvt_date[0.0]*100)/(df_pvt_date['Sum'])
    #df_pvt_date['% Withdraw (per level)']=(df_pvt_date[1.0]*100)/(df_pvt_date['Sum'])
    #df_pvt_date=df_pvt_date.rename(columns={0.0:'N Completed',1.0:'N Withdrawn'})
    #df_pvt_date.drop('Sum',axis=1,inplace=True)

    #logger.log('INFO',df_pvt_date)

    #plt.close("all")
    #for l in range(2,8):
    #    logger.log('INFO',"LEVEL: {}".format(l))
    #    df_pvt_lvl=df_pvt_date[df_pvt_date['Level']==l].copy()
    #    logger.log('INFO',df_pvt_date[df_pvt_date['Level']==l])
        
    #    plt.plot(df_pvt_lvl['QMonth'],df_pvt_lvl['% Withdraw (per level)'],label="L {} Withdrawal (%)".format(l))
    #    #plt.plot(df_pvt_lvl['QMonth'],df_pvt_lvl['% Complete (per level)'],label=cols_social"L {} Complete (%)".format(l))
    #plt.ylabel("(%) of candidates finishing by start month")
    #plt.xlabel('Start Date (Quarter)')
    #plt.legend()
    #plt.show()





    #pvt_table_date_end=pd.pivot_table(
    #    df_plot_grpby,
    #    columns=['Withdraw'],
    #    index=['Level','QMonth_End'],
    #    values='ApprenticeshipId',
    #    aggfunc='count'
    #    )
    #
    #logger.log('INFO',pvt_table_date.reset_index().rename_axis(None,axis=1))

    #df_pvt_date_end=pvt_table_date_end.reset_index().rename_axis(None,axis=1)
    #df_pvt_date_end['Sum']=df_pvt_date_end[0.0]+df_pvt_date_end[1.0]
    #df_pvt_date_end['% Complete (per level)']=(df_pvt_date_end[0.0]*100)/(df_pvt_date_end['Sum'])
    #df_pvt_date_end['% Withdraw (per level)']=(df_pvt_date_end[1.0]*100)/(df_pvt_date_end['Sum'])
    #df_pvt_date_end=df_pvt_date_end.rename(columns={0.0:'N Completed',1.0:'N Withdrawn'})
    #df_pvt_date_end.drop('Sum',axis=1,inplace=True)

    #logger.log('INFO',df_pvt_date_end)

    #plt.close("all")
    #for l in range(2,8):
    #    logger.log('INFO',"LEVEL: {}".format(l))
    #    df_pvt_lvl=df_pvt_date_end[df_pvt_date_end['Level']==l].copy()
    #    logger.log('INFO',df_pvt_date_end[df_pvt_date_end['Level']==l])
    #    
    #    plt.plot(df_pvt_lvl['QMonth_End'],df_pvt_lvl['% Withdraw (per level)'],label="L {} Withdrawal (%)".format(l))
    #    #plt.plot(df_pvt_lvl['QMonth'],df_pvt_lvl['% Complete (per level)'],label="L {} Complete (%)".format(l))
    #plt.ylabel("(%) of candidates finishing by End Quarter")
    #plt.xlabel('Expected End Date (Quarter)')
    #plt.legend()
    ##plt.show()

    # %%
    df_input=df_merge_poll.copy()
    for c in df_input.columns:
        if("PCA" in c):
            logger.log('INFO',"Reformatting {}".format(c))
            df_input[c]=pd.to_numeric(df_input[c])
        if("CompletionStatus" in c):
            df_input[c]=pd.to_numeric(df_input[c])
            
    lc=list(df_input.select_dtypes(['int64','float64']).columns)
    logger.log('INFO',lc)
    logger.log('INFO',df_input.dtypes)
    logger.log('INFO',list(df_input.columns))
    logger.log('INFO',lc)
    Withdrawflag='CompletionStatus'
    IDs_to_discard=[
        #'Unnamed: 0',
        'Id',
        'CommitmentId',
        'ApprenticeshipId',
        'ULN',
        #'HASHED_ULN',
        'SectorSubjectAreaTier1',
        #'SectorSubjectAreaTier2',
        'EmployerAccountId',
        'FrameworkOrStandardLarsCode',
        'LARSCODE',
        'CompletionStatus',
        'Withdraw',
        #'HASHED_UKPRN',
        #'unixtimediff_exp_end',
        'weighted_average_annual_minwage',
        'weighted_average_annual_maxwage',
        'ProviderUkprn',
        'InterpolatedFlag',
        'ActualDuration',
        #'LSOA code (2011)',
        'DelLoc_Pst_Lower_Layer_SOA',
    # 'Unnamed: 0.1'
    ]
    for c in df_input.columns:
        if("PCA_SOCIAL" in c):
            IDs_to_discard.append(c)

    #df_calib=df_input.select_dtypes(['float64','int64']).copy()

    #for i in IDs_to_discard:
    #    try:
    #        df_calib.drop(i,axis=1,inplace=True)
    #    except:
    #        pass
    df_calib=df_input.copy()
    logger.log('INFO',df_calib.columns)

    # %%
    df_in_sort=df_calib.sort_values(by='unixtimediff_start',ascending=True).copy()
    logger.log('INFO',list(df_calib.columns))
    logger.log('INFO',df_in_sort['unixtimediff_start'])



    from sklearn.model_selection import train_test_split
    # plot some variables on the date field

    df_profiling_pretraining=df_in_sort.copy()
    df_profiling_pretraining['Withdraw']=df_profiling_pretraining['CompletionStatus'].apply(lambda x:
                                                                                            -1 if x==1
                                                                                            else 0 if x==2
                                                                                            else 1 if x==3
                                                                                            else np.nan
                                                                                            )
    #df_profile_tmp=df_profiling_pretraining[df_profiling_pretraining['Withdraw']>=0]
    #datefield_start=df_profiling_pretraining['Learner start date']
    #datefield_end=df_profiling_pretraining['Learner planned end date']

    #datefield_delta_start=df_profiling_pretraining['unixtimediff_start']
    #datefield_delta_start=df_profiling_pretraining['unixtimediff_exp_end']


    import matplotlib.pyplot as plt
    #df_profiling_pretraining[['Learner start date','Learner planned end date','unixtimediff_start','unixtimediff_exp_end']].head(2)
    #df_profile_tmp=df_profiling_pretraining[list_columns_train+['Learner start date','Learner planned end date','unixtimediff_start','unixtimediff_exp_end']]


    df_calib_t=df_input.select_dtypes(['float64','int64']).copy()

    for i in IDs_to_discard:
        try:
            df_calib_t.drop(i,axis=1,inplace=True)
        except:
            pass
    l_cols_train=list(df_calib_t.columns)
    df_profile_tmp=df_profiling_pretraining.copy()

    x_prof=df_profile_tmp.copy()
    logger.log('INFO',list(df_profile_tmp.columns))
    x_prof=x_prof.drop(['Withdraw',
                        #'CompletionStatus',
                        'weighted_average_annual_minwage',
                        'weighted_average_annual_maxwage'
                        #'SectorSubjectAreaTier2.1',
                        #'ProviderUKPRN'
                        ,'unixtimediff_exp_end'
                        #,'LSOA code (2011)'
                        ,'DelLoc_Pst_Lower_Layer_SOA'
                        ],axis=1)
    y_prof=df_profile_tmp['Withdraw']

    x_train_profile,x_test_profile,y_train_profile,y_test_profile = train_test_split(x_prof,y_prof,test_size=0.2,shuffle=False)

    logger.log('INFO',"TRAINING SET")
    logger.log('INFO',x_train_profile[['unixtimediff_start']])

    logger.log('INFO',"TEST SET")
    logger.log('INFO',(x_test_profile[['unixtimediff_start']]))


    # %% [markdown]
    # ## Train the BDT model

    # %%
    import numpy as np
    df_tmp=df_profile_tmp.copy()
    df_batch=df_tmp.copy()

    x=df_tmp[l_cols_train+IDs_to_discard]
    #x=df_tmp.copy()
    y=df_tmp["Withdraw"]

    list_mask_variables=list(IDs_to_discard)
    x_train_t, x_test_t, y_train_t, y_test_t = train_test_split(x, y,test_size=10e-9,shuffle=False)

    mask_train_t=x_train_t[list_mask_variables].copy()
    mask_test_t=x_test_t[list_mask_variables].copy()


    x_train_t=x_train_t.drop(list_mask_variables,axis=1); 
    x_test_t=x_test_t.drop(list_mask_variables,axis=1)

    logger.log('INFO',"COLUMNS FOR TRAINING: {}".format(len(x_train_t.columns)))
    for c in list(x_train_t.columns):
        logger.log('INFO',c)

    logger.log('INFO',"******GLOBAL MODEL*****************")

    import xgboost as xgb
    evalset_t=[(x_train_t,y_train_t),(x_test_t,y_test_t)]
    model_t = xgb.XGBClassifier(use_label_encoder=False,objective='binary:logistic',n_estimators=1000,early_stopping_rounds=45,learning_rate=0.05,eval_metric='logloss')
    logger.log('INFO',"def model, get traintest etc")
    #model_t.fit(x_train_t,y_train_t,eval_set=evalset_t,verbose=True)
    #logger.log('INFO',"BEST ITERATION: {}".format(model_t.get_booster().best_iteration))

    #model_t.load_model("./ML_Models/Models/Model_BDT_NoLocation.model")
    model_t.load_model("./ML_Models/Models/BDTStepConfig/Model_BDT__DUMMYDATA.model")
    logger.log("BDT MODEL LOADED OK")

    if(RunMemCheck):
        print("**************************")
        print("MEMORY CHECKPOINT: BDT load")
        memtracker.print_diff()
        print("**************************")
    preds_t=model_t.predict(x_test_t)
    
    df_testset_t=x_test_t.copy()
    df_testset_t['Predicted Withdrawal']=preds_t
    df_testset_t['Actual Withdrawal'] = y_test_t.copy()
    df_testset_t[list_mask_variables]=mask_test_t
    df_testset_t['BDT_PROB_COMPLETE']=model_t.predict_proba(x_test_t)[:,0]
    df_testset_t['BDT_PROB_WITHDRAW']=model_t.predict_proba(x_test_t)[:,1]
    logger.log('INFO',np.array(model_t.predict_proba(x_test_t)).shape)

    df_modelinput=x_train_t.copy()
    #df_modelinput_noPredVariables=df_modelinput.copy()
    logger.log("BDT MODEL MADE A PREDICTION ON ALL DATA SET")
    df_modelinput['Actual Withdrawal']=y_train_t.copy()
    df_modelinput['Predicted Withdrawal']=model_t.predict(x_train_t)
    df_modelinput[list_mask_variables]=mask_train_t.copy()
    df_modelinput['BDT_PROB_COMPLETE']=model_t.predict_proba(x_train_t)[:,0]
    df_modelinput['BDT_PROB_WITHDRAW']=model_t.predict_proba(x_train_t)[:,1]

    df_modeloutput=df_testset_t.copy()

    del x_train_t
    del y_train_t
    del x_test_t
    del y_test_t
    logger.log('INFO',f"LENGTHCHECK {[len(df_modelinput),len(df_modeloutput),len(df_modelinput)+len(df_modeloutput),len(x)]}")


    logger.log('INFO',"FINISHED BDT APPLICATION")

    if(RunMemCheck):
        print("**************************")
        print("MEMORY CHECKPOINT: PRE DF CONCAT")
        memtracker.print_diff()
        #biggest_objects=muppy.sort(muppy.get_objects())[-100:]
        #print("BIGGEST OBJECTS")
        print("****")
    df_model_allout=pd.concat([df_modelinput,df_modeloutput])
    # now that I bolted everything back together, remove the copies
    del df_modelinput
    del df_modeloutput
    print("Finished cleanup of concat - hopefully")
    if(RunMemCheck):
        print("**************************")
        print("MEMORY CHECKPOINT: BDT EVALUATION & SORT")
        memtracker.print_diff()
        biggest_objects=muppy.sort(muppy.get_objects())[-100:]
        #print("BIGGEST OBJECTS")
        #import objgraph   
        #objgraph.show_backrefs(biggest_objects,filename="REFFILE.png")
        print("**************************")

    df_model_ABsorting=df_model_allout.sort_values(by='BDT_PROB_COMPLETE',ascending=True).copy()
    
    dblen=len(df_model_ABsorting)
    import math
    partsize=math.ceil(dblen/2)
    logger.log('INFO',f'{(dblen,partsize)}')

    df_pred_wdraw=df_model_ABsorting[df_model_ABsorting['Predicted_Withdrawal']==1]
    df_pred_complete=df_model_ABsorting[df_model_ABsorting['Predicted_Withdrawal']==0]

    df_model_ABsorting['Email_Classification']='Email A'
    emailBalloc_wdraw=df_pred_wdraw['Email_Classification'].sample(frac=0.5,random_state=42)
    emailBalloc_complete=df_pred_complete['Email_Classification'].sample(frac=0.5,random_state=42)
    
    df_model_ABsorting['Email_Classification'].iloc[emailBalloc_wdraw.index]='Email B'
    df_model_ABsorting['Email_Classification'].iloc[emailBalloc_complete.index]='Email B'

    #partition=df_model_ABsorting.index[0:]

    #df_model_ABsorting['HiLoPartition']="NONE"

    #df_model_ABsorting['HiLoPartition'].iloc[df_model_ABsorting.index<partsize]='Lo'
    #df_model_ABsorting['HiLoPartition'].iloc[df_model_ABsorting.index>=partsize]='Hi'
    #logger.log('INFO',df_model_ABsorting['HiLoPartition'].value_counts())

    #LoGroup=len(df_model_ABsorting[df_model_ABsorting['HiLoPartition']=='Lo']['HiLoPartition'])
    #HiGroup=len(df_model_ABsorting[df_model_ABsorting['HiLoPartition']=='Lo']['HiLoPartition'])

    #partsize_Lo=math.ceil(len(df_model_ABsorting[df_model_ABsorting['HiLoPartition']=='Lo'])/2)
    #partsize_Hi=math.ceil(len(df_model_ABsorting[df_model_ABsorting['HiLoPartition']=='Hi'])/2)

    #df_model_ABsorting['ABPartition']='B'
    #df_model_hi_A=df_model_ABsorting[df_model_ABsorting['HiLoPartition']=='Hi']['HiLoPartition'].sample(frac=0.5,random_state=42)
    #df_model_lo_A=df_model_ABsorting[df_model_ABsorting['HiLoPartition']=='Lo']['HiLoPartition'].sample(frac=0.5,random_state=42)
    #logger.log('INFO',f"LOWA PARTITION {len(df_model_lo_A)}")
    #df_model_ABsorting['ABPartition'].iloc[df_model_hi_A.index]='A'
    #df_model_ABsorting['ABPartition'].iloc[df_model_lo_A.index]='A'


    #df_part_lo=df_model_ABsorting.iloc[:partsize]
    #df_part_hi=df_model_ABsorting.iloc[partsize:]
    #logger.log('INFO',f"{[len(df_part_hi),len(df_part_lo),len(df_part_hi)+len(df_part_lo)]}")

    df_model_ABsorting.to_csv(outfile)

    return
    
    logger.log('INFO',"********************************************")
    logger.log('INFO',df_model_ABsorting['HiLoPartition'].value_counts(dropna=False))
    logger.log('INFO',df_model_ABsorting['ABPartition'].value_counts(dropna=False))



    for c in df_model_ABsorting.columns:

        df_hi_A=df_model_ABsorting[(df_model_ABsorting['ABPartition']=='A') & (df_model_ABsorting['HiLoPartition']=='Hi')]
        df_lo_A=df_model_ABsorting[(df_model_ABsorting['ABPartition']=='A') & (df_model_ABsorting['HiLoPartition']=='Lo')]

        df_hi_B=df_model_ABsorting[(df_model_ABsorting['ABPartition']=='B') & (df_model_ABsorting['HiLoPartition']=='Hi')]
        df_lo_B=df_model_ABsorting[(df_model_ABsorting['ABPartition']=='B') & (df_model_ABsorting['HiLoPartition']=='Lo')]
    

        #if(plot):
        plotdir="../Plots/ABTEST/"

        if(plots):
            #if('int' not in df_model_ABsorting[c].dtype and 'float' not in df_model_ABsorting[c].dtype):
            #    logger.log('INFO',c,df_model_ABsorting[c].dtype)
            #    continue
            logger.log('INFO',"Plotting: ",c)
            logger.log('INFO',len(df_hi_A),len(df_lo_A),len(df_hi_A),len(df_hi_B),len(df_lo_B))
            try:
                plt.close('all')
                minval=df_model_ABsorting[c].min()
                maxval=df_model_ABsorting[c].max()
                rangeval=maxval-minval
                nbins=20
                binwidth=rangeval/nbins
                #if(rangeval<10e-4):
                #    continue
                binspec=np.arange(minval,maxval+binwidth,binwidth)
                
                plt.hist(df_hi_A[c],bins=binspec,label="Email A - High BDT score",stacked=False,histtype='step')
                plt.hist(df_lo_A[c],bins=binspec,label="Email A - Low BDT score",stacked=False,histtype='step')
                plt.hist(df_hi_B[c],bins=binspec,label='Email B - High BDT score',stacked=False,histtype='step')
                plt.hist(df_lo_B[c],bins=binspec,label='Email B - Low BDT score',stacked=False,histtype='step')
                plt.legend()
                colname=c.lower().replace(" ","_")
                plt.xlabel(colname)
                plt.ylabel("Freq.")
                
                plt.savefig(plotdir+"ABPartitionTest"+colname+".png")
            except Exception as e:
                logger.log('INFO',"Exception :{}".format(e))
                logger.log('INFO',"Column: {}".format(c))
                


                


    plt.close('all')

    dict_variables_t=model_t.feature_importances_
    logger.log('INFO',dict_variables_t)

    #sorted_idx = np.argsort(model_t.feature_importances_)[::-1]
    #for index in sorted_idx:
    #    logger.log('INFO',[x_train_t.columns[index], model_t.feature_importances_[index]])





    df_model_allout=df_model_allout[['Predicted Withdrawal','Actual Withdrawal','Level']]


    #df_grouped_t=df_testset_t.groupby(by=['Predicted Withdrawal','Actual Withdrawal'])['Level'].count().reset_index(name="nApprentices")
    #logger.log('INFO',df_grouped_t)
    #pivot_table_t=df_grouped_t.pivot_table(index='Predicted Withdrawal',columns='Actual Withdrawal',values='nApprentices',aggfunc="sum")
    #logger.log('INFO',pivot_table_t.columns)
    #logger.log('INFO',pivot_table_t)


    #metrics 
    #total_entries=len(df_testset_t)
    #logger.log('INFO',"TOTAL",total_entries)
    #pivot_table_t.to_csv("./Plots/MVAconfig/pivot_t_alllvls_inclgeodata_economics_ECON.csv")
    #Accuracy = (TP + TN) / (TP+TN+FP+FN) = (TP+TN)/2*TOT

    #TP=pivot_table_t[1][1]
    #TN=pivot_table_t[0][0]

    #logger.log('INFO',"TP: {}".format(TP))
    #logger.log('INFO',"TN: {}".format(TN))
    #FP=pivot_table_t[0][1]
    #logger.log('INFO',"FP: {}".format(FP))
    #FN=pivot_table_t[1][0]
    #logger.log('INFO',"FN: {}".format(FN))

    #Acc=100*(TP+TN)/ (TP+TN+FP+FN)
    #logger.log('INFO',"SUM TP + TN +FP + FN: {}".format(TP+TN+FP+FN))


    #Classification error rate

    #ClassError=100*(FP+FN)/(TP+TN+FP+FN)
    #Precision=100*(TP)/(TP+FP)
    #Sensitivity=100*(TP)/(TP+FN)
    #Specificity=100*(TN)/(TN+FP)
    #Acc2=100*(TP+TN)/(total_entries)

    #logger.log('INFO',"ACCURACY: {}".format(round(Acc,2)))
    #logger.log('INFO',"ClassificationError: {}".format(round(ClassError,2)))
    #logger.log('INFO',"Precision: {}".format(round(Precision,2)))
    #logger.log('INFO',"Sensitivity: {}".format(round(Sensitivity,2)))
    #logger.log('INFO',"Specificity: {}".format(round(Specificity,2)))
    from sklearn.metrics import accuracy_score
    df_model_allout=df_model_allout[df_model_allout['Actual Withdrawal'].notna()].copy()
    df_model_allout=df_model_allout[df_model_allout['Actual Withdrawal']>=0].copy()
    #logger.log('INFO',df_model_allout[['Actual Withdrawal','Predicted Withdrawal']].value_counts())
    acc_xgb=accuracy_score(y_true=df_model_allout['Actual Withdrawal'] ,y_pred=df_model_allout['Predicted Withdrawal'] )
    logger.log('INFO',"XGB ACC calc (Test set): {}".format(acc_xgb))

    if(RunMemCheck):
        print("MEMORY CHECKPOINT: END OF JOB")
        memtracker.print_diff()
    return

    metrics=(model_t.evals_result())
    logger.log('INFO',metrics)
    logger.log('INFO',)
    # plot learning curves
    plt.close('all')
    plt.plot(metrics['validation_0']['logloss'], label='train_lvl{}'.format("ALL"))
    plt.plot(metrics['validation_1']['logloss'], label='test_lvl{}'.format("ALL"))
    plt.ylabel("Validation logloss")
    plt.xlabel("Iterations")
    # show the legend
    plt.legend()
    #plt.savefig("./Plots/MVAconfig/ROC_BDT_timevarying_allLvls_inclGeoData_ECON.png")
    #plt.show()

    plt.close("all")

    xgb.plot_importance(model_t)
    plt.tight_layout()

    #plt.savefig("./Plots/MVAconfig/RelImportance_timevarying_startordered_alllvls_inclGeoData_ECON.png")
    #plt.show()
    plt.close('all')


    df_x=df_tmp.copy()
    #purge any nulls/nans lingering around as the LogRegressor won't like them.
    df_x=df_x.dropna()


    #x=df_x[x_prof.columns]
    #y=df_x['Withdraw'] 



    #x_train_t, x_test_t, y_train_t, y_test_t = train_test_split(x, y,test_size=0.2,shuffle=False)
    #from sklearn.linear_model import LogisticRegression
    #from sklearn.preprocessing import StandardScaler
    #from sklearn.pipeline import make_pipeline
    #pipe=make_pipeline(StandardScaler(),LogisticRegression())
    #logger.log('INFO',pipe)
    #try:
    #    pipe.fit(x_train_t,y_train_t)

        #clf=LogisticRegression(random_state=0).fit(x_train_t,y_train_t)
        ##y_pred=clf.predict(x_test_t)
        #logregacc=accuracy_score(y_pred=y_pred,y_true=y_test_t)
    #    logregacc=pipe.score(x_test_t,y_test_t)
    #    logger.log('INFO',"Log Reg accuracy: {}".format(logregacc))
    #except: 
    #    logregacc=0

    # %%
    from shap import TreeExplainer, summary_plot
        
    #bkg=df_modelinput[np.random.choice(df_modelinput.shape,20,replace=False)]
    #bkg=df_modelinput_noPredVariables.sample(5000,random_state=420)
    bkg=df_modelinput_noPredVariables.sample(500,random_state=42)
    import sklearn_pandas as skpd
    logger.log('INFO',list(bkg.columns))    
    # stick columns back on it
    df_reco=bkg
        
    explainer=TreeExplainer(model_t,df_reco)
    shap_values=explainer.shap_values(df_reco,
                                    check_additivity=False
                                    )
    plt.close('all')

    summary_plot(shap_values,features=df_reco,feature_names=df_reco.columns,plot_type='bar',show=False,
                max_print=40
                )
    #plt.tight_layout()
    plt.gcf()
    shpctr=0
    for feature in list(df_reco.columns):
        logger.log('INFO',feature,)
        shpctr+=1
    #plt.savefig("./Plots/MVAconfig/SHAP_BDT_GEODATA_ECON.png")
    #plt.show()
    plt.close("all")


    for l in range(2,8):
        df_modelinput_noPredVariables_lvl=df_modelinput_noPredVariables[df_modelinput_noPredVariables['Level']==l].copy()

        df_reco=df_modelinput_noPredVariables_lvl.sample(500,random_state=42)

        explainer=TreeExplainer(model_t,df_reco)
        shap_values=explainer.shap_values(df_reco,
                                    check_additivity=False
                                    )
        plt.close('all')

        summary_plot(shap_values,features=df_reco,feature_names=df_reco.columns,plot_type='bar',show=False,
                max_print=40
                )
        plt.title("SHAP VALUES FOR Level {} apprentices specifically".format(l))
        plt.gcf()
        
        plt.show()
        plt.close('all')

    # %%
    #from  ydata_profiling import ProfileReport

    #rep=ProfileReport(df_profile_tmp.copy(),minimal=True)
    #rep.to_file("BDT_INPUT.html")

    # %% [markdown]
    # 

    # %%
    df_model_allout=pd.concat([df_modelinput,df_modeloutput])

    logger.log('INFO',df_model_allout)

    l_cols_red=list(df_model_allout.columns)
    l_cols_red.remove('Actual Withdrawal')
    l_cols_red.remove('Predicted Withdrawal')

    logger.log('INFO',l_cols_red)


    # %% [markdown]
    # ## Profile the outputs

    # %%

    #from  ydata_profiling import ProfileReport

    #rep=ProfileReport(df_model_allout,minimal=True)
    #rep.to_file("BDT_OUTPUT.html")

    # %%
    l_cols_red

    import random
    random.seed(42)
    ct=0
    #for p in range(0,4):
    #    randomlist=random.sample(l_cols_red,15)
    #    logger.log('INFO',randomlist)
    #    try:
    #        randomlist.remove('unixtimediff_exp_end')
    #    except:
    #        pass
    #
    #    rep=ProfileReport(df_model_allout[randomlist+['Actual Withdrawal','Predicted Withdrawal']].copy())
    #    rep.to_file("./HTML/BDT_OUTPUT_{}.html".format(ct))
    #    ct+=1
    try:
        del range
    except:
        pass
    try:
        del max
    except: 
        pass
    try:
        del min
    except: pass

    # %%
    logger.log('INFO',range(2,3))

    # %%
    #jf=open("./CSV/BDTAccScores_Geodata_ECON.json","w+")
    from sklearn.metrics import precision_score,recall_score
    from scipy import stats as scipystats
    import json
    logger.log('INFO',"ACCURACY: {}".format(round(Acc,1)))
    #logger.log('INFO',"ClassificationError: {}".format(round(ClassError,2)))
    logger.log('INFO',"Precision: {}".format(round(Precision,1)))
    logger.log('INFO',"Sensitivity: {}".format(round(Sensitivity,1)))
    logger.log('INFO',"Specificity: {}".format(round(Specificity,1)))

    dict_bdt_results={}
    dict_bdt_results['GLOBAL']={
        'Accuracy':round(Acc,1),
        'Precision':round(Precision,1),
        'recall':round(Sensitivity,1),
        'logregacc':-1}

    ct=range(2,8)
    logger.log('INFO',ct)

    for itr in range(2,8):
        #logger.log('INFO',itr)
        tempdf=df_modeloutput[df_modeloutput['Level'] ==itr ].copy()
        tempmodelinput_df=df_modelinput[df_modelinput['Level']==itr].copy()
        
        y_true=tempdf['Actual Withdrawal']
        y_pred=tempdf['Predicted Withdrawal']
        logger.log('INFO',"***************")
        logger.log('INFO',"LEVEL: {}".format(itr))
        acc_score=round(accuracy_score(y_true=y_true,y_pred=y_pred)*100,1)
        precision=round(precision_score(y_true,y_pred)*100,1)
        recall=round(recall_score(y_true,y_pred)*100,1)

        logger.log('INFO',"ACCURACY: {} (%)".format(acc_score))
        logger.log('INFO',"PRECISION: {} (%)".format(precision))
        logger.log('INFO',"RECALL: {} (%)".format(recall))
        

        
        tempdf_nonulls=tempdf.dropna()
        tempmodelinput_df=tempmodelinput_df.dropna()
        cols_to_keep=[x for x in tempdf.columns if "Withdrawal" not in x]
        logger.log('INFO',cols_to_keep)
        testset_x=tempdf_nonulls[cols_to_keep]
        testset_y=tempdf_nonulls['Actual Withdrawal']
        trainset_x=tempmodelinput_df[cols_to_keep]
        trainset_y=tempmodelinput_df['Actual Withdrawal']
        try:
            #logregacc=round(pipe.score(testset_x,testset_y)*100,1)
            logregacc=0
        except:
            logregacc=0
        
        dict_bdt_results['level_{}'.format(itr)]={
            'Accuracy':
            acc_score,
            'Precision':precision,
            'recall':recall,
            'nrows':len(tempdf),
            'logregacc':logregacc}
        
        #plt.close('all')
        #plt.hist(tempdf['ExpectedCourseDuration'],label='Level_{}'.format(itr))
        #plt.legend()
        #plt.xlabel("Expected course duration (months)")
        #plt.ylabel("Number of Apprentices at given level")
        #plt.savefig("../Plots/MVAconfig/LVL_ExpectedDuration_Level{}_inclGeoData.png".format(itr))

        


        #bkg=trainset_x.sample(100)
        #import sklearn_pandas as skpd
        
        ## stick columns back on it
        #df_reco=bkg

        #explainer=TreeExplainer(model_t,df_reco)
        #shap_values=explainer.shap_values(df_reco)
        #plt.close('all')

        #summary_plot(shap_values,features=df_reco,feature_names=df_reco.columns,plot_type='bar',show=False,max_print=40)
        #plt.tight_layout()
        #plt.gcf()

        #plt.savefig("../Plots/MVAconfig/SHAP_BDT_GEODATA_ECON_LVL{}.png".format(itr))
        #plt.close('all')

    #df_L2_results=df_modeloutput[df_modeloutput["Level of Apprenticeship"]==2]
    #df_L3_results=df_modeloutput[df_modeloutput["Level of Apprenticeship"]==3]
    #df_L4_results=df_modeloutput[df_modeloutput["Level of Apprenticeship"]==4]
    #json.dump(dict_bdt_results,jf,indent=4,sort_keys=True)
    #jf.close()


    # now look at what the BDT pulled out

    df_modeloutput_completed=df_modeloutput[df_modeloutput['Predicted Withdrawal']==0]
    df_modeloutput_withdrew=df_modeloutput[df_modeloutput['Predicted Withdrawal']==1]

    #df_model_allout_completed=pd.concat(df_modeloutput,df_modelinput,axis=1)
    logger.log('INFO',"LENGTHCHECK_2",len(df_modelinput),len(df_modeloutput),len(df_model_allout))
    logger.log('INFO',df_model_allout.columns)
    df_model_allout_completed=df_model_allout[df_model_allout['Predicted Withdrawal']==0]
    df_model_allout_withdrew=df_model_allout[df_model_allout['Predicted Withdrawal']==1]


    logger.log('INFO',df_model_allout_completed['Level'])



    for col in df_modeloutput.columns:
        if("ithdraw" in col.lower() or "Before" in col):
            continue
        logger.log('INFO',"Profileing variable plot for variable {}".format(col))
        binmax=df_modeloutput[col].max()
        binmin=df_modelinput[col].min()
        
        logger.log('INFO',binmin,binmax)

        nbins=20

        binwidth=(binmax -binmin)/nbins

        variable=col.lower()
        try:
            plt.close("all")
            plt.hist(df_modeloutput_completed[col],color="red",label="Pred. Completed",bins=np.arange(binmin,binmax+binwidth,binwidth),linestyle=":",histtype='step',stacked=True,density=True)
            plt.hist(df_modeloutput_withdrew[col],color="blue",label="Pred. Withdrawn",bins=np.arange(binmin,binmax+binwidth,binwidth),linestyle=":",histtype='step',stacked=True,density=True)
            plt.ylabel("Freq density")
            plt.xlabel(col)
            plt.legend()
            #plt.savefig("./Plots/BDTProfiling/varplot_testset_{}.png".format(variable))
        
            plt.close("all")
        except:
            logger.log('INFO',col)
            logger.log('INFO',"SKIPPED")
            continue

        logger.log('INFO',"Made plot")

        plt.close('all')


        binmax=max(df_model_allout_completed[col].max(),df_model_allout_withdrew[col].max())
        binmin=min(df_model_allout_withdrew[col].min(),df_model_allout_completed[col].min())
        
        logger.log('INFO',"ALL DATA, NOT TEST SET",binmin,binmax)

        nbins=20

        binwidth=(binmax -binmin)/nbins
        try:
            binspec=np.arange(binmin,binmax+binwidth,binwidth)
        except:
            logger.log('INFO',"Variable {} skipped as bin range = 0".format(col))
            continue

        if("unixtime" in col):
            binspec=np.arange(-72,72,12)
        if(col=="Course_Duration"):
            binspec=np.arange(0,72,1)
        logger.log('INFO',col)
        plt.hist(df_model_allout_completed[col],color="red",label="Pred. Completed",bins=binspec,density=True,linestyle=":",histtype='step',stacked=True)
        plt.hist(df_model_allout_withdrew[col],color="blue",label="Pred. Withdrawn",bins=binspec,density=True,linestyle=":",histtype='step',stacked=True)
        plt.ylabel("Freq density")
        if("unixtime" in col):
            if("exp_end" in col):
                plt.xlabel("n Months from Expected End date to 23/3/2020")
            else:
                plt.xlabel("n Months from Start date to 23/3/2020")

        else:    
            plt.xlabel(col)
        plt.legend()
        #plt.savefig("./Plots/BDTProfiling/varplot_alldata_{}.png".format(variable))


        #logger.log('INFO',df_model_allout_withdrew[col])
        #logger.log('INFO',df_model_allout_completed[col])
        logger.log('INFO',"LENGTHCHECK: {} ,{}, {}, {}, {}".format(col,len(df_model_allout_completed),len(df_model_allout_withdrew),len(df_modeloutput_completed),len(df_modeloutput_withdrew)))
        plt.close('all')
        plt.hist(df_model_allout_completed[col],color="red",label="Pred. Completed",bins=binspec,linestyle=":",histtype='step',stacked=True)
        plt.hist(df_model_allout_withdrew[col],color="blue",label="Pred. Withdrawn",bins=binspec,linestyle="--",histtype='step',stacked=True)
        plt.ylabel("Freq.")
        
        plt.xlabel(col)
        plt.legend()

        
        ks_value=scipystats.ks_2samp(df_model_allout_completed[col].sort_values(),df_model_allout_withdrew[col].sort_values())
        logger.log('INFO',"KS VALUE FOR COL {}: KS SCORE: {} KS PVALUE: {}".format(col,ks_value.statistic,ks_value.pvalue,4))

        title="K-S test score: {}, p-value: {}".format(round(ks_value.statistic,4),ks_value.pvalue)
        plt.title(title)
        #plt.savefig("./Plots/BDTProfiling/varplot_alldata_{}_nodensity.png".format(variable))
        
        




    # %%
    startfield='unixtimediff_start'
    endfield='unixtimediff_end'

    # profile the startfield,endfield arrays to see whether they're buggy

    logger.log('INFO',"MODEL INPUT MODEL OUTPUT")
    logger.log('INFO',"TEST SET {} TRAIN SET: {}, TOTAL: {}".format(len(df_modeloutput),len(df_modelinput),len(df_modelinput)+len(df_modeloutput)))
    logger.log('INFO',"MERGED ROWS: {}".format(len(df_model_allout_completed)))



    df_allout_completed_true=df_model_allout_completed[df_model_allout_completed['Actual Withdrawal']==0]
    df_allout_completed_false=df_model_allout_completed[df_model_allout_completed['Actual Withdrawal']==1]

    df_allout_withdrawn_true=df_model_allout_withdrew[df_model_allout_withdrew['Actual Withdrawal']==1]

    df_allout_withdrawn_false=df_model_allout_withdrew[df_model_allout_withdrew['Actual Withdrawal'] ==0]
    logger.log('INFO',df_allout_withdrawn_false.columns)
    for column in df_allout_withdrawn_true.columns:
        if("Actual Withdrawal" in column.lower()):
            continue
        if("ithdraw" in column.lower()):
            continue

        logger.log('INFO',"Processing {}".format(column))
        binmax=max(df_model_allout_completed[column].max(),df_model_allout_withdrew[column].max())
        binmin=min(df_model_allout_withdrew[column].min(),df_model_allout_completed[column].min())
        binwidth=(binmax -binmin)/nbins
        try:
            bs=np.arange(binmin,binmax+binwidth,binwidth)
            if (binwidth<1e-4):
                continue
        except:
            continue
        plt.close("all")
        plt.hist(df_allout_withdrawn_true[column],color="red",label="True Positive",bins=np.arange(binmin,binmax+binwidth,binwidth),stacked=True,density=True,histtype='step')
        plt.hist(df_allout_withdrawn_false[column],color="blue",label="False Positive",bins=np.arange(binmin,binmax+binwidth,binwidth),stacked=True,density=True,histtype='step')

        plt.hist(df_allout_completed_true[column],color="green",label="True Negative",bins=np.arange(binmin,binmax+binwidth,binwidth),stacked=True,density=True,histtype='step')
        plt.hist(df_allout_completed_false[column],color="black",label="False Negative",bins=np.arange(binmin,binmax+binwidth,binwidth),stacked=True,density=True,histtype='step')
    
        plt.ylabel("Freq density")
        
        plt.xlabel(column)
        plt.legend()
        plt.title(column)
        #plt.savefig("./Plots/BDTProfiling/AccuracyMetrics/diagnostic_alldata_metrics_{}.png".format(column.lower().replace(" ","_")))
        #plt.show()
        plt.close("all")

        # no density plot

        plt.close("all")
        plt.hist(df_allout_withdrawn_true[column],color="red",label="True Positive",bins=np.arange(binmin,binmax+binwidth,binwidth),histtype='step')
        plt.hist(df_allout_withdrawn_false[column],color="blue",label="False Positive",bins=np.arange(binmin,binmax+binwidth,binwidth),histtype='step')

        plt.hist(df_allout_completed_true[column],color="green",label="True Negative",bins=np.arange(binmin,binmax+binwidth,binwidth),histtype='step')
        plt.hist(df_allout_completed_false[column],color="black",label="False Negative",bins=np.arange(binmin,binmax+binwidth,binwidth),histtype='step')
    
        plt.ylabel("Frequency")
        
        plt.xlabel(column)
        plt.legend()
        plt.title(column)
        #plt.savefig("./Plots/BDTProfiling/AccuracyMetrics/diagnostic_alldata_metrics_{}_nodensity.png".format(column.lower().replace(" ","_")))
        #plt.show()
        plt.close("all")


    #from matplotlib import colors 
    #plt.close("all")
    #plt.hist2d(df_model_allout['unixtimediff_start'],df_model_allout['unixtimediff_exp_end'],bins=20,cmap="gray",norm=colors.LogNorm())
    #plt.xlabel("start time")
    #plt.ylabel("Academic Year")
    ##plt.zlabel("Frequency")
    #plt.savefig("../Plots/BDTProfiling/AccuracyMetrics/DIAGNOSIS_BIVARIATE_startend.png")
    #plt.close('all')

    #plt.hist(df_model_allout['Academic Year'],bins=np.arange(201415,202425,10000))
    #plt.xlabel("Academic Year")
    #plt.ylabel("Frequency.")
    #plt.savefig("../Plots/BDTProfiling/AccuracyMetrics/DIAGNOSTIC_ACADEMIC_YEAR.png")
    #plt.close('all')


    # %%
    logger.log('INFO',df_model_allout[['ULN','ApprenticeshipId','CompletionStatus','Withdraw','Level']])

    df_discrete_levels=df_model_allout[(df_model_allout['Level']==2)
                                    |(df_model_allout['Level']==3)
                                    |(df_model_allout['Level']==4)
                                    |(df_model_allout['Level']==5)
                                    |(df_model_allout['Level']==6)
                                    |(df_model_allout['Level']==7)
                                    ].copy()

    # %%
    df_cts_levels=df_model_allout.drop(index=df_discrete_levels.index).copy()

    logger.log('INFO',df_cts_levels)

    logger.log('INFO',len(df_cts_levels)*100/len(df_model_allout))

    for c in df_model_allout:
        try:
            maximum=df_model_allout[c].max()
        
            minimum=df_model_allout[c].min()
            TheRange=maximum-minimum
            logger.log('INFO',c,maximum,minimum)
        except:
            continue
        try:
            Q1=df_model_allout[c].quantile(0.25)
            Q3=df_model_allout[c].quantile(0.75)
            IQR=Q3-Q1
            logger.log('INFO',c,Q1,Q3,IQR)
        except:
            continue
        logger.log('INFO',"")
        pass


    # %%
    df_edgecase=df_model_allout[df_model_allout['unixtimediff_exp_end']>100][['ULN','ApprenticeshipId','unixtimediff_exp_end']]
    logger.log('INFO',df_edgecase)
    logger.log('INFO',len(df_edgecase)*100/len(df_model_allout))

    # %%
    df_model_allout    

    # %%

    for l in range(2,8):
        logger.log('INFO',l)
        df_lvl=df_model_allout[df_model_allout['Level']==l].copy()
        
        df_model_lvl_completed=df_lvl[df_lvl['Predicted Withdrawal']==0].copy()
        df_model_lvl_withdrew=df_lvl[df_lvl['Predicted Withdrawal']==1].copy()



        df_lvl_completed_true=df_model_lvl_completed[df_model_lvl_completed['Actual Withdrawal']==0].copy()
        df_lvl_completed_false=df_model_lvl_completed[df_model_lvl_completed['Actual Withdrawal']==1].copy()

        df_lvl_withdrawn_true=df_model_lvl_withdrew[df_model_lvl_withdrew['Actual Withdrawal']==1].copy()

        df_lvl_withdrawn_false=df_model_lvl_withdrew[df_model_lvl_withdrew['Actual Withdrawal'] ==0].copy()
        logger.log('INFO',df_allout_withdrawn_false.columns)
        for column in df_allout_withdrawn_true.columns:
            if("Actual Withdrawal" in column.lower()):
                continue
            if("ithdraw" in column.lower()):
                continue

            logger.log('INFO',"Processing {}".format(column))
            binmax=max(df_model_allout_completed[column].max(),df_model_allout_withdrew[column].max())
            binmin=min(df_model_allout_withdrew[column].min(),df_model_allout_completed[column].min())
            binwidth=(binmax -binmin)/nbins
            if(column=="Course_Duration"):
                binmax=96
                binmin=0
                binwidth=1
            try:
                bs=np.arange(binmin,binmax+binwidth,binwidth)
                if (binwidth<1e-4):
                    continue
            except:
                continue
            plt.close("all")
            plt.hist(df_lvl_withdrawn_true[column],color="red",label="True Positive",bins=np.arange(binmin,binmax+binwidth,binwidth),stacked=True,density=True,histtype='step')
            plt.hist(df_lvl_withdrawn_false[column],color="blue",label="False Positive",bins=np.arange(binmin,binmax+binwidth,binwidth),stacked=True,density=True,histtype='step')

            plt.hist(df_lvl_completed_true[column],color="green",label="True Negative",bins=np.arange(binmin,binmax+binwidth,binwidth),stacked=True,density=True,histtype='step')
            plt.hist(df_lvl_completed_false[column],color="black",label="False Negative",bins=np.arange(binmin,binmax+binwidth,binwidth),stacked=True,density=True,histtype='step')
        
            plt.ylabel("Freq density")
            
            plt.xlabel(column)
            plt.legend()
            plt.title(column +" Level {}".format(l))
            #plt.savefig("./Plots/BDTProfiling/AccuracyMetrics/LevelPlots/L{}/diagnostic_alldata_metrics_{}.png".format(l,column.lower().replace(" ","_")))
            #plt.show()
            plt.close("all")

            # no density plot

            plt.close("all")
            plt.hist(df_lvl_withdrawn_true[column],color="red",label="True Positive",bins=np.arange(binmin,binmax+binwidth,binwidth),histtype='step')
            plt.hist(df_lvl_withdrawn_false[column],color="blue",label="False Positive",bins=np.arange(binmin,binmax+binwidth,binwidth),histtype='step')

            plt.hist(df_lvl_completed_true[column],color="green",label="True Negative",bins=np.arange(binmin,binmax+binwidth,binwidth),histtype='step')
            plt.hist(df_lvl_completed_false[column],color="black",label="False Negative",bins=np.arange(binmin,binmax+binwidth,binwidth),histtype='step')
        
            plt.ylabel("Frequency")
            
            plt.xlabel(column)
            plt.legend()
            plt.title(column+" Level {}".format(l))
            #plt.savefig("./Plots/BDTProfiling/AccuracyMetrics/LevelPlots/L{}/diagnostic_alldata_metrics_{}_nodensity.png".format(l,column.lower().replace(" ","_")))
            #plt.show()
            plt.close("all")


    # %%
    df_model_allout.to_csv(outfile)



    # %%

    #df_timeseries=df_model_allout.copy()
    #df_timeseries['Percent_Duration_Completed']=(df_timeseries['ActualDuration'].multiply(100)/df_timeseries['Course_Duration'])

    #filter any weird timestamps,where it is negative, and also sort out any weird cases where the course was extended/delayed.
    #df_timeseries['Percent_Duration_Completed']=df_timeseries['Percent_Duration_Completed'].apply(lambda x: 0 if x<0 else (100 if x>100 else x ))
    # only discard cases if -ve (and get handle any infinite values - how is expected course duration 0?)

    #df_timeseries['Percent_Duration_Completed']=df_timeseries['Percent_Duration_Completed'].apply(lambda x: 0 if x<0 else (x if x<300 else 0) )
    #df_timeseries['PreCovid']=df_timeseries['EndDate'].apply(lambda x: 1 if x<pd.Timestamp("23-03-2020") else 0)
if __name__=="__main__":
    import argparse

    parser=argparse.ArgumentParser('GenerateBDT_Predictions.py')
    parser.add_argument('--infile',action='store',dest='infile',help='Input file (.csv)')
    parser.add_argument('--outfile',action='store',dest='outfile',help="Output File (.csv)")
    parser.add_argument("--p",action='store_true',default=False,dest='plots',help="Make diagnostic plots of A/B sample")
    parser.add_argument("--memdebug",action='store_true',default=False,help="Run Memory Profiler? (muppy https://pythonhosted.org/Pympler/muppy.html)",dest='memcheck')
    args=parser.parse_args()

    if(args.memcheck):
        RunBDTModel(args.infile,args.outfile,args.plots,pd.DataFrame(),args.memcheck)
    else:
        RunBDTModel(args.infile,args.outfile,args.plots,pd.DataFrame(),False)
