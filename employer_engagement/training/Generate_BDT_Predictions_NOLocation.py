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
    df_model_ABsorting['Email_Classification']='A'
    df_pred_wdraw=df_model_ABsorting[df_model_ABsorting['Predicted Withdrawal']==1]
    df_pred_complete=df_model_ABsorting[df_model_ABsorting['Predicted Withdrawal']==0]

    
    emailBalloc_wdraw=df_pred_wdraw['Email_Classification'].sample(frac=0.5,random_state=42)
    emailBalloc_complete=df_pred_complete['Email_Classification'].sample(frac=0.5,random_state=42)
    
    df_model_ABsorting['Email_Classification'].iloc[emailBalloc_wdraw.index]='B'
    df_model_ABsorting['Email_Classification'].iloc[emailBalloc_complete.index]='B'

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
    
    df_model_ABsorting[['ApprenticeshipId','Email_Classification']].to_csv(outfile)
    
    #output to datamart
    df_datamart_output=df_model_ABsorting[['ApprenticeshipId','Predicted Withdrawal','BDT_PROB_WITHDRAW']]
    df_datamart_output=df_datamart_output.rename(columns={'BDT_PROB_WITHDRAW':'BDT_proba'})
    #force INT on Withdraw/Complete
    df_datamart_output['BDT_prediction']=df_datamart_output['Predicted Withdrawal'].astype(int)
    df_datamart_output['ApprenticeshipId']=df_datamart_output['ApprenticeshipId'].astype(int)
    df_datamart_output['BDT_prediction_description']=df_datamart_output['BDT_prediction'].apply(lambda x: "Predict withdraw" if x>0 else "Predict complete")

    df_datamart_output[['ApprenticeshipId','BDT_prediction','BDT_proba','BDT_prediction_description']].to_csv(outfile.replace(".csv","_DATAMART_INPUT.csv"))

    logger.log("INFO","BDT JOB FINISHED, RETURNING COMPLETE")
    return
    
    
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
