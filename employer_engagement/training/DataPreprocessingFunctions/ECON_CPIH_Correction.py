# Economic CPIH inflation calculation -> Correcting salary for inflation at start of apprenticeship
import pandas as pd
from pandas.tseries.offsets import MonthBegin
import numpy as np
import matplotlib.pyplot as plt




def GetInterveningMonths(refdate,benchdate):
    months=[]

    dr=pd.date_range(benchdate,refdate,freq='YS')
    #.strftime('%y-%m-%d').tolist()
    #print(dr)
    mths=pd.to_datetime(dr.values.astype('datetime64[M]')).tolist()
    #print("INTFN",mths)
    return mths 
    
def CorrectCPIH_Inflation(date=pd.Timestamp("2020-03-01"),wage=0,CPIH_figures=pd.Series(dtype="float64"),BenchDate=pd.Timestamp("2020-01-01"),rowindex=1,returnMultiplier=False,returnDivisor=False):
    if(rowindex %1000 ==0):
        print("Processing entry: {}".format(rowindex))
    if(date==BenchDate):
        # in weird case that we have an exact match, read off the date 
        return wage
    if(np.isnan(wage)):
        return wage
    months_CPIH=[]
    if(date>BenchDate):
        months_CPIH=GetInterveningMonths(date,BenchDate)
    else:
        months_CPIH=GetInterveningMonths(BenchDate,date)
    #print("MONTHS_CPIH",months_CPIH)
    inflation_vals=[]
    compound_multiplier=1.0
    compound_divisor=1.0
    for mth in months_CPIH:
        try:
            inflation_val=CPIH_figures[CPIH_figures['tmp_date']==mth]['cpih_inflation'].values[0]
        except: 
            inflation_val=1.0 # if null entry for some reason, skip and set to 1
        inflation_vals.append(inflation_val)
        #print(mth,inflation_val)
        #inflation is typically stated as a percentage, reduce to fraction
        frac_inflation=inflation_val/100
        compound_divisor=compound_divisor*(1/(1+frac_inflation))
        compound_multiplier=compound_multiplier*(1+frac_inflation)

        #print("Multiplier: ",compound_multiplier)
        #print("DIVISOR: ",compound_divisor)
        #print("SALARY: ",wage)
    if(returnMultiplier):
        return compound_multiplier
    

    else:        
        if(date>BenchDate):
            return wage*compound_divisor
        else:
            return wage*compound_multiplier




def ProcessSalaryData(indf=pd.DataFrame(),UseASData=False,updated_CPIH_Figs=False,logger=None):
    import os
    dirname=os.path.dirname(__file__)+"/"
    df_DB=indf.copy(deep=True)
    if(updated_CPIH_Figs==False):
        df_ECON_CPIH=pd.read_csv(dirname+"../ML_Models/ONSData/calculated_features.csv")
    else:
        df_ECON_CPIH=pd.read_csv(dirname+"../ML_Models/ONSData/CPIH_figures.csv",index_col=0)
        


    #for temp fix, reduce df_DB to 20 entries
    #df_DB=df_DB.dropna()
    #df_DB=df_DB.head(5)
    df_ECON_CPIH['tmp_date']=pd.to_datetime(df_ECON_CPIH['date'].values.astype('datetime64[M]'))
    print(list(df_ECON_CPIH.columns))


    #print(df_ECON_CPIH['cpih_inflation'])

    #print(df_DB['Learner start date'])


    if(UseASData):
        logger.log('INFO',"USING AS DATA")
        df_DB['Learner start date']=df_DB['StartDate']
    df_DB['tmp_startdate']=pd.to_datetime(df_DB['Learner start date'].values.astype('datetime64[M]'))

        
    date="2020-03-01"
    salary=1000
    #logger.log('INFO',"***************")
    #print(GetInterveningMonths(date,"2019-05-01"))


    #print("+++++++++++++++++++++")

    CorrectCPIH_Inflation(pd.Timestamp(date),salary,df_ECON_CPIH,pd.Timestamp("2020-03-01"))
    #print([x for x in list(df_DB.columns) if "wage" in x])

    logger.log('INFO',"Test correction complete so lib works")
    import time
    start=time.time()

    df_DB['Corr_weightedaverage_minwage']=df_DB.apply(lambda x: 
                                                    CorrectCPIH_Inflation(
                                                        x['tmp_startdate'],
                                                        x['weighted_average_annual_minwage'],
                                                        df_ECON_CPIH,
                                                        pd.Timestamp("2020-03-01"),

                                                        ),
                                                        axis=1)

    it1=time.time()
    logger.log('INFO',"ITERATION 1 FINISHED IN : {}".format(it1-start))

    df_DB['Corr_weightedaverage_maxwage']=df_DB.apply(lambda x: 
                                                    CorrectCPIH_Inflation(
                                                        x['tmp_startdate'],
                                                        x['weighted_average_annual_maxwage'],
                                                        df_ECON_CPIH,
                                                        pd.Timestamp("2020-03-01"),
                                                        ),axis=1)

    it2=time.time()
    logger.log("ITERATION 2 FINISHED IN :{} ".format(it2-it1))
    logger.log("TIME SINCE START: {}".format(it2-start))
    
    for c in list(df_DB.columns):
        if("wage" in c):
            print(c)

    #print(df_DB[['Learner start date','Corr_weightedaverage_maxwage','Corr_weightedaverage_minwage','weighted_average_annual_maxwage','weighted_average_annual_minwage']])




    #dr=pd.date_range('2014-10-10','2016-01-07', 
    #              freq='MS').strftime('%y-%m-%d').tolist()
    #print(dr)


    #plt.close("all")
    #bins=np.linspace(0,40000,40)

    #plt.hist(df_DB['weighted_average_annual_maxwage'],bins=bins,color="r",linestyle="--",histtype='step',label="No CPIH correction")
    #plt.hist(df_DB['Corr_weightedaverage_maxwage'],bins=bins,color="b",linestyle="--",histtype='step',label="CPIH corrected")
    #plt.xlabel("Weighted average annual max wage")
    #plt.ylabel("Freq.")
    #plt.legend()
    #try:
    #    plt.savefig("..\\Plots\\PCAConfig\\CPI_CORR\\Salary_Profile_Max{}.png".format("ASDATA" if UseASData else ""))
    #except:
    #    pass

    #plt.close("all")
    #bins=np.linspace(0,40000,40)

    #plt.hist(df_DB['weighted_average_annual_minwage'],bins=bins,color="r",linestyle="--",histtype='step',label="No CPIH correction")
    #plt.hist(df_DB['Corr_weightedaverage_minwage'],bins=bins,color="b",linestyle="--",histtype='step',label="CPIH corrected")
    #plt.xlabel("Weighted average annual min wage")
    #plt.ylabel("Freq.")
    #plt.legend()
    #try:
    #    plt.savefig("..\\Plots\\PCAConfig\\CPI_CORR\\Salary_Profile_Min{}.png".format("ASDATA" if UseASData else ""))
    #except:
    #    pass
    return df_DB

if __name__=="__main__":
    import os
    try:
        os.chdir("Python/")
    except:
        pass
    df_DB=pd.read_csv("../CSV/Processed_CSVs/TestCSV_Imputation.csv",index_col=0)
    
    df_out=ProcessSalaryData(df_DB)
    df_out.to_csv("../CSV/Processed_CSVs/TestCSV_Imputation_CPIHCorr.csv")

