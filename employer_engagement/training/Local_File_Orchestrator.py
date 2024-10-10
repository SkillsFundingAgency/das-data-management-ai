
import Generate_BDT_Predictions_NOLocation as BDTCode
import pandas as pd
#pycmd="python .\Generate_BDT_Predictions_NOLocation.py --infile ..\..\..\Fake_Dataframe_Autoencoded.csv --outfile ..\..\..\Fake_Dataframe_BDTOutput.csv"

infile="../../../Fake_Dataframe_Autoencoded.csv"
outfile="../../../Fake_Dataframe_BDTOutput.csv"
pandasinput=pd.read_csv(infile,index_col=0)
BDTCode.RunBDTModel(infile="",outfile=outfile,plots=False,PandasInput=pandasinput)
