import pickle
import os
import pandas as pd
import pyarrow.parquet as pq
import joblib
from azureml.core.run import Run
from azureml.core import Dataset, Datastore, Model
from azureml.data.datapath import DataPath
from sklearn.linear_model import LogisticRegression

# Set up config of workspace and datastore
aml_workspace = Run.get_context().experiment.workspace
datastore = Datastore.get(aml_workspace, datastore_name='datamgmtdb')

# Remove parquet file as a starting point
# rm_file = 'test_scored_model_scoring3.parquet'
# rm_location = "./outputs"
# Path
# rm_path = os.path.join(rm_location, rm_file)
# os.remove(rm_path)

# Create model scoring data into x_train dataframe
query = DataPath(datastore, 'SELECT * FROM Stg.AI_TestData')
tabular = Dataset.Tabular.from_sql_query(query, query_timeout=10)
df = tabular.to_pandas_dataframe()
x_train=df[['dep_var2','dep_var3']]

# load registered model 
global loaded_model

model_path = Model.get_model_path('test_model')
loaded_model = joblib.load(model_path)

#score dataframe using saved model onto the base
scored=loaded_model.predict_proba(x_train)
df['model_prediction']=scored[:,1]

#print model coefficients to check correct model is in place (testing only)
print(loaded_model.coef_)
print(loaded_model.intercept_)

run = Run.get_context()
run.log('test log', 'test log')

#write out scored file to parquet
df.to_parquet('./outputs/test_scored_model_scoring3.parquet')