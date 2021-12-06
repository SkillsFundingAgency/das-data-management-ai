import pickle
import os
import pandas as pd
import pyarrow.parquet as pq
from azureml.core.run import Run
from azureml.core import Dataset, Datastore, Model
from azureml.data.datapath import DataPath
from sklearn.linear_model import LogisticRegression

aml_workspace = Run.get_context().experiment.workspace
# datastore = Datastore.get(aml_workspace, datastore_name='trainingdata')

datastore = Datastore.get(aml_workspace, datastore_name='datamgmtdb')
query = DataPath(datastore, 'SELECT * FROM Stg.AI_TestData')
tabular = Dataset.Tabular.from_sql_query(query, query_timeout=10)
df = tabular.to_pandas_dataframe()
x_train=df[['dep_var2','dep_var3']]

# os.makedirs('./outputs', exist_ok=True)

# load the model from disk
# loaded_model = pickle.load(open('./outputs/test_model.pkl','rb'))

# model_path = os.path.join('outputs')
global loaded_model

model_path = Model.get_model_path('test_model')
loaded_model = joblib.load(model_path)

#score dataset back onto the base
scored=loaded_model.predict_proba(x_train)
df['model_prediction']=scored[:,1]
print(df.head(5))

#print model coefficients
print(loaded_model.coef_)
print(loaded_model.intercept_)

run = Run.get_context()
run.log('test log', 'test log')

#write out scored file to parquet
df.to_parquet('./outputs/test_scored_model_scoring.parquet')