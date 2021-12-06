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

# Create model build data into df dataframe
query = DataPath(datastore, 'SELECT * FROM Stg.AI_TestData')
tabular = Dataset.Tabular.from_sql_query(query, query_timeout=10)
df = tabular.to_pandas_dataframe()

#split data into target and independent sets
y_train=df['target']
x_train=df[['dep_var2','dep_var3']]

#build logistic model
clf=LogisticRegression(random_state=0).fit(x_train,y_train)

#print model coefficients
print(clf.coef_)
print(clf.intercept_)

run = Run.get_context()
run.log('test log', 'test log')

# Save the trained model in the outputs folder
print("Saving model...")
os.makedirs('outputs', exist_ok=True)
model_file = os.path.join('outputs', 'test_model_XX.pkl')
joblib.dump(value=clf, filename=model_file)

# Register the model to pick up in scoring
print('Registering model...')
Model.register(workspace=aml_workspace,
               model_path = model_file,
               model_name = 'test_model')