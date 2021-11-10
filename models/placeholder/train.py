import pickle
import os
import pandas as pd
from azureml.core.run import Run
from azureml.core import Datastore
from azureml.data.datapath import DataPath

aml_workspace = Run.get_context().experiment.workspace
# datastore = Datastore.get(aml_workspace, datastore_name='trainingdata')

datastore = Datastore.get(aml_workspace, datastore_name='datamgmtdb')
query = DataPath(datastore, 'SELECT * FROM Stg.AI_TestData')
tabular = Dataset.Tabular.from_sql_query(query, query_timeout=10)
df = tabular.to_pandas_dataframe()

os.makedirs('./outputs', exist_ok=True)

run = Run.get_context()
run.log('test log', 'test log')

# Save model in the outputs folder so it automatically get uploaded when running on AML Compute
# model_file_name = 'output.pkl'
# with open(os.path.join('./outputs/', model_file_name), 'wb') as file:
#     pickle.dump('placeholder', file)

