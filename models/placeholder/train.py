import pickle
import os
import pandas as pd
from azureml.core.run import Run
from azureml.core import Dataset, Datastore
from azureml.data.datapath import DataPath
from sklearn.linear_model import LogisticRegression

aml_workspace = Run.get_context().experiment.workspace
# datastore = Datastore.get(aml_workspace, datastore_name='trainingdata')

datastore = Datastore.get(aml_workspace, datastore_name='datamgmtdb')
query = DataPath(datastore, 'SELECT * FROM Stg.AI_TestData')
tabular = Dataset.Tabular.from_sql_query(query, query_timeout=10)
df = tabular.to_pandas_dataframe()

os.makedirs('./outputs', exist_ok=True)

#frequency count of target variable
print(df.groupby(['target'])['dep_var3'].value_counts())

#split data into target and independent sets
y_train=df['target']
x_train=df[['dep_var2','dep_var3']]

#look at the data
x_train.head(5)

#build logistic model
clf=LogisticRegression(random_state=0).fit(x_train,y_train)

#print model coefficients
print(clf.coef_)
print(clf.intercept_)

run = Run.get_context()
run.log('test log', 'test log')

# Save model in the outputs folder so it automatically get uploaded when running on AML Compute
model_file_name = 'test_model.pkl'
with open(os.path.join('./outputs/', model_file_name), 'wb') as file:
    pickle.dump(clf, file)

