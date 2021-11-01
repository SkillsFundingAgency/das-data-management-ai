import pickle
import os
from azureml.core.run import Run
from azureml.core import Datastore

aml_workspace = Run.get_context().experiment.workspace
datastore = Datastore.get(aml_workspace, datastore_name='trainingdata')

os.makedirs('./outputs', exist_ok=True)

run = Run.get_context()
run.log('test log', 'test log')

# Save model in the outputs folder so it automatically get uploaded when running on AML Compute
model_file_name = 'output.pkl'
with open(os.path.join('./outputs/', model_file_name), 'wb') as file:
    pickle.dump('placeholder', file)
