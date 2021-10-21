import pickle
import os
from azureml.core.run import Run

os.makedirs('./outputs', exist_ok=True)

run = Run.get_context()
run.log('thest log', 'test log')
# Save model in the outputs folder so it automatically get uploaded when running on AML Compute
model_file_name = 'output.pkl'
with open(os.path.join('./outputs/', model_file_name), 'wb') as file:
    pickle.dump('placeholder', file)
