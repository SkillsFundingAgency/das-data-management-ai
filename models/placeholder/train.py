import pickle
import os
from azureml.core.run import Run

from utils import mylib

os.makedirs('./outputs', exist_ok=True)


run = Run.get_context()



# Save model in the outputs folder so it automatically get uploaded when running on AML Compute
model_file_name = 'output.pkl'
with open(os.path.join('./outputs/', model_file_name), 'wb') as file:
    pickle.dump(reg, file)
