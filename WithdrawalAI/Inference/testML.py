from azureml.core import Workspace, Datastore, Dataset, ComputeTarget, Experiment, ScriptRunConfig, Environment, Model
from azureml.core.run import Run
 
# Set up config of workspace and datastore


aml_workspace = Run.get_context().experiment.workspace
run=Run.get_context()
run.log("HELLO THERE, STARTING JOB")



exit(0)
