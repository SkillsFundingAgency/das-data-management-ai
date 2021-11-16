import azureml.core
from azureml.core import Workspace, Datastore, Dataset
from azureml.core.run import Run

ws = Run.get_context().experiment.workspace
compute_target = ws.compute_targets["STANDARD_NC6"]
experiment = Experiment(ws, 'placeholder-exp') 

score_step = PythonScriptStep(
    name='scoring'
    script_name="model_scoring.py",
    target=compute_target,
    source_directory=placeholder)

steps = [score_step]

pipeline = Pipeline(workspace=ws, steps=steps)

pipeline_run = experiment.submit(pipeline)
pipeline_run.wait_for_completion()
