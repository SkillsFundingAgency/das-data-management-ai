import azureml.core
from azureml.core import Workspace, Datastore, Dataset, ComputeTarget, Experiment
from azureml.core.run import Run
from azureml.pipeline.steps import PythonScriptStep
from azureml.pipeline.core import Pipeline, PipelineData
 
ws = Run.get_context().experiment.workspace
compute_target = ws.compute_targets["cpucluster"]
experiment = Experiment(ws, 'placeholder-exp') 
source_dir = "./outputs"

score_step = PythonScriptStep(
    name='scoring',
    script_name="model_scoring.py",
    target=compute_target,
    source_directory=source_dir)

steps = [score_step]

pipeline = Pipeline(workspace=ws, steps=steps)

pipeline_run = experiment.submit(pipeline)
pipeline_run.wait_for_completion()
