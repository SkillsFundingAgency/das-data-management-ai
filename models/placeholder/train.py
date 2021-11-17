import azureml.core
from azureml.core import Workspace, Datastore, Dataset, ComputeTarget, Experiment
from azureml.core.run import Run
from azureml.pipeline.steps import PythonScriptStep
from azureml.pipeline.core import Pipeline, PipelineData
from azureml.pipeline.core import PublishedPipeline

ws = Run.get_context().experiment.workspace
compute_target = ws.compute_targets["cpucluster"]
experiment = Experiment(ws, 'placeholder-exp') 
source_dir = "./"

score_step = PythonScriptStep(
    name='scoring',
    script_name="model_scoring.py",
    compute_target=compute_target,
    source_directory=source_dir)

steps = [score_step]

# Create pipeline
pipeline = Pipeline(workspace=ws, steps=steps)
pipeline.validate()

# Publish pipeline to AzureML
published_pipeline = pipeline.publish('model-scoring-pipeline')

# pipeline_run = experiment.submit(pipeline)
# pipeline_run.wait_for_completion()
