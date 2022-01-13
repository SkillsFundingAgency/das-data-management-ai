from azureml.core import Workspace
from azureml.core.compute import AmlCompute
from azureml.core.compute import ComputeTarget
from azureml.exceptions import ComputeTargetException
from ml_service.util.env_variables import Env

def get_quality_score() :

    score_source_dir="./employer-engagement/scoring"
    score_step = PythonScriptStep(
        name='scoring',
        script_name="score.py",
        compute_target=aml_compute,
        runconfig=pipeline_run_config,
        source_directory=score_source_dir,
        allow_reuse=False)

    # Create sequence of steps
    score_step_sequence = StepSequence(steps = [score_step])

    # Create pipeline
    score_pipeline = Pipeline(workspace=aml_workspace, steps=score_step_sequence)
    score_pipeline.validate()

    score_pipeline_run = experiment.submit(score_pipeline,regenerate_outputs=True)

    # RunDetails(pipeline_run).show()
    score_pipeline_run.wait_for_completion()

    # Publish pipeline to AzureML
    score_published_pipeline = score_pipeline.publish('scoring-pipeline')

    return