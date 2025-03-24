import mlflow
from mlflow_utils import create_mlflow_experiment
from mlflow_utils import get_mlflow_experiment

if __name__ == "__main__":

    experiment_id = create_mlflow_experiment(
        experiment_name = "testing_mlflow1",
        artifact_location="testing_mlflow1_artifacts",
        tags={"env":"dev", "version":"1.0.0"},
    )

    experiment = get_mlflow_experiment(experiment_id=experiment_id)
    print("Experiment name: {}".format(experiment.name))
    
    # Which experiemnt you are working on
    # mlflow.set_experiment(experiment_name="testing_mlflow1")

    with mlflow.start_run(run_name="testing", experiment_id = experiment.experiment_id) as run:

        # Your ML code goes here
        mlflow.log_param("learning_rate", 0.01)

        # Print run info
        print("Run ID: {}".format(run.info.run_id), "\n",
              "Experiment ID: {}".format(run.info.experiment_id), "\n",
              "Status: {}".format(run.info.status), "\n",
              "Start time: {}".format(run.info.start_time), "\n",
              "End time: {}".format(run.info.end_time), "\n",
              "Lifecycle_stage: {}".format(run.info.lifecycle_stage))
        