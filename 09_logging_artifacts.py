import mlflow
from mlflow_utils import get_mlflow_experiment

if __name__ == "__main__":

    experiment = get_mlflow_experiment(experiment_name="testing_mlflow1")
    print("Experiment ID: {}".format(experiment.experiment_id))
    
    # Which experiemnt you are working on
    # mlflow.set_experiment(experiment_name="testing_mlflow1")

    with mlflow.start_run(run_name="logging_artifacts"
                          ,experiment_id = experiment.experiment_id
                          ) as run:

        # Your ML code goes here
        with open("mock_model_artifact.txt", "w") as f:
            f.write("mock model artifact text file")

        mlflow.log_artifact(local_path="mock_model_artifact.txt", artifact_path="mock_artifact_dir")

        # Print run info
        print("Run ID: {}".format(run.info.run_id), "\n",
              "Experiment ID: {}".format(run.info.experiment_id), "\n",
              "Status: {}".format(run.info.status), "\n",
              "Start time: {}".format(run.info.start_time), "\n",
              "End time: {}".format(run.info.end_time), "\n",
              "Lifecycle_stage: {}".format(run.info.lifecycle_stage))
        