import mlflow
from mlflow_utils import get_mlflow_experiment

if __name__ == "__main__":

    experiment = get_mlflow_experiment(experiment_name="testing_mlflow1")
    print("Experiment ID: {}".format(experiment.experiment_id))
    
    # Which experiemnt you are working on
    # mlflow.set_experiment(experiment_name="testing_mlflow1")

    with mlflow.start_run(run_name="logging_metrics"
                          ,experiment_id = experiment.experiment_id
                          ) as run:

        # Your ML code goes here
        mlflow.log_metric("random_metric", 0.001)

        metrics = {
            "mse": 0.01,
            "mae": 0.1,
            "rmse": 10,
            "r2": 100,
        }

        mlflow.log_metrics(metrics)

        # Print run info
        print("Run ID: {}".format(run.info.run_id), "\n",
              "Experiment ID: {}".format(run.info.experiment_id), "\n",
              "Status: {}".format(run.info.status), "\n",
              "Start time: {}".format(run.info.start_time), "\n",
              "End time: {}".format(run.info.end_time), "\n",
              "Lifecycle_stage: {}".format(run.info.lifecycle_stage))
        