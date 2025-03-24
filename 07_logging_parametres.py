import mlflow
from mlflow_utils import get_mlflow_experiment

if __name__ == "__main__":

    experiment = get_mlflow_experiment(experiment_name="testing_mlflow1")
    print("Experiment ID: {}".format(experiment.experiment_id))
    
    # Which experiemnt you are working on
    with mlflow.start_run(run_name="logging_params", experiment_id = experiment.experiment_id) as run:

        # Your ML code goes here
        mlflow.log_param("learning_rate", 0.01)

        parametres = {
            "learning_rate": 0.01,
            "epochs": 10,
            "batch_size": 100,
            "loss_function": "mse",
            "optimizer": "adam",
        }

        mlflow.log_params(parametres)

        # Print run info
        print("Run ID: {}".format(run.info.run_id), "\n",
              "Experiment ID: {}".format(run.info.experiment_id), "\n",
              "Status: {}".format(run.info.status), "\n",
              "Start time: {}".format(run.info.start_time), "\n",
              "End time: {}".format(run.info.end_time), "\n",
              "Lifecycle_stage: {}".format(run.info.lifecycle_stage))
        