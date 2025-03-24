import mlflow
from mlflow_utils import get_mlflow_experiment

if __name__ == "__main__":
    # retrieve the mlflow experiment
    # experiment = get_mlflow_experiment(experiment_name="testing_mlflow2")
    experiment = get_mlflow_experiment(experiment_id="713180441381320880")

    print("name: {}".format(experiment.name), "\n",
          "Experiment_id: {}".format(experiment.experiment_id), "\n",
          "Artifact location: {}".format(experiment.artifact_location), "\n",
          "Tags: {}".format(experiment.tags), "\n",
          "Lifecycle_stage: {}".format(experiment.lifecycle_stage), "\n",
          "Creation_time: {}".format(experiment.creation_time))