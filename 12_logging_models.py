import mlflow
from mlflow_utils import get_mlflow_experiment

from sklearn.datasets import make_classification 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import PrecisionRecallDisplay
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import ConfusionMatrixDisplay

import matplotlib.pyplot as plt 

if __name__=="__main__":
    experiment = get_mlflow_experiment(experiment_name="testing_mlflow1")
    print("Name: {}".format(experiment.name))

    with mlflow.start_run(run_name="logging_models", experiment_id=experiment.experiment_id) as run:

        X, y = make_classification(n_samples=1000, n_features=10, n_informative=5, n_redundant=5, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Log model parametres
        mlflow.autolog()

        # Random Forest
        rfc = RandomForestClassifier(n_estimators=100, random_state=42)
        rfc.fit(X_train, y_train)
        y_pred = rfc.predict(X_test)

        # Print run info
        print("Run ID: {}".format(run.info.run_id), "\n",
              "Experiment ID: {}".format(run.info.experiment_id), "\n",
              "Status: {}".format(run.info.status), "\n",
              "Start time: {}".format(run.info.start_time), "\n",
              "End time: {}".format(run.info.end_time), "\n",
              "Lifecycle_stage: {}".format(run.info.lifecycle_stage))
        




