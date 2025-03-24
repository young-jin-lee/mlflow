from mlflow_utils import create_dateset
from mlflow_utils import create_mlflow_experiment

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score

from hyperopt import fmin 
from hyperopt import tpe 
from hyperopt import Trials 
from hyperopt import hp

from typing import Dict
from typing import List 
from typing import Optional

import pandas as pd
import mlflow
from functools import partial


def get_classification_metrics(
        y_true:pd.Series, y_pred:pd.Series, prefix:str
    ) -> Dict[str, float]:
    """
    Get classification metrics for logging.

    :param y_true: True labels.
    :param y_pred: Predicted labels.
    :param prefix: Prefix for the metric names.
    :return: Classification metrics.
    """
    metrics = {
        f"{prefix}_accuracy": accuracy_score(y_true, y_pred),
        f"{prefix}_precision": precision_score(y_true, y_pred),
        f"{prefix}_recall": recall_score(y_true, y_pred),
        f"{prefix}_f1": f1_score(y_true, y_pred),
        f"{prefix}_roc_auc": roc_auc_score(y_true, y_pred),
    }

    return metrics

def get_sklearn_pipeline(
        numerical_features: List[str], categorical_features: Optional[List[str]] = []
    ) -> Pipeline:
    """
    Get the sklearn pipeline.
    """

    preprocessing = ColumnTransformer(
        transformers=[
            (
                "numerical", 
                SimpleImputer(strategy="median"), 
                numerical_features),
            (
                "categorical",
                OneHotEncoder(),
                categorical_features
            ),
        ]
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessing", preprocessing),
            ("model", RandomForestClassifier()),
        ]
    )

    return pipeline

def objective_function(
        params: Dict, 
        X_train:pd.DataFrame, 
        X_test: pd.DataFrame, 
        y_train:pd.DataFrame, 
        y_test:pd.DataFrame, 
        numerical_features: List[str], 
        categorical_features: List[str],
        experiment_id) -> float:
    
    # Find minimum value of the function; y = (x - 1)^2 + 2
    # y = (params["x"] - 1) ** 2 + 2

    pipeline = get_sklearn_pipeline(numerical_features=numerical_features)
    params.update({"model__max_depth": int(params["model__max_depth"])})
    params.update({"model__n_estimators": int(params["model__n_estimators"])})
    pipeline.set_params(**params)

    with mlflow.start_run(experiment_id=experiment_id, nested=True) as run: # runs in optimization process are always child runs 
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        metrics = get_classification_metrics(
            y_true=y_test, y_pred=y_pred, prefix="test"
        )

        mlflow.log_params(pipeline["model"].get_params())
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(pipeline, f"{run.info.run_id}-model")

    return -metrics["test_f1"]

if __name__ == "__main__":
    mlflow.set_tracking_uri("file:///C:/Users/dof07/Desktop/mlflow/mlruns")  # Store runs in a more central location

    df = create_dateset()
    print(df.head())

    X_train, X_test, y_train, y_test = train_test_split(
        df.drop("target", axis = 1),
        df["target"],
        test_size = 0.2,
        random_state = 42
    )

    numerical_features = [f for f in X_train.columns if f.startswith("feature")]
    print(numerical_features)
    
    search_space = {
        "model__n_estimators": hp.quniform("model__n_estimators", 20, 200, 10),
        "model__max_depth": hp.quniform("model__max_depth", 10, 100, 10),
    }

    experiment_id = create_mlflow_experiment(
        "hyperopt_experiment",
        artifact_location="hyperopt_mlflow_artifacts",
        tags={"mlflow.note.content":"hyperopt experiment"}
    )
    print("Experiment_id: ", experiment_id)

    with mlflow.start_run(experiment_id = experiment_id, run_name="hyperparameter_optimization") as run:

        best_params = fmin(
            fn=partial( # partial when you want more parameters than default fn.
                objective_function,
                X_train = X_train,
                X_test = X_test,
                y_train = y_train,
                y_test = y_test,
                numerical_features = numerical_features,
                categorical_features = None,
                experiment_id = experiment_id,
            ),
            space=search_space,
            algo=tpe.suggest,
            max_evals=10,
            trials=Trials(),
        )
    pipeline = get_sklearn_pipeline(numerical_features=numerical_features)
    best_params.update({"model__max_depth": int(best_params["model__max_depth"])})
    best_params.update({"model__n_estimators": int(best_params["model__n_estimators"])})
    
    pipeline.set_params(**best_params)
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    metrics = get_classification_metrics(
        y_true=y_test, y_pred=y_pred, prefix="best_model_test"
    )

    with mlflow.start_run(experiment_id = experiment_id) as run:
        mlflow.log_params(pipeline["model"].get_params())
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(pipeline, f"{run.info.run_id}-best-model")


