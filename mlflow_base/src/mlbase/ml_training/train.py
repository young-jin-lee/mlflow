import pandas as pd
from sklearn.pipeline import Pipeline
import mlflow
from mlflow.models.signature import infer_signature
from typing import Tuple
from hyperopt import fmin 
from hyperopt import tpe 
from hyperopt import Trials 
from functools import partial
from mlbase.utils.utils import objective_function
from hyperopt import hp

def train_model(
    pipeline: Pipeline, run_name: str, model_name:str, artifact_path: str, x: pd.DataFrame, y: pd.DataFrame
    ) -> Tuple[str, Pipeline]:
    """
    Train a model and log it to MLflow.

    :param pipeline: Pipeline to train.
    :param run_name: Name of the run.
    :param x: Input features.
    :param y: Target variable.
    :return: Run ID.
    """

    signature = infer_signature(x, y)

    with mlflow.start_run(run_name=run_name) as run:
        pipeline = pipeline.fit(x, y)
        mlflow.sklearn.log_model(sk_model=pipeline, artifact_path=artifact_path,signature=signature, registered_model_name=model_name)
        
    model = pipeline[-1]

    return run.info.run_id, pipeline, model

def run_hyperopt(X_train, X_val, y_train, y_val, numerical_features, experiment_id):
    search_space = {
    "model__n_estimators": hp.quniform("model__n_estimators", 20, 100, 20),
    "model__max_depth": hp.quniform("model__max_depth", 10, 20, 5),
    }
    
    trials = Trials()
    with mlflow.start_run(experiment_id = experiment_id, run_name="opt_hyperparam") as run:
        
        best_params = fmin(
            fn = partial( # partial when you want more parameters than default fn.
                objective_function,
                X_train = X_train,
                X_val = X_val,
                y_train = y_train,
                y_val = y_val,
                numerical_features = numerical_features,
                categorical_features = None,
                experiment_id = experiment_id,
                trials_object = trials,
            ),
            space=search_space,
            algo=tpe.suggest,
            max_evals=5,
            trials=trials,
        )

        mlflow.log_dict(best_params, "best_params.json")
        mlflow.log_params(best_params)
        mlflow.set_tag("stage", "hyperopt")

    best_params["model__max_depth"] = int(best_params["model__max_depth"])
    best_params["model__n_estimators"] = int(best_params["model__n_estimators"])
    return best_params