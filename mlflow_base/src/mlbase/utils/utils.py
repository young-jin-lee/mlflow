import mlflow
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict
from typing import List

from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import PrecisionRecallDisplay
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score

from mlbase.ml_training.model_pipeline import get_pipeline


def set_or_create_experiment(experiment_name: str) -> str:
    """
    Get or create an experiment.

    :param experiment_name: Name of the experiment.
    :return: Experiment ID.
    """

    try:
        experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id
    except:
        experiment_id = mlflow.create_experiment(experiment_name)
    finally:
        mlflow.set_experiment(experiment_name=experiment_name)

    return experiment_id


def get_performance_plots(
    y_true: pd.DataFrame, y_pred: pd.DataFrame, prefix: str
    ) -> Dict[str, any]:
    """
    Get performance plots.

    :param y_true: True labels.
    :param y_pred: Predicted labels.
    :param prefix: Prefix for the plot names.
    :return: Performance plots.
    """
    roc_figure = plt.figure()
    RocCurveDisplay.from_predictions(y_true, y_pred, ax=plt.gca())

    cm_figure = plt.figure()
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, ax=plt.gca())

    pr_figure = plt.figure()
    PrecisionRecallDisplay.from_predictions(y_true, y_pred, ax=plt.gca())

    return {
        f"{prefix}_roc_curve": roc_figure,
        f"{prefix}_confusion_matrix": cm_figure,
        f"{prefix}_precision_recall_curve": pr_figure,
    }


def get_classification_metrics(
    y_true: pd.DataFrame, y_pred: pd.DataFrame, prefix: str
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

def objective_function(
        params: Dict, 
        X_train:pd.DataFrame, 
        X_val: pd.DataFrame, 
        y_train:pd.DataFrame, 
        y_val:pd.DataFrame, 
        numerical_features: List[str], 
        categorical_features: List[str],
        experiment_id,
        trials_object) -> float:

    run_index = len(trials_object.trials)
    run_name = f"trial_{run_index}"

    pipeline = get_pipeline(numerical_features=numerical_features)
    params.update({"model__max_depth": int(params["model__max_depth"])})
    params.update({"model__n_estimators": int(params["model__n_estimators"])})
    pipeline.set_params(**params)

    with mlflow.start_run(experiment_id=experiment_id, nested=True, run_name=run_name) as run: # runs in optimization process are always child runs 
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_val)
        metrics = get_classification_metrics(
            y_true=y_val, y_pred=y_pred, prefix="val"
        )

        mlflow.log_params(pipeline["model"].get_params())
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(pipeline, f"{run.info.run_id}-model")

        print("Optimization in Progress - ", f"Run Name: {run_name}", " ", "Hyperparams: ", params)

    return -metrics["val_f1"]

def log_cv_metrics(scores, cv, mean_f1, std_f1):
    # Log metrics
    mlflow.log_metric("cv_f1_mean", mean_f1)
    mlflow.log_metric("cv_f1_std", std_f1)

    # Log all individual scores too
    for i, score in enumerate(scores):
        mlflow.log_metric(f"cv_f1_fold_{i+1}", score)

    # Optional: log a tag
    mlflow.set_tag("cv_info", "StratifiedKFold, 5 splits, f1 macro")
    mlflow.set_tag("stage", "cv")

    mlflow.log_param("cv_n_splits", 5)
    mlflow.log_param("cv_shuffle", True)
    mlflow.log_param("cv_random_state", 42)