import mlflow
from sklearn.model_selection import cross_val_score, StratifiedKFold
from mlbase.utils.utils import get_classification_metrics
from mlbase.utils.utils import get_performance_plots
from mlbase.utils.utils import log_cv_metrics
import numpy as np

def run_cross_validation(pipeline, X_all, y_all, experiment_id):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    with mlflow.start_run(run_name="cross_validation", experiment_id=experiment_id) as run:
        scores = cross_val_score(pipeline, X_all, y_all, cv=cv, scoring="f1", n_jobs=-1)
        mean_f1 = np.mean(scores)
        std_f1 = np.std(scores)
        log_cv_metrics(scores, cv, mean_f1, std_f1)
        return mean_f1, std_f1

def evaluate_on_test(pipeline, X_test, y_test, y_pred, X_all, run_id):
    # Evaluate on test set
    metrics = get_classification_metrics(
        y_true=y_test, y_pred=y_pred, prefix="best_model_test"
    )
    performance_plots = get_performance_plots(y_true=y_test, y_pred=y_pred, prefix="test")

    # Log performance metrics and plots 
    mlflow.start_run(run_id=run_id)
    mlflow.log_params(pipeline["model"].get_params())
    mlflow.log_metrics(metrics)

    mlflow.set_tag("model_artifact_path", f"best-model")
    mlflow.set_tag("train_samples", len(X_all))
    mlflow.set_tag("test_samples", len(X_test))
    mlflow.set_tags({"type": "classifier"})
    mlflow.set_tag("mlflow.note.content", "This is a binary classifier for the house pricing dataset.")
    mlflow.set_tag("stage", "register_and_eval")

    for plot_name, fig in performance_plots.items():
        mlflow.log_figure(fig, artifact_file=f"{plot_name}.png")
    mlflow.end_run()

    return metrics
