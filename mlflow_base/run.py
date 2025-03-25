from functools import partial
import mlflow
from mlbase.utils.utils import objective_function
from mlbase.feature.data_processing import get_feature_dataframe
from mlbase.ml_training.retrieval import get_train_val_test_set
from mlbase.ml_training.train import train_model
from mlbase.ml_training.model_pipeline import get_pipeline
from mlbase.utils.utils import set_or_create_experiment
from mlbase.utils.utils import get_classification_metrics
from mlbase.utils.utils import get_performance_plots

from hyperopt import fmin 
from hyperopt import tpe 
from hyperopt import Trials 
from hyperopt import hp

from sklearn.model_selection import cross_val_score, StratifiedKFold
import numpy as np
import pandas as pd


if __name__=="__main__":
    
    mlflow.set_tracking_uri("file:///C:/Users/dof07/Desktop/mlflow/mlruns")  # Store runs in a more central location

    experiment_name = "base_classifier"
    model_name = "registered_model"

    # Create or set experiment
    experiment_id = set_or_create_experiment(experiment_name=experiment_name)

    # Load data
    df = get_feature_dataframe()

    # Preprocess
    X_train, X_val, X_test, y_train, y_val, y_test = get_train_val_test_set(df)
    numerical_features = [feature for feature in X_train.columns if feature not in ["id", "target", "MedHouseVal"]]
    
    # hyper-param tuning
    search_space = {
        "model__n_estimators": hp.quniform("model__n_estimators", 20, 200, 10),
        "model__max_depth": hp.quniform("model__max_depth", 10, 15, 20),
    }

    with mlflow.start_run(experiment_id = experiment_id, run_name="opt_hyperparam") as run:
        trials = Trials()
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
    best_params.update({"model__max_depth": int(best_params["model__max_depth"])})
    best_params.update({"model__n_estimators": int(best_params["model__n_estimators"])})

    # Build a pipeline with hyper-param tuning
    pipeline = get_pipeline(numerical_features=numerical_features, categorical_features=[])
    pipeline.set_params(**best_params)


    X_all = pd.concat([X_train, X_val])
    y_all = pd.concat([y_train, y_val])

    # Cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    with mlflow.start_run(run_name="cross_validation", experiment_id=experiment_id) as cv_run:
        cv_scores = cross_val_score(
            estimator=pipeline,
            X=X_all[numerical_features],
            y=y_all,
            cv=cv,
            scoring="f1",
            n_jobs=-1
        )

        mean_f1 = np.mean(cv_scores)
        std_f1 = np.std(cv_scores)

        # Log metrics
        mlflow.log_metric("cv_f1_mean", mean_f1)
        mlflow.log_metric("cv_f1_std", std_f1)

        # Log all individual scores too
        for i, score in enumerate(cv_scores):
            mlflow.log_metric(f"cv_f1_fold_{i+1}", score)

        # Optional: log a tag
        mlflow.set_tag("cv_info", "StratifiedKFold, 5 splits, f1 macro")

        # Save run_id to link with next steps
        cv_run_id = cv_run.info.run_id

    run_id, trained_pipeline, model = train_model(
        pipeline=pipeline,
        run_name="register_model",
        model_name=model_name,
        artifact_path=f"{run.info.run_id}-best-model",
        x=X_all,
        y=y_all,
    )

    # Make prediction on test set
    y_pred = pipeline.predict(X_test[numerical_features])

    # Evaluate on test set
    metrics = get_classification_metrics(
        y_true=y_test, y_pred=y_pred, prefix="best_model_test"
    )
    performance_plots = get_performance_plots(y_true=y_test, y_pred=y_pred, prefix="test")

    # Log performance metrics and plots 
    mlflow.start_run(run_id=run_id)
    mlflow.log_params(pipeline["model"].get_params())
    mlflow.log_metrics(metrics)
    mlflow.set_tags({"type": "classifier"})
    mlflow.set_tag("mlflow.note.content", "This is a binary classifier for the house pricing dataset.")
    for plot_name, fig in performance_plots.items():
        mlflow.log_figure(fig, artifact_file=f"{plot_name}.png")
    mlflow.end_run()