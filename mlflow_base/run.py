from functools import partial
import mlflow
from mlbase.utils.utils import objective_function
from mlbase.feature.data_processing import get_feature_dataframe
from mlbase.ml_training.retrieval import get_split_set, get_train_val_set, get_train_val_test_set
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
    X_train, X_val, X_test, y_train, y_val, y_test = get_split_set(use_test_set, df)
    numerical_features = [feature for feature in X_train.columns if feature not in ["id", "target", "MedHouseVal"]]
    
    # hyper-param tuning
    search_space = {
        "model__n_estimators": hp.quniform("model__n_estimators", 20, 200, 10),
        "model__max_depth": hp.quniform("model__max_depth", 10, 15, 20),
    }

    with mlflow.start_run(experiment_id = experiment_id, run_name="hyperparameter_optimization") as run:
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

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    scores = cross_val_score(
        estimator=pipeline,
        X=X_train[numerical_features],  # Or just X_full if you're using full features
        y=y_train,
        cv=cv,
        scoring="f1",  # or "accuracy", "roc_auc", etc.
        n_jobs=-1  # Use all cores
    )

    print("Cross-Validation F1 scores:", scores)
    print("Average F1:", np.mean(scores))

    run_id, trained_pipeline, model = train_model(
        pipeline=pipeline,
        run_name="register_best_model",
        model_name=model_name,
        artifact_path=f"{run.info.run_id}-best-model",
        x=X_train,
        y=y_train,
    )

    # Make prediction on validation set
    y_pred = pipeline.predict(X_val[numerical_features])

    # Evaluate on validation set
    metrics = get_classification_metrics(
        y_true=y_val, y_pred=y_pred, prefix="best_model_val"
    )
    performance_plots = get_performance_plots(y_true=y_val, y_pred=y_pred, prefix="val")

    # Log performance metrics and plots 
    mlflow.start_run(run_id=run_id)
    mlflow.log_params(pipeline["model"].get_params())
    mlflow.log_metrics(metrics)
    mlflow.set_tags({"type": "classifier"})
    mlflow.set_tag("mlflow.note.content", "This is a binary classifier for the house pricing dataset.")
    for plot_name, fig in performance_plots.items():
        mlflow.log_figure(fig, artifact_file=f"{plot_name}.png")
    mlflow.end_run()