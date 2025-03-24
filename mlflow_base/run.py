from functools import partial
import mlflow
from package.utils.utils import objective_function
from package.feature.data_processing import get_feature_dataframe
from package.ml_training.retrieval import get_train_val_test_set
from package.ml_training.train import train_model
from package.ml_training.model_pipeline import get_pipeline
from package.utils.utils import set_or_create_experiment
from package.utils.utils import get_classification_metrics
from package.utils.utils import get_performance_plots

from hyperopt import fmin 
from hyperopt import tpe 
from hyperopt import Trials 
from hyperopt import hp


if __name__=="__main__":
    
    mlflow.set_tracking_uri("file:///C:/Users/dof07/Desktop/mlflow/mlruns")  # Store runs in a more central location

    experiment_name = "base_classifier"
    run_name = "training_classifier"
    model_name = "registered_model"
    artifact_path = "model"

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
        "model__max_depth": hp.quniform("model__max_depth", 10, 100, 10),
    }

    with mlflow.start_run(experiment_id = experiment_id, run_name="hyperparameter_optimization") as run:

        best_params = fmin(
            fn = partial( # partial when you want more parameters than default fn.
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
            max_evals=5,
            trials=Trials(),
        )

    # Build a pipeline with hyper-param tuning
    pipeline = get_pipeline(numerical_features=numerical_features, categorical_features=[])
    best_params.update({"model__max_depth": int(best_params["model__max_depth"])})
    best_params.update({"model__n_estimators": int(best_params["model__n_estimators"])})
    pipeline.set_params(**best_params)
    pipeline.fit(X_train, y_train)

    # Make prediction on validation set
    y_pred = pipeline.predict(X_val[numerical_features])

    # Evaluate
    metrics = get_classification_metrics(
        y_true=y_val, y_pred=y_pred, prefix="best_model_val"
    )
    performance_plots = get_performance_plots(y_true=y_val, y_pred=y_pred, prefix="val")

    # Log performance metrics
    with mlflow.start_run(experiment_id=experiment_id):
        mlflow.log_params(pipeline["model"].get_params())
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(pipeline, f"{run.info.run_id}-best-model")

        mlflow.set_tags({"type": "classifier"})
        mlflow.set_tag("mlflow.note.content", "This is a binary classifier for the house pricing dataset.")
        for plot_name, fig in performance_plots.items():
            mlflow.log_figure(fig, artifact_file=f"{plot_name}.png")
    