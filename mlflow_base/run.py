from mlbase.ml_training.evaluation import evaluate_on_test, run_cross_validation
from mlbase.feature.data_processing import get_feature_dataframe, get_feature_types
from mlbase.ml_training.retrieval import get_train_val_test_set
from mlbase.ml_training.train import run_hyperopt, train_model
from mlbase.ml_training.model_pipeline import get_pipeline
from mlbase.utils.utils import set_or_create_experiment
import pandas as pd
from mlbase.config.config import EXPERIMENT_NAME, MODEL_NAME


if __name__=="__main__":
    
    # Create or set experiment
    experiment_id = set_or_create_experiment(experiment_name=EXPERIMENT_NAME)

    # Load data
    df = get_feature_dataframe()

    # Preprocess
    X_train, X_val, X_test, y_train, y_val, y_test = get_train_val_test_set(df)
    excluded_cols = ["id", "target", "MedHouseVal"]
    numerical_features, categorical_features = get_feature_types(X_train, excluded_cols)

    # Fine-tuning
    best_params = run_hyperopt(X_train, X_val, y_train, y_val, numerical_features, experiment_id)

    # Build a pipeline with hyper-param tuning
    pipeline = get_pipeline(numerical_features=numerical_features, categorical_features=[])
    pipeline.set_params(**best_params)

    X_all = pd.concat([X_train, X_val])
    y_all = pd.concat([y_train, y_val])

    # Cross-validation
    mean_f1, std_f1 = run_cross_validation(
        pipeline, 
        X_all[numerical_features], 
        y_all, experiment_id
    )

    # Train Model
    run_id, trained_pipeline, model = train_model(
        pipeline=pipeline,
        run_name=f"register_model_f1_{mean_f1:.3f}",
        model_name=MODEL_NAME,
        artifact_path=f"best-model",
        x=X_all,
        y=y_all,
    )

    # Make prediction on test set
    y_pred = pipeline.predict(X_test[numerical_features])

    # Evluate on test set
    metrics = evaluate_on_test(
        pipeline=pipeline,
        X_test=X_test,
        y_test=y_test,
        y_pred=y_pred,
        X_all=X_all,
        run_id=run_id
    )

    # print the result
    print("\n=== Summary ===")
    print("Best Parameters:", best_params)
    print("CV F1 Mean:", round(mean_f1, 4))
    print("Test Metrics:", metrics)
    print("Run ID:", run_id)
