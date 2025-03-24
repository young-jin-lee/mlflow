from package.feature.data_processing import print_text_from_feature
from package.ml_training.train import print_something_from_train

from package.feature.data_processing import get_feature_dataframe
from package.ml_training.retrieval import get_train_val_test_set
from package.ml_training.train import train_model
from package.ml_training.preprocessing_pipeline import get_pipeline
from package.utils.utils import set_or_create_experiment
from package.utils.utils import get_classification_metrics
from package.utils.utils import get_performance_plots
from package.utils.utils import register_model_with_client
import mlflow

if __name__=="__main__":
    # print_text_from_feature("Hello from feature")
    # print_something_from_train("Hello from train")
    
    mlflow.set_tracking_uri("file:///C:/Users/dof07/Desktop/mlflow/mlruns")  # Store runs in a more central location

    experiment_name = "house_pricing_classifier"
    run_name = "training_classifier"
    model_name = "registered_model"
    artifact_path = "model"

    df = get_feature_dataframe()

    X_train, X_val, X_test, y_train, y_val, y_test = get_train_val_test_set(df)

    features = [feature for feature in X_train.columns if feature not in ["id", "target", "MedHouseVal"]]

    pipeline = get_pipeline(numerical_features=features, categorical_features=[])

    experiment_id = set_or_create_experiment(experiment_name=experiment_name)

    run_id, pipeline, model = train_model(pipeline = pipeline, run_name = run_name, model_name = model_name, artifact_path=artifact_path, x = X_train[features], y = y_train)

    y_pred = model.predict(X_val[features])

    binary_metrics = get_classification_metrics(y_true=y_val, y_pred=y_pred, prefix="val")
 
    performance_plots = get_performance_plots(y_true=y_val, y_pred=y_pred, prefix="val")

    # Register model1(providing the same model_name will create a new version per run)
    # mlflow.register_model(model_uri=f"runs:/{run_id}/{artifact_path}", name=model_name)
    
    # Register model2(providing the same model_name will create a new version per run) - more manual way but if there is an existing model with the same model_name, it will output an error.
    # register_model_with_client(model_name=model_name, run_id=run_id, artifact_path=artifact_path)

    # Instead, you can register the model when logging the trained model by simply passing the model_name as a parametre. 

    # Log performance metrics
    with mlflow.start_run(run_id=run_id):

        # Log metrics
        mlflow.log_metrics(binary_metrics)

        # Log params
        mlflow.log_params(model.get_params())

        # Log tags
        mlflow.set_tags({"type": "classifier"})

        # Log description
        mlflow.set_tag("mlflow.note.content", "This is a binary classifier for the house pricing dataset.")

        # Log plots
        for plot_name, fig in performance_plots.items():
            mlflow.log_figure(fig, artifact_file=f"{plot_name}.png")
    