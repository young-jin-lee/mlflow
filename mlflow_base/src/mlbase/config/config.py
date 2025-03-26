import mlflow


mlflow.set_tracking_uri("file:///C:/Users/dof07/Desktop/mlflow/mlruns")  # Store runs in a more central location
EXPERIMENT_NAME = "base_classifier"
MODEL_NAME = "registered_model"