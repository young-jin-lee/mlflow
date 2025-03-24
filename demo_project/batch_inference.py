from package.feature.data_processing import get_feature_dataframe
from package.ml_training.retrieval import get_train_val_test_set
from sklearn.metrics import classification_report
import pandas as pd
import mlflow 

if __name__=="__main__":

    mlflow.set_tracking_uri("file:///C:/Users/dof07/Desktop/mlflow/mlruns")  # Store runs in a more central location
    
    df = get_feature_dataframe()
    X_train, X_val, X_test, y_train, y_val, y_test = get_train_val_test_set(df)
    features = [f for f in X_train.columns if f not in ["id", "target", "MedHouseVal"]]

    model_uri = "models:/registered_model/latest"
    mlflow_model = mlflow.sklearn.load_model(model_uri=model_uri) # This is the pipeline of the model
    print("Loaded pipeline: ", mlflow_model)

    predictions = mlflow_model.predict(X_test[features])

    scored_data = pd.DataFrame({"prediction": predictions, "target": y_test})

    classification_report = classification_report(y_test, predictions)
    print(classification_report)
    print(scored_data.head(10))
