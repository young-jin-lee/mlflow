from package.feature.data_processing import get_feature_dataframe
from package.ml_training.retrieval import get_train_val_test_set

import json
import requests
from pprint import pprint

import mlflow


def make_online_inference(payload, y_test):
    BASE_URI = "http://127.0.0.1:5001/"
    headers = {"Content-Type": "application/json"}
    endpoint = BASE_URI + "invocations"
    r = requests.post(endpoint, data=json.dumps(payload), headers=headers)
    print(f"STATUS CODE: {r.status_code}")
    print(f"PREDICTIONS: {r.text}")
    print(f"TARGET: {y_test.iloc[1:2]}")

if __name__ == "__main__":
    mlflow.set_tracking_uri("file:///C:/Users/dof07/Desktop/mlflow/mlruns")  # Store runs in a more central location

    df = get_feature_dataframe()
    X_train, X_val, X_test, y_train, y_val, y_test = get_train_val_test_set(df)
    features = [f for f in X_train.columns if f not in ["id", "target", "MedHouseVal"]]

    feature_values = json.loads(X_test[features].iloc[1:2].to_json(orient="split"))
    # print(feature_values)
    payload = {"dataframe_split": feature_values}
    # pprint(
    #     payload,
    #     indent=4,
    #     depth=10,
    #     compact=True,
    # )

    make_online_inference(payload, y_test)

    