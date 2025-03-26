from sklearn.datasets import fetch_california_housing
import pandas as pd
import os

def load_data() -> pd.DataFrame:
    cur_directory_path = os.path.abspath(os.path.dirname(__file__))
    data = fetch_california_housing(
        data_home=f"{cur_directory_path}/data/", as_frame=True, download_if_missing=True
    )
    return data.frame

def get_feature_dataframe() -> pd.DataFrame:
    """
    Define X and y
    """
    df = load_data()
    df["id"] = df.index
    df["target"] = df["MedHouseVal"] >= df["MedHouseVal"].median()
    df["target"] = df["target"].astype(int)

    return df

def get_feature_types(df, excluded_cols=None):
    """
    Splits features into numerical and categorical based on dtypes.

    :param df: Input DataFrame
    :param excluded_cols: List of columns to exclude from both types
    :return: numerical_features, categorical_features
    """
    if excluded_cols is None:
        excluded_cols = []

    numerical_features = (
        df.select_dtypes(include=["int64", "float64"])
        .columns.difference(excluded_cols)
        .tolist()
    )

    categorical_features = (
        df.select_dtypes(include=["object", "category"])
        .columns.difference(excluded_cols)
        .tolist()
    )

    return numerical_features, categorical_features