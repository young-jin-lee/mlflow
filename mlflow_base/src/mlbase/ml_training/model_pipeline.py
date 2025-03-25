from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier

from typing import List


def get_pipeline(
    numerical_features: List[str], categorical_features: List[str] = []
    ) -> Pipeline:
    """
    Get sklearn pipeline.

    :param numerical_features: List of numerical features.
    :param categorical_features: List of categorical features.
    :return: Sklearn pipeline.
    """

    preprocessing = ColumnTransformer(
        transformers = [
            (
                "numerical_imputer", 
                SimpleImputer(strategy="median"), 
                numerical_features), # Deal with NA values
            (
                "one_hot_encoder",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_features,
            ),
        ]
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessing", preprocessing), 
            ("model", RandomForestClassifier()),
        ]
    )

    return pipeline