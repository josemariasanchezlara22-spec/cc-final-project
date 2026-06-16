from typing import NamedTuple

from kfp.dsl import component, Input, Artifact


@component(
    base_image="python:3.11",
    packages_to_install=[
        "pandas",
        "pyarrow",
        "xgboost",
        "joblib",
        "scikit-learn",
        "gcsfs"
    ]
)
def evaluate_component(
    model_artifact: Input[Artifact],
    processed_data_path: str
) -> NamedTuple(
    "Outputs",
    [
        ("roc_auc", float),
        ("accuracy", float)
    ]
):
    """
    Evalúa el modelo utilizando el dataset
    generado por preprocess_component.
    """

    import json

    import joblib
    import pandas as pd

    from collections import namedtuple

    from sklearn.metrics import (
        accuracy_score,
        roc_auc_score
    )

    model = joblib.load(
        model_artifact.path
    )

    X_test = pd.read_parquet(
        f"{processed_data_path}/X_test.parquet"
    )

    y_test = pd.read_parquet(
        f"{processed_data_path}/y_test.parquet"
    )["target"]

    y_pred = model.predict(
        X_test
    )

    y_prob = model.predict_proba(
        X_test
    )[:, 1]

    accuracy = float(
        accuracy_score(
            y_test,
            y_pred
        )
    )

    roc_auc = float(
        roc_auc_score(
            y_test,
            y_prob
        )
    )

    metrics = {
        "accuracy": accuracy,
        "roc_auc": roc_auc
    }

    with open(
        "/tmp/metrics.json",
        "w"
    ) as file:

        json.dump(
            metrics,
            file,
            indent=4
        )

    print(metrics)

    outputs = namedtuple(
        "Outputs",
        [
            "roc_auc",
            "accuracy"
        ]
    )

    return outputs(
        roc_auc,
        accuracy
    )