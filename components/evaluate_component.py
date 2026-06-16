from typing import NamedTuple
from kfp.dsl import component

@component(
    base_image="python:3.11",
    packages_to_install=[
        "pandas",
        "pyarrow",
        "xgboost",
        "joblib",
        "scikit-learn",
        "gcsfs",
        "google-cloud-storage"
    ]
)
def evaluate_component(
    model_output_dir: str,
    processed_data_path: str
) -> NamedTuple(
    "Outputs",
    [
        ("roc_auc", float),
        ("accuracy", float)
    ]
):
    import json
    import joblib
    import pandas as pd
    from collections import namedtuple
    from sklearn.metrics import accuracy_score, roc_auc_score
    from google.cloud import storage

    # Descargar modelo desde la ruta de producción en GCS
    uri_clean = model_output_dir.replace("gs://", "")
    bucket_name = uri_clean.split("/")[0]
    blob_prefix = "/".join(uri_clean.split("/")[1:])
    
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(f"{blob_prefix}/model.joblib")
    
    local_path = "/tmp/model.joblib"
    blob.download_to_filename(local_path)
    model = joblib.load(local_path)

    X_test = pd.read_parquet(f"{processed_data_path}/X_test.parquet")
    y_test = pd.read_parquet(f"{processed_data_path}/y_test.parquet")["target"]

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    accuracy = float(accuracy_score(y_test, y_pred))
    roc_auc = float(roc_auc_score(y_test, y_prob))

    metrics = {"accuracy": accuracy, "roc_auc": roc_auc}
    print(metrics)

    outputs = namedtuple("Outputs", ["roc_auc", "accuracy"])
    return outputs(roc_auc, accuracy)