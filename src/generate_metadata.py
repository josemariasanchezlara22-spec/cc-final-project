import json
import os

from datetime import datetime, UTC

import pandas as pd

from config import (
    MODEL_NAME,
    MODEL_VERSION
)


def load_data():
    """
    Carga datasets procesados para obtener
    número de registros y variables utilizadas.
    """

    X_train = pd.read_parquet(
        "data/processed/X_train.parquet"
    )

    X_test = pd.read_parquet(
        "data/processed/X_test.parquet"
    )

    return X_train, X_test


def load_metrics():
    """
    Carga métricas generadas durante la evaluación.
    """

    metrics_path = (
        f"artifacts/{MODEL_VERSION}/metrics.json"
    )

    with open(metrics_path, "r") as file:
        metrics = json.load(file)

    return metrics


def generate_metadata():
    """
    Construye metadata del modelo.
    """

    X_train, X_test = load_data()

    metrics = load_metrics()

    metadata = {
        "model_name": MODEL_NAME,
        "version": MODEL_VERSION,
        "algorithm": "XGBoost",
        "trained_at": datetime.now(UTC).isoformat(),
        "roc_auc": metrics["roc_auc"],
        "accuracy": metrics["accuracy"],
        "n_features": len(X_train.columns),
        "features": X_train.columns.tolist(),
        "n_train_records": len(X_train),
        "n_test_records": len(X_test)
    }

    return metadata


def save_metadata(metadata):
    """
    Guarda metadata como JSON.
    """

    output_dir = f"artifacts/{MODEL_VERSION}"

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    output_path = (
        f"{output_dir}/model_metadata.json"
    )

    with open(output_path, "w") as file:
        json.dump(
            metadata,
            file,
            indent=4
        )

    print(
        f"Metadata guardada en: {output_path}"
    )


def main():

    metadata = generate_metadata()

    save_metadata(
        metadata
    )


if __name__ == "__main__":
    main()