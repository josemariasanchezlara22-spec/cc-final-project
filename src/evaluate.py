import json
import os
import joblib
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)

from config import (
    TARGET_COLUMN,
    MODEL_NAME,
    MODEL_VERSION
)


def load_data():

    X_test = pd.read_parquet(
        "data/processed/X_test.parquet"
    )

    y_test = pd.read_parquet(
        "data/processed/y_test.parquet"
    )[TARGET_COLUMN]

    print(f"X_test: {X_test.shape}")
    print(f"y_test: {y_test.shape}")

    return X_test, y_test


def load_model():

    model_path = f"artifacts/{MODEL_VERSION}/{MODEL_NAME}.pkl"

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"No se encontró el modelo: {model_path}"
        )

    model = joblib.load(
        model_path
    )

    print(f"Modelo cargado desde: {model_path}")

    return model


def evaluate_model(model, X_test, y_test):

    y_pred = model.predict(
        X_test
    )

    y_prob = model.predict_proba(
        X_test
    )[:, 1]

    metrics = {
        "accuracy": accuracy_score(
            y_test,
            y_pred
        ),
        "precision_default": precision_score(
            y_test,
            y_pred
        ),
        "recall_default": recall_score(
            y_test,
            y_pred
        ),
        "f1_default": f1_score(
            y_test,
            y_pred
        ),
        "roc_auc": roc_auc_score(
            y_test,
            y_prob
        )
    }

    report = classification_report(
        y_test,
        y_pred,
        output_dict=True
    )

    cm = confusion_matrix(
        y_test,
        y_pred
    ).tolist()

    print("Métricas:")
    print(metrics)

    return metrics, report, cm


def save_metrics(metrics, report, cm):

    output_dir = f"artifacts/{MODEL_VERSION}"

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    metrics_path = f"{output_dir}/metrics.json"
    report_path = f"{output_dir}/classification_report.json"
    cm_path = f"{output_dir}/confusion_matrix.json"

    with open(metrics_path, "w") as file:
        json.dump(
            metrics,
            file,
            indent=4
        )

    with open(report_path, "w") as file:
        json.dump(
            report,
            file,
            indent=4
        )

    with open(cm_path, "w") as file:
        json.dump(
            cm,
            file,
            indent=4
        )

    print(f"Métricas guardadas en: {metrics_path}")
    print(f"Reporte guardado en: {report_path}")
    print(f"Matriz de confusión guardada en: {cm_path}")


def main():

    X_test, y_test = load_data()

    model = load_model()

    metrics, report, cm = evaluate_model(
        model,
        X_test,
        y_test
    )

    save_metrics(
        metrics,
        report,
        cm
    )


if __name__ == "__main__":
    main()