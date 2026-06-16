import json
import os
import joblib
import pandas as pd

from xgboost import XGBClassifier

from config import (
    RANDOM_STATE,
    TARGET_COLUMN,
    MODEL_NAME,
    MODEL_VERSION
)


def load_data():

    X_train = pd.read_parquet(
        "data/processed/X_train.parquet"
    )

    y_train = pd.read_parquet(
        "data/processed/y_train.parquet"
    )[TARGET_COLUMN]

    print(f"X_train: {X_train.shape}")
    print(f"y_train: {y_train.shape}")

    return X_train, y_train


def load_best_params():

    params_path = f"artifacts/{MODEL_VERSION}/best_params.json"

    if not os.path.exists(params_path):
        raise FileNotFoundError(
            f"No se encontró el archivo: {params_path}"
        )

    with open(params_path, "r") as file:
        best_result = json.load(file)

    best_params = best_result["best_params"]

    print("Hiperparámetros cargados desde best_params.json:")
    print(best_params)

    return best_params


def calculate_scale_pos_weight(y_train):

    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()

    scale_pos_weight = neg / pos

    print(f"Negativos: {neg}")
    print(f"Positivos: {pos}")
    print(f"scale_pos_weight: {scale_pos_weight}")

    return scale_pos_weight


def train_model(X_train, y_train, best_params, scale_pos_weight):

    params = {
        **best_params,
        "scale_pos_weight": scale_pos_weight,
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "random_state": RANDOM_STATE,
        "n_jobs": -1
    }

    model = XGBClassifier(
        **params
    )

    model.fit(
        X_train,
        y_train
    )

    print("Modelo entrenado correctamente.")

    return model


def save_artifacts(model, X_train):

    output_dir = f"artifacts/{MODEL_VERSION}"

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    model_path = f"{output_dir}/{MODEL_NAME}.pkl"
    columns_path = f"{output_dir}/model_columns.pkl"

    joblib.dump(
        model,
        model_path
    )

    joblib.dump(
        X_train.columns.tolist(),
        columns_path
    )

    print(f"Modelo guardado en: {model_path}")
    print(f"Columnas guardadas en: {columns_path}")

    return model_path, columns_path


def main():

    X_train, y_train = load_data()

    best_params = load_best_params()

    scale_pos_weight = calculate_scale_pos_weight(
        y_train
    )

    model = train_model(
        X_train,
        y_train,
        best_params,
        scale_pos_weight
    )

    save_artifacts(
        model,
        X_train
    )


if __name__ == "__main__":
    main()