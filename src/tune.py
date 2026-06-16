import json
import os

import optuna
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from xgboost import XGBClassifier

from config import (
    RANDOM_STATE,
    TARGET_COLUMN,
    MODEL_VERSION
)


OPTUNA_SAMPLE_SIZE = 300000
N_TRIALS = 20


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


def create_optuna_sample(X_train, y_train):

    sample_size = min(
        OPTUNA_SAMPLE_SIZE,
        len(X_train)
    )

    X_optuna, _, y_optuna, _ = train_test_split(
        X_train,
        y_train,
        train_size=sample_size,
        random_state=RANDOM_STATE,
        stratify=y_train
    )

    print(f"X_optuna: {X_optuna.shape}")
    print(f"y_optuna: {y_optuna.shape}")

    return X_optuna, y_optuna


def split_optuna_data(X_optuna, y_optuna):

    X_opt_train, X_opt_valid, y_opt_train, y_opt_valid = train_test_split(
        X_optuna,
        y_optuna,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y_optuna
    )

    return (
        X_opt_train,
        X_opt_valid,
        y_opt_train,
        y_opt_valid
    )


def calculate_scale_pos_weight(y):

    neg = (y == 0).sum()
    pos = (y == 1).sum()

    return neg / pos


def create_objective(
    X_opt_train,
    X_opt_valid,
    y_opt_train,
    y_opt_valid,
    scale_pos_weight
):

    def objective(trial):

        params = {
            "n_estimators": trial.suggest_int(
                "n_estimators",
                150,
                500
            ),
            "max_depth": trial.suggest_int(
                "max_depth",
                3,
                7
            ),
            "learning_rate": trial.suggest_float(
                "learning_rate",
                0.01,
                0.15,
                log=True
            ),
            "subsample": trial.suggest_float(
                "subsample",
                0.7,
                1.0
            ),
            "colsample_bytree": trial.suggest_float(
                "colsample_bytree",
                0.7,
                1.0
            ),
            "min_child_weight": trial.suggest_int(
                "min_child_weight",
                1,
                10
            ),
            "gamma": trial.suggest_float(
                "gamma",
                0.0,
                5.0
            ),
            "reg_alpha": trial.suggest_float(
                "reg_alpha",
                0.0,
                2.0
            ),
            "reg_lambda": trial.suggest_float(
                "reg_lambda",
                0.5,
                5.0
            ),
            "scale_pos_weight": scale_pos_weight,
            "objective": "binary:logistic",
            "eval_metric": "auc",
            "random_state": RANDOM_STATE,
            "n_jobs": -1
        }

        model = XGBClassifier(
            **params
        )

        model.fit(
            X_opt_train,
            y_opt_train
        )

        valid_prob = model.predict_proba(
            X_opt_valid
        )[:, 1]

        auc = roc_auc_score(
            y_opt_valid,
            valid_prob
        )

        return auc

    return objective


def save_best_params(study):

    output_dir = f"artifacts/{MODEL_VERSION}"

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    output_path = f"{output_dir}/best_params.json"

    best_result = {
        "best_roc_auc": study.best_value,
        "best_params": study.best_params
    }

    with open(output_path, "w") as file:
        json.dump(
            best_result,
            file,
            indent=4
        )

    print(f"Mejores parámetros guardados en: {output_path}")

    return output_path


def main():

    X_train, y_train = load_data()

    X_optuna, y_optuna = create_optuna_sample(
        X_train,
        y_train
    )

    (
        X_opt_train,
        X_opt_valid,
        y_opt_train,
        y_opt_valid
    ) = split_optuna_data(
        X_optuna,
        y_optuna
    )

    scale_pos_weight = calculate_scale_pos_weight(
        y_opt_train
    )

    print(f"scale_pos_weight_opt: {scale_pos_weight}")

    objective = create_objective(
        X_opt_train,
        X_opt_valid,
        y_opt_train,
        y_opt_valid,
        scale_pos_weight
    )

    study = optuna.create_study(
        direction="maximize"
    )

    study.optimize(
        objective,
        n_trials=N_TRIALS
    )

    print("Mejor ROC-AUC:", study.best_value)
    print("Mejores hiperparámetros:")
    print(study.best_params)

    save_best_params(
        study
    )


if __name__ == "__main__":
    main()