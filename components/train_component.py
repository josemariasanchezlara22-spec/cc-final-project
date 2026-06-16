from kfp.dsl import component, Output, Artifact


@component(
    base_image="python:3.11",
    packages_to_install=[
        "pandas",
        "pyarrow",
        "xgboost",
        "joblib",
        "gcsfs"
    ]
)
def train_component(
    processed_data_path: str,
    model_artifact: Output[Artifact]
):
    """
    Entrena el modelo utilizando los datasets
    generados por preprocess_component.
    """

    import joblib
    import pandas as pd

    from xgboost import XGBClassifier

    print("processed_data_path recibido:")
    print(processed_data_path)

    print(
        f"{processed_data_path}/X_train.parquet"
    )
    
    X_train = pd.read_parquet(
        f"{processed_data_path}/X_train.parquet"
    )

    y_train = pd.read_parquet(
        f"{processed_data_path}/y_train.parquet"
    )["target"]

    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()

    scale_pos_weight = neg / pos

    model = XGBClassifier(
        n_estimators=348,
        max_depth=6,
        learning_rate=0.059284370976004655,
        subsample=0.8702765514333507,
        colsample_bytree=0.952941904232232,
        min_child_weight=4,
        gamma=2.501536380722278,
        reg_alpha=0.90853661986719,
        reg_lambda=4.823822796963244,
        scale_pos_weight=scale_pos_weight,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1
    )

    model.fit(
        X_train,
        y_train
    )

    joblib.dump(
        model,
        model_artifact.path
    )

    print(
        f"Modelo guardado en {model_artifact.path}"
    )