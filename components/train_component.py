from kfp.dsl import component

@component(
    base_image="python:3.11",
    packages_to_install=[
        "pandas",
        "pyarrow",
        "xgboost",
        "joblib",
        "gcsfs",
        "scikit-learn",
        "google-cloud-storage"
    ]
)
def train_component(
    processed_data_path: str,
    model_output_dir: str
):
    """
    Entrena el modelo utilizando los datasets generados 
    y guarda el archivo 'model.joblib' directamente en la ruta fija de GCS.
    """
    import joblib
    import pandas as pd
    from xgboost import XGBClassifier
    from google.cloud import storage

    print(f"Leyendo datos desde: {processed_data_path}")
    X_train = pd.read_parquet(f"{processed_data_path}/X_train.parquet")
    y_train = pd.read_parquet(f"{processed_data_path}/y_train.parquet")["target"]

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

    model.fit(X_train, y_train)

    # Guardar localmente primero
    local_path = "/tmp/model.joblib"
    joblib.dump(model, local_path)

    # Subir directamente a la ruta fija de GCS donde Vertex lo buscará
    uri_clean = model_output_dir.replace("gs://", "")
    bucket_name = uri_clean.split("/")[0]
    blob_prefix = "/".join(uri_clean.split("/")[1:])
    
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(f"{blob_prefix}/model.joblib")
    blob.upload_from_filename(local_path)

    print(f"Modelo subido exitosamente a GCS: {model_output_dir}/model.joblib")