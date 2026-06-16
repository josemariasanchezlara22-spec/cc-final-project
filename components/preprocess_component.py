from kfp.dsl import component


@component(
    base_image="python:3.11",
    packages_to_install=[
        "pandas",
        "numpy",
        "pyarrow",
        "scikit-learn",
        "gcsfs"
    ]
)
def preprocess_component(
    input_csv_path: str,
    output_path: str
):
    """
    Lee el dataset desde Cloud Storage,
    realiza preprocesamiento y genera
    X_train, X_test, y_train, y_test.
    """

    import pandas as pd

    from sklearn.model_selection import train_test_split

    RANDOM_STATE = 42

    candidate_features = [
        "loan_amnt",
        "term",
        "int_rate",
        "installment",
        "grade",
        "sub_grade",
        "emp_length",
        "home_ownership",
        "annual_inc",
        "verification_status",
        "application_type",
        "purpose",
        "dti",
        "delinq_2yrs",
        "inq_last_6mths",
        "open_acc",
        "pub_rec",
        "revol_bal",
        "revol_util",
        "total_acc",
        "mort_acc",
        "pub_rec_bankruptcies",
        "tax_liens"
    ]

    # -----------------------------
    # Carga dataset
    # -----------------------------

    df = pd.read_csv(
        input_csv_path,
        low_memory=False
    )

    # -----------------------------
    # Target
    # -----------------------------

    target_status = [
        "Fully Paid",
        "Charged Off"
    ]

    df = df[
        df["loan_status"].isin(
            target_status
        )
    ].copy()

    df["target"] = (
        df["loan_status"] == "Charged Off"
    ).astype(int)

    # -----------------------------
    # Selección variables
    # -----------------------------

    df = df[
        candidate_features +
        ["target"]
    ].copy()

    # -----------------------------
    # Imputación
    # -----------------------------

    numeric_impute_cols = [
        "pub_rec_bankruptcies",
        "dti",
        "tax_liens",
        "inq_last_6mths"
    ]

    for col in numeric_impute_cols:

        df[col] = df[col].fillna(
            df[col].median()
        )

    # -----------------------------
    # One Hot Encoding
    # -----------------------------

    categorical_features = (
        df
        .select_dtypes(include=["object"])
        .columns
        .tolist()
    )

    df = pd.get_dummies(
        df,
        columns=categorical_features,
        drop_first=True,
        dtype=int
    )

    df.columns = (
        df.columns
        .str.replace("[", "_", regex=False)
        .str.replace("]", "_", regex=False)
        .str.replace("<", "_lt_", regex=False)
        .str.replace(">", "_gt_", regex=False)
    )

    # -----------------------------
    # Split
    # -----------------------------

    X = df.drop(
        columns=["target"]
    )

    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y
    )

    # -----------------------------
    # Guardar en GCS
    # -----------------------------

    X_train.to_parquet(
        f"{output_path}/X_train.parquet",
        index=False
    )

    X_test.to_parquet(
        f"{output_path}/X_test.parquet",
        index=False
    )

    y_train.to_frame(
        name="target"
    ).to_parquet(
        f"{output_path}/y_train.parquet",
        index=False
    )

    y_test.to_frame(
        name="target"
    ).to_parquet(
        f"{output_path}/y_test.parquet",
        index=False
    )

    print("Preprocesamiento completado")