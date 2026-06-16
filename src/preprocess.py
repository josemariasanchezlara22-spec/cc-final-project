import pandas as pd

from sklearn.model_selection import train_test_split

from config import (
    CANDIDATE_FEATURES,
    TARGET_COLUMN,
    RANDOM_STATE,
    TEST_SIZE
)


def clean_column_names(columns):

    return (
        columns
        .str.replace("[", "_", regex=False)
        .str.replace("]", "_", regex=False)
        .str.replace("<", "_lt_", regex=False)
        .str.replace(">", "_gt_", regex=False)
    )


def load_dataset():

    df = pd.read_parquet(
        "data/processed/raw_dataset.parquet"
    )

    print(f"Dataset cargado: {df.shape}")

    return df


def select_features(df):

    df = df[
        CANDIDATE_FEATURES + [TARGET_COLUMN]
    ].copy()

    print(f"Dataset con variables seleccionadas: {df.shape}")

    return df


def handle_missing_values(df):

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

    print("Valores faltantes tratados.")

    return df


def encode_features(df):

    categorical_features = (
        df
        .select_dtypes(include=["object"])
        .columns
        .tolist()
    )

    df_encoded = pd.get_dummies(
        df,
        columns=categorical_features,
        drop_first=True,
        dtype=int
    )

    df_encoded.columns = clean_column_names(
        df_encoded.columns
    )

    print(f"Dataset codificado: {df_encoded.shape}")

    return df_encoded


def split_data(df):

    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    print(f"X_train: {X_train.shape}")
    print(f"X_test : {X_test.shape}")
    print(f"y_train: {y_train.shape}")
    print(f"y_test : {y_test.shape}")

    return X_train, X_test, y_train, y_test


def save_outputs(X_train, X_test, y_train, y_test):

    X_train.to_parquet(
        "data/processed/X_train.parquet",
        index=False
    )

    X_test.to_parquet(
        "data/processed/X_test.parquet",
        index=False
    )

    y_train.to_frame(name=TARGET_COLUMN).to_parquet(
        "data/processed/y_train.parquet",
        index=False
    )

    y_test.to_frame(name=TARGET_COLUMN).to_parquet(
        "data/processed/y_test.parquet",
        index=False
    )

    print("Datasets procesados guardados correctamente.")


def main():

    df = load_dataset()

    df = select_features(df)

    df = handle_missing_values(df)

    df = encode_features(df)

    X_train, X_test, y_train, y_test = split_data(df)

    save_outputs(
        X_train,
        X_test,
        y_train,
        y_test
    )


if __name__ == "__main__":
    main()