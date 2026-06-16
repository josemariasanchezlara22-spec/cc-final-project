import pandas as pd

from config import (
    RAW_DATA_PATH,
    TARGET_STATUS
)


def load_data():

    df = pd.read_csv(
        RAW_DATA_PATH,
        low_memory=False
    )

    print(f"Dataset cargado: {df.shape}")

    return df


def create_target(df):

    df = df[
        df["loan_status"].isin(
            TARGET_STATUS
        )
    ].copy()

    df["target"] = (
        df["loan_status"] == "Charged Off"
    ).astype(int)

    print(
        f"Dataset después del filtro: {df.shape}"
    )

    print(
        df["target"].value_counts()
    )

    return df


def save_dataset(df):

    output_path = (
        "data/processed/raw_dataset.parquet"
    )

    df.to_parquet(
        output_path,
        index=False
    )

    print(
        f"Dataset guardado en {output_path}"
    )


def main():

    df = load_data()

    df = create_target(df)

    save_dataset(df)


if __name__ == "__main__":
    main()