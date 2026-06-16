import os

from google.cloud import storage

from config import (
    PROJECT_ID,
    BUCKET_NAME,
    MODEL_VERSION
)


def upload_directory_to_gcs(local_dir, gcs_prefix):

    client = storage.Client(
        project=PROJECT_ID
    )

    bucket = client.bucket(
        BUCKET_NAME
    )

    for root, _, files in os.walk(local_dir):

        for file in files:

            local_path = os.path.join(
                root,
                file
            )

            relative_path = os.path.relpath(
                local_path,
                local_dir
            )

            blob_path = f"{gcs_prefix}/{relative_path}".replace(
                "\\",
                "/"
            )

            blob = bucket.blob(
                blob_path
            )

            blob.upload_from_filename(
                local_path
            )

            print(
                f"Subido: {local_path} -> gs://{BUCKET_NAME}/{blob_path}"
            )


def main():

    local_dir = f"artifacts/{MODEL_VERSION}"
    gcs_prefix = f"models/{MODEL_VERSION}"

    upload_directory_to_gcs(
        local_dir,
        gcs_prefix
    )


if __name__ == "__main__":
    main()