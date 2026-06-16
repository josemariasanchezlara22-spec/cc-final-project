from google.cloud import aiplatform

from config import (
    PROJECT_ID,
    REGION,
    BUCKET_NAME,
    MODEL_NAME,
    MODEL_VERSION
)


def register_model():

    aiplatform.init(
        project=PROJECT_ID,
        location=REGION,
        staging_bucket=f"gs://{BUCKET_NAME}"
    )

    artifact_uri = (
        f"gs://{BUCKET_NAME}/vertex_models/{MODEL_VERSION}"
    )

    serving_container_image_uri = (
        "us-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-5:latest"
    )

    model = aiplatform.Model.upload(
        display_name=f"{MODEL_NAME}_{MODEL_VERSION}",
        artifact_uri=artifact_uri,
        serving_container_image_uri=serving_container_image_uri,
        labels={
            "model": "credit-risk",
            "framework": "xgboost",
            "version": MODEL_VERSION.replace("_", "-")
        }
    )

    print("Modelo registrado correctamente.")
    print(f"Model resource name: {model.resource_name}")

    return model


def main():

    register_model()


if __name__ == "__main__":
    main()