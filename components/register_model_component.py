from kfp.dsl import component


@component(
    base_image="python:3.11",
    packages_to_install=[
        "google-cloud-aiplatform"
    ]
)
def register_model_component(
    project_id: str,
    region: str,
    bucket_name: str,
    model_version: str
):
    """
    Registra el modelo en Vertex AI Model Registry.
    """

    from google.cloud import aiplatform

    aiplatform.init(
        project=project_id,
        location=region,
        staging_bucket=f"gs://{bucket_name}"
    )

    model = aiplatform.Model.upload(
        display_name=f"xgboost_credit_risk_{model_version}",
        artifact_uri=(
            f"gs://{bucket_name}/vertex_models/{model_version}"
        ),
        serving_container_image_uri=(
            "us-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-5:latest"
        )
    )

    print(
        f"Modelo registrado: {model.resource_name}"
    )