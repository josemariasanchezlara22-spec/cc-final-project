from kfp.dsl import component

@component(
    base_image="python:3.11",
    packages_to_install=[
        "google-cloud-aiplatform"
    ]
)
def register_and_deploy_model_component(
    project_id: str,
    region: str,
    bucket_name: str,
    model_version: str
):
    """
    Registra el modelo en Vertex Model Registry asignando las rutas nativas
    del contenedor y lo despliega automáticamente a un Endpoint productivo.
    """
    from google.cloud import aiplatform

    aiplatform.init(
        project=project_id,
        location=region,
        staging_bucket=f"gs://{bucket_name}"
    )

    serving_container_image_uri = (
        f"us-central1-docker.pkg.dev/{project_id}/credit-risk-serving/predictor:latest"
    )

    # 1. Registrar modelo con la configuración del contenedor adaptada
    print("Registrando modelo en Vertex AI Model Registry...")
    model = aiplatform.Model.upload(
        display_name=f"xgboost_credit_risk_{model_version}",
        artifact_uri=f"gs://{bucket_name}/vertex_models/{model_version}",
        serving_container_image_uri=serving_container_image_uri,
        serving_container_predict_route="/predict",
        serving_container_health_route="/health",
        serving_container_ports=[8080]
    )
    print(f"Modelo registrado exitosamente: {model.resource_name}")

    # 2. Crear el Endpoint de predicciones
    print("Creando Endpoint en Vertex AI...")
    endpoint = aiplatform.Endpoint.create(
        display_name=f"endpoint_xgboost_credit_risk_{model_version}"
    )
    print(f"Endpoint creado: {endpoint.resource_name}")

    # 3. Desplegar el modelo al Endpoint
    print("Desplegando modelo al Endpoint (esto puede tardar unos minutos)...")
    model.deploy(
        endpoint=endpoint,
        deployed_model_display_name=f"deployed_xgb_{model_version}",
        machine_type="n1-standard-2",  # Instancia económica ideal para procesamiento estándar
        min_replica_count=1,
        max_replica_count=1
    )
    print("¡Proceso completado! Tu Endpoint está listo para recibir tráfico.")