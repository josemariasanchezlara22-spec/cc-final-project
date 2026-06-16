import os
from google.cloud import aiplatform

def trigger_pipeline_on_upload(event, context=None):
    # En Gen 2, los atributos pueden venir directo en 'event' o en 'event.data' según la librería de CloudEvents
    data = event.get("data", event)
    
    bucket = data.get("bucket")
    file_name = data.get("name")
    
    PROJECT_ID = "cloud-computing-jm-2026v2"
    REGION = "us-central1"
    BUCKET_NAME = "am-up-01-credit-riskv2"
    
    PIPELINE_TEMPLATE_GCS = f"gs://{BUCKET_NAME}/pipelines/credit_risk_pipeline.yaml"
    PIPELINE_ROOT = f"gs://{BUCKET_NAME}/pipelines/activity_root"

    print(f"Evento detectado en GCS -> Bucket: {bucket} | Archivo: {file_name}")

    # Filtramos únicamente por nombre de archivo y ruta exacta
    if bucket == BUCKET_NAME and file_name == "data/raw/loan.csv":
        print("¡Archivo de datos confirmado! Solicitando ejecución a Vertex AI...")
        try:
            aiplatform.init(project=PROJECT_ID, location=REGION, staging_bucket=f"gs://{BUCKET_NAME}")
            
            job = aiplatform.PipelineJob(
                display_name="automated-retrain-loan-upload",
                template_path=PIPELINE_TEMPLATE_GCS,
                pipeline_root=PIPELINE_ROOT,
                enable_caching=False
            )
            
            job.submit()
            print(f"Pipeline enviado con éxito a Vertex AI. ID de recurso: {job.resource_name}")
            
        except Exception as e:
            print(f"Error crítico al lanzar el pipeline en Vertex AI: {str(e)}")
    else:
        print("El archivo modificado no corresponde a la ruta de datos requerida. Ignorando evento.")