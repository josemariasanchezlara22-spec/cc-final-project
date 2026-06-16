import sys
from pathlib import Path
import subprocess  # Importado correctamente a nivel global
from google.cloud import aiplatform

# Detectar la ubicación de la carpeta 'src'
DIR_ACTUAL = Path(__file__).resolve().parent
sys.path.append(str(DIR_ACTUAL))

# Importar variables de configuración global
from config import (
    PROJECT_ID,
    REGION,
    BUCKET_NAME
)

# Configuración de rutas en GCS
PIPELINE_ROOT = f"gs://{BUCKET_NAME}/pipeline_root"
PIPELINE_TEMPLATE = str(DIR_ACTUAL.parent / "credit_risk_pipeline.json")

def main():
    # 1. Apuntar dinámicamente a la carpeta 'pipeline/' que está en la raíz
    script_pipeline = DIR_ACTUAL.parent / "pipeline" / "pipeline.py"
    
    print(f"Compilando el archivo pipeline.py desde: {script_pipeline}...")
    subprocess.run(["python", str(script_pipeline)], check=True)
    
    # 2. Inicializar el SDK de Vertex AI
    print("Conectando con Google Cloud Vertex AI...")
    aiplatform.init(
        project=PROJECT_ID,
        location=REGION,
        staging_bucket=f"gs://{BUCKET_NAME}"
    )

    # 3. Configurar el Job del Pipeline desactivando el caché
    print("Subiendo y lanzando el pipeline a Vertex AI Pipelines...")
    job = aiplatform.PipelineJob(
        display_name="credit-risk-pipeline-execution",
        template_path=PIPELINE_TEMPLATE,
        pipeline_root=PIPELINE_ROOT,
        enable_caching=False
    )

    job.submit()
    print(f"\n¡Pipeline enviado con éxito! Resource Name: {job.resource_name}")

    # 4. Copiar la plantilla JSON a GCS usando la importación global limpia
    print("Actualizando la plantilla JSON en Google Cloud Storage para la Cloud Function...")
    subprocess.run([
        "gsutil", "cp", 
        PIPELINE_TEMPLATE, 
        f"gs://{BUCKET_NAME}/pipeline_root/credit_risk_pipeline.json"
    ], check=True)
    print("¡Plantilla en GCS actualizada con éxito!")

if __name__ == "__main__":
    main()