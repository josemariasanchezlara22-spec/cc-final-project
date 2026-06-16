import os
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from google.cloud import storage

app = FastAPI()
model = None

def download_model():
    global model
    storage_uri = os.environ.get("AIP_STORAGE_URI")
    if not storage_uri:
        raise RuntimeError("La variable de entorno AIP_STORAGE_URI no está definida.")
    
    print(f"Descargando modelo desde Vertex Storage URI: {storage_uri}")
    
    # Limpiar y parsear la ruta de GCS (gs://bucket/path)
    uri_clean = storage_uri.replace("gs://", "")
    bucket_name = uri_clean.split("/")[0]
    blob_prefix = "/".join(uri_clean.split("/")[1:])
    
    if blob_prefix and not blob_prefix.endswith("/"):
        blob_prefix += "/"
        
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    
    # Vertex AI guarda el artefacto como 'model.joblib'
    blob_name = f"{blob_prefix}model.joblib"
    blob = bucket.blob(blob_name)
    
    local_path = "/tmp/model.joblib"
    print(f"Descargando blob: {blob_name} en {local_path}")
    blob.download_to_filename(local_path)
    
    model = joblib.load(local_path)
    print("Modelo cargado exitosamente en memoria.")

@app.on_event("startup")
def startup():
    print("========== ENTORNO VERTEX AI ==========")
    for key, value in sorted(os.environ.items()):
        if "AIP" in key:
            print(f"{key}={value}")
    print("=======================================")
    try:
        download_model()
    except Exception as e:
        print(f"Error crítico cargando el modelo en el startup: {e}")

@app.get("/health")
def health():
    if model is None:
        raise HTTPException(status_code=503, detail="El modelo no está listo o falló al cargar.")
    return {"status": "ok"}

@app.post("/predict")
async def predict(request: Request):
    if model is None:
        raise HTTPException(status_code=503, detail="Modelo no disponible.")
    
    try:
        # Extraer el JSON crudo que envía Vertex AI de forma nativa
        body = await request.json()
        instances = body.get("instances", [])
        
        if not instances:
            return {"predictions": []}
        
        # Convertir datos entrantes a DataFrame
        df = pd.DataFrame(instances)
        
        # ALINEACIÓN DE COLUMNAS (Obligatorio para XGBoost entrenado con One-Hot Encoding)
        if hasattr(model, "get_booster"):
            feature_names = model.get_booster().feature_names
            if feature_names:
                for col in feature_names:
                    if col not in df.columns:
                        df[col] = 0  # Rellenar con 0 si falta la categoría binaria
                df = df[feature_names] # Reordenar exactamente igual que en el entrenamiento
        
        # Ejecutar inferencia en lote
        predictions = model.predict(df).tolist()
        probabilities = model.predict_proba(df)[:, 1].tolist()
        
        # Retornar con la estructura exacta que Vertex AI requiere mapear
        return {
            "predictions": [
                {"label": int(pred), "probability": float(prob)} 
                for pred, prob in zip(predictions, probabilities)
            ]
        }
    except Exception as e:
        print(f"Error detectado durante la predicción: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Error en la predicción: {str(e)}")