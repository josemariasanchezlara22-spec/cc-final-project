from google.cloud import aiplatform

# 1. Configuración con tus datos reales de la captura
PROJECT_ID = "cloud-computing-jm-2026v2"
REGION = "us-central1"
ENDPOINT_ID = "6078901272566562816"  # <--- Tu ID de la imagen

aiplatform.init(project=PROJECT_ID, location=REGION)

# 2. Conectar al Endpoint activo
endpoint = aiplatform.Endpoint(endpoint_name=ENDPOINT_ID)

# 3. Formato de datos del cliente (Estructura cruda antes del One-Hot Encoding)
cliente_prueba = [{
    "loan_amnt": 12000,
    "term": " 36 months",
    "int_rate": 11.4,
    "installment": 395.0,
    "grade": "B",
    "sub_grade": "B3",
    "emp_length": "5 years",
    "home_ownership": "MORTGAGE",
    "annual_inc": 55000.0,
    "verification_status": "Verified",
    "application_type": "Individual",
    "purpose": "debt_consolidation",
    "dti": 18.2,
    "delinq_2yrs": 0,
    "inq_last_6mths": 1,
    "open_acc": 12,
    "pub_rec": 0,
    "revol_bal": 11000,
    "revol_util": 45.5,
    "total_acc": 22,
    "mort_acc": 1,
    "pub_rec_bankruptcies": 0,
    "tax_liens": 0
}]

print("Enviando cliente de prueba al endpoint de Vertex AI...")

# 4. Invocar la predicción
response = endpoint.predict(instances=cliente_prueba)

# 5. Imprimir el resultado estructurado por tu FastAPI
print("\n========== RESPUESTA DEL MODELO ==========")
for pred in response.predictions:
    print(f"Clase predicha (0=Paga, 1=Impago): {pred['label']}")
    print(f"Probabilidad de riesgo de impago: {pred['probability']:.4%}")
print("===========================================")