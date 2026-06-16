import os
from fastapi import FastAPI

app = FastAPI()


@app.on_event("startup")
def startup():

    print("========== ENVIRONMENT ==========")

    for key, value in sorted(os.environ.items()):

        if "AIP" in key:
            print(f"{key}={value}")

    print("================================")


@app.get("/health")
def health():

    return {
        "status": "ok"
    }


@app.post("/predict")
def predict():

    return {
        "status": "test"
    }