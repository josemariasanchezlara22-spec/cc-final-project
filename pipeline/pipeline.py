from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from kfp import compiler
from kfp import dsl

from components.preprocess_component import preprocess_component
from components.train_component import train_component
from components.evaluate_component import evaluate_component
from components.register_model_component import register_and_deploy_model_component

@dsl.pipeline(
    name="credit-risk-pipeline"
)
def credit_risk_pipeline():

    processed_data_path = "gs://am-up-01-credit-riskv2/data/processed"
    model_output_dir = "gs://am-up-01-credit-riskv2/vertex_models/xgb_optuna_v2"

    # 1. Preprocesamiento de datos
    preprocess_task = preprocess_component(
        input_csv_path="gs://am-up-01-credit-riskv2/data/raw/loan.csv",
        output_path=processed_data_path
    )

    # 2. Entrenamiento (Se ejecuta estrictamente DESPUÉS del preprocesamiento)
    train_task = train_component(
        processed_data_path=processed_data_path,
        model_output_dir=model_output_dir
    ).after(preprocess_task)

    # 3. Evaluación de métricas en Test set
    evaluate_task = evaluate_component(
        model_output_dir=model_output_dir,
        processed_data_path=processed_data_path
    ).after(train_task)

    # 4. Umbral de validación de calidad del modelo (ROC AUC >= 70%)
    with dsl.If(evaluate_task.outputs["roc_auc"] >= 0.70):
        # Solo invócalo, no lo iguales a "deploy_task = ..." ya que no regresa nada
        register_and_deploy_model_component(
            project_id="cloud-computing-jm-2026v2",
            region="us-central1",
            bucket_name="am-up-01-credit-riskv2",
            model_version="xgb_optuna_v2"
        ).after(evaluate_task)

if __name__ == "__main__":
    compiler.Compiler().compile(
        pipeline_func=credit_risk_pipeline,
        package_path="credit_risk_pipeline.json"
    )
    print("Pipeline compilado correctamente en 'credit_risk_pipeline.json'")