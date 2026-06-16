from google.cloud import aiplatform

from config import (
    PROJECT_ID,
    REGION,
    BUCKET_NAME
)


PIPELINE_ROOT = (
    f"gs://{BUCKET_NAME}/pipeline_root"
)


PIPELINE_TEMPLATE = (
    "credit_risk_pipeline.json"
)


def main():

    aiplatform.init(
        project=PROJECT_ID,
        location=REGION,
        staging_bucket=f"gs://{BUCKET_NAME}"
    )

    job = aiplatform.PipelineJob(
        display_name="credit-risk-pipeline",
        template_path=PIPELINE_TEMPLATE,
        pipeline_root=PIPELINE_ROOT,
        enable_caching=False
    )

    job.submit()

    print(
        f"Pipeline enviado: {job.resource_name}"
    )


if __name__ == "__main__":
    main()