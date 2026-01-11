import mlflow
import yaml
from datetime import datetime
from .compute_metrics import compute_metrics
from mlflow.tracking import MlflowClient
import logging
import os

# ---- Config ----
# ConfigMap-mounted path
CONFIG_PATH = os.getenv("MODEL_CONFIG_PATH", "/app/configmap/model_config.yaml")

with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

PVC_MODEL_PATH = "/models/generation_model"  # where the generation model is stored in PVC

MLFLOW_URI = config["mlflow_uri"]
MODEL_NAME = config["generation_model"]["label"]
HF_MODEL_ID = config["generation_model"]["hf_id"]
THRESHOLDS = config["thresholds"]


# ---- Setup MLflow ----
mlflow.set_tracking_uri(MLFLOW_URI)
mlflow.set_experiment(MODEL_NAME)
client = MlflowClient()

# ---- Compute metrics ----
metrics = compute_metrics()
logging.info(f"Computed metrics: {metrics}")

# ---- Check thresholds safely ----
pass_metrics = all(metrics.get(k, 0) >= THRESHOLDS[k] for k in THRESHOLDS)

# ---- Start MLflow run ----
run_name = f"eval-{HF_MODEL_ID}-{datetime.now().strftime('%Y%m%d%H%M%S')}"
with mlflow.start_run(run_name=run_name) as run:
    run_id = run.info.run_id
    version = datetime.now().strftime("v%Y%m%d%H%M%S")

    mlflow.log_params({
        "candidate_version": version,
        "hf_model_id": HF_MODEL_ID,
        "passed_thresholds": pass_metrics
    })

    for k, v in metrics.items():
        mlflow.log_metric(k, v)

    mlflow.log_artifact(CONFIG_PATH)

    if pass_metrics:
        try:
            # Log HF model as pyfunc or via HF log_model
            mlflow.pyfunc.log_model(
                artifact_path="model",
                python_model=None,
                registered_model_name=MODEL_NAME,
                code_path=[str(PVC_MODEL_PATH)]
            )
            # Safely transition to Staging
            latest_versions = client.get_latest_versions(MODEL_NAME, stages=["None"])
            if latest_versions:
                model_version = latest_versions[0].version
                client.transition_model_version_stage(
                    name=MODEL_NAME,
                    version=model_version,
                    stage="Staging"
                )
                mlflow.set_tag("promotion_candidate", "true")
                logging.info(f"✅ Model registered as STAGING (v{model_version})")
            else:
                logging.warning("No versions found to transition to Staging")
        except Exception as e:
            logging.error(f"MLflow registration failed: {e}")
    else:
        mlflow.set_tag("promotion_candidate", "false")
        logging.warning("❌ Model failed thresholds — not registered")