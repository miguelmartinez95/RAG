import mlflow
import mlflow.transformers
import yaml
import logging
import os
from datetime import datetime
from pathlib import Path
from transformers import pipeline
from mlflow.tracking import MlflowClient
from .compute_metrics import compute_metrics

# ---------------- Config ----------------
CONFIG_PATH = os.getenv("MODEL_CONFIG_PATH", "/app/configmap/model_config.yaml")
with open(CONFIG_PATH) as f:
    config = yaml.safe_load(f)

MLFLOW_URI = config["mlflow_uri"]
MODEL_NAME = config["generation_model"]["label"]
HF_MODEL_ID = config["generation_model"]["hf_id"]
THRESHOLDS = config["thresholds"]
LOCAL_MODEL_PATH = Path(os.getenv("GENERATION_MODEL_PATH", "/models/generation_model")).resolve()

# ---------------- Logging ----------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------- Pipeline ----------------
logger.info(f"Loading HF model locally from {LOCAL_MODEL_PATH}")
gen_pipe = pipeline("text-generation", model=str(LOCAL_MODEL_PATH), device=-1)  # CPU; set 0 for GPU

# ---------------- MLflow ----------------
mlflow.set_tracking_uri(MLFLOW_URI)
mlflow.set_experiment(MODEL_NAME)
client = MlflowClient()

# ---------------- Evaluate ----------------
metrics = compute_metrics()
logger.info(f"Computed metrics: {metrics}")
pass_metrics = all(metrics.get(k, 0) >= THRESHOLDS[k] for k in THRESHOLDS)

# ---------------- Log + Register ----------------
run_name = f"eval-{HF_MODEL_ID}-{datetime.utcnow():%Y%m%d%H%M%S}"
with mlflow.start_run(run_name=run_name) as run:
    mlflow.log_params({"hf_model_id": HF_MODEL_ID, "passed_thresholds": pass_metrics})
    for k, v in metrics.items():
        mlflow.log_metric(k, v)
    mlflow.log_artifact(CONFIG_PATH)

    if not pass_metrics:
        mlflow.set_tag("promotion_candidate", "false")
        logger.warning("Model failed thresholds — not registered")
        exit(0)

    # ✅ Log HF pipeline to MLflow
    result = mlflow.transformers.log_model(
    transformers_model=gen_pipe,
    name="model",
    registered_model_name=MODEL_NAME,
    pip_requirements=[
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "mlflow>=2.6.0",
        "PyYAML>=6.0"
    ]
)

    version = result.version

    # 🔹 Set registry alias for promotion
    client.set_registered_model_alias(name=MODEL_NAME, alias="staging", version=version)
    client.set_model_version_tag(name=MODEL_NAME, version=version, key="hf_model_id", value=HF_MODEL_ID)
    client.set_model_version_tag(name=MODEL_NAME, version=version, key="framework", value="transformers")
    mlflow.set_tag("promotion_candidate", "true")

    logger.info(f"Model registered as {MODEL_NAME} v{version} with alias @staging")
