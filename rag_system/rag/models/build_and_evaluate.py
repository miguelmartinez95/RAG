import mlflow
import mlflow.transformers
import yaml
import logging
import os
from datetime import datetime
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from mlflow.tracking import MlflowClient

from .compute_metrics import compute_metrics

# ---------------- Config ----------------
CONFIG_PATH = os.getenv(
    "MODEL_CONFIG_PATH",
    "/app/configmap/model_config.yaml"
)

with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

MLFLOW_URI = config["mlflow_uri"]
MODEL_NAME = config["generation_model"]["label"]
HF_MODEL_ID = config["generation_model"]["hf_id"]
THRESHOLDS = config["thresholds"]

# Local HF model path (CI / PVC / workspace)
LOCAL_MODEL_PATH = Path(
    os.getenv(
        "GENERATION_MODEL_PATH",
        "rag_system/tests/ci_models/generation"
    )
).resolve()

# ---------------- Logging ----------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------- Load HF model ----------------
logger.info(f"Loading HF model from {LOCAL_MODEL_PATH}")

tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(LOCAL_MODEL_PATH)

# ---------------- MLflow ----------------
mlflow.set_tracking_uri(MLFLOW_URI)
mlflow.set_experiment(MODEL_NAME)
client = MlflowClient()

# ---------------- Evaluate ----------------
metrics = compute_metrics()
logger.info(f"Computed metrics: {metrics}")

pass_metrics = all(
    metrics.get(k, 0) >= THRESHOLDS[k] for k in THRESHOLDS
)

# ---------------- Log + Register ----------------
run_name = f"eval-{HF_MODEL_ID}-{datetime.utcnow():%Y%m%d%H%M%S}"

with mlflow.start_run(run_name=run_name) as run:
    mlflow.log_params({
        "hf_model_id": HF_MODEL_ID,
        "passed_thresholds": pass_metrics,
    })

    for k, v in metrics.items():
        mlflow.log_metric(k, v)

    mlflow.log_artifact(CONFIG_PATH)

    if not pass_metrics:
        mlflow.set_tag("promotion_candidate", "false")
        logger.warning("Model failed thresholds — not registered")
        exit(0)

    logger.info("Logging model to MLflow (transformers flavor)")

    result = mlflow.transformers.log_model(
        transformers_model=model,
        tokenizer=tokenizer,
        artifact_path="model",
        registered_model_name=MODEL_NAME,
    )

    version = result.version

    # 🔥 NEW REGISTRY: use aliases
    client.set_registered_model_alias(
        name=MODEL_NAME,
        alias="staging",
        version=version,
    )

    client.set_model_version_tag(
        name=MODEL_NAME,
        version=version,
        key="hf_model_id",
        value=HF_MODEL_ID,
    )

    client.set_model_version_tag(
        name=MODEL_NAME,
        version=version,
        key="framework",
        value="transformers",
    )

    mlflow.set_tag("promotion_candidate", "true")

    logger.info(
        f"Model registered as {MODEL_NAME} v{version} with alias @staging"
    )
