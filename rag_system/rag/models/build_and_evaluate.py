import mlflow
import yaml
from datetime import datetime
from .compute_metrics import compute_metrics
from mlflow.tracking import MlflowClient
from mlflow.models import Model
import logging
from pathlib import Path
import os

import mlflow.pyfunc

class HFDirectoryModel(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        self.model_dir = context.artifacts["model_dir"]

    def predict(self, context, model_input):
        raise NotImplementedError("Inference not implemented yet")

# ---- Config ----
# ConfigMap-mounted path
CONFIG_PATH = os.getenv("MODEL_CONFIG_PATH", "/app/configmap/model_config.yaml")

with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

#PVC_MODEL_PATH = os.path.join("C:", "kind-data", "actions-runner", "actions-runner","_work", "RAG","RAG","rag_system","tests","ci_models","generation")


PVC_MODEL_PATH = Path(
    "C:/kind-data/actions-runner/actions-runner/_work/"
    "RAG/RAG/rag_system/tests/ci_models/generation"
).resolve()

PVC_MODEL_URI = PVC_MODEL_PATH.as_uri()

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
            mlflow.pyfunc.log_model(
                artifact_path="model",
                python_model=HFDirectoryModel(),
                artifacts={"model_dir": PVC_MODEL_URI},
                metadata={
                    "hf_model_id": HF_MODEL_ID,
                    "framework": "transformers"
                }
            )

            # 2️⃣ Register model
            result = mlflow.register_model(
                model_uri=f"runs:/{run_id}/model",
                name=MODEL_NAME
            )

            # 3️⃣ Promote to Staging
            client.transition_model_version_stage(
                name=MODEL_NAME,
                version=result.version,
                stage="Staging"
            )

            mlflow.set_tag("promotion_candidate", "true")
            logging.info(f"Model registered as STAGING (v{result.version})")
            client.set_model_version_tag(
                name=MODEL_NAME,
                version=version,
                key="hf_model_id",
                value=HF_MODEL_ID
            )
            client.set_model_version_tag(
                name=MODEL_NAME,
                version=version,
                key="model_type",
                value="huggingface"
            )

        except Exception as e:
            logging.error(f"MLflow registration failed: {e}")
    else:
        mlflow.set_tag("promotion_candidate", "false")
        logging.warning("Model failed thresholds — not registered")