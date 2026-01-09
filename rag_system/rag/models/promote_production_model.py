from mlflow.tracking import MlflowClient
import yaml
import os

# ConfigMap-mounted path
CONFIG_PATH = os.getenv("MODEL_CONFIG_PATH", "/app/config/model_config.yaml")

with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

MODEL_NAME = config["generation_model"]["label"]

client = MlflowClient()
versions = client.get_latest_versions(MODEL_NAME, stages=["Staging"])
if not versions:
    raise RuntimeError("No model in Staging")


version = versions[0].version
client.transition_model_version_stage(
    name=MODEL_NAME,
    version=version,
    stage="Production",
    archive_existing_versions=True
)

print(f"Promoted {MODEL_NAME} v{version} to Production")