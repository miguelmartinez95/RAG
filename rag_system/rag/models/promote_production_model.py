from mlflow.tracking import MlflowClient
import yaml
import os

CONFIG_PATH = os.getenv("MODEL_CONFIG_PATH", "/app/config/model_config.yaml")
with open(CONFIG_PATH) as f:
    config = yaml.safe_load(f)

MODEL_NAME = config["generation_model"]["label"]
client = MlflowClient()

# ✅ Use MLflow API that filters by stage/alias
staging_versions = client.get_latest_versions(MODEL_NAME, stages=["staging"])

if not staging_versions:
    raise RuntimeError(f"No model version with alias 'staging' found for {MODEL_NAME}")

# Choose the latest version if multiple exist
staging = staging_versions[-1]

# Promote to production
client.set_registered_model_alias(
    name=MODEL_NAME,
    alias="production",
    version=staging.version,
)

print(f"Promoted {MODEL_NAME} v{staging.version} → @production")
