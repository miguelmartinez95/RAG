from mlflow.tracking import MlflowClient
import yaml
import os

CONFIG_PATH = os.getenv("MODEL_CONFIG_PATH", "/app/config/model_config.yaml")

with open(CONFIG_PATH) as f:
    config = yaml.safe_load(f)

MODEL_NAME = config["generation_model"]["label"]

client = MlflowClient()

# Find all versions of the model
versions = client.search_model_versions(f"name='{MODEL_NAME}'")

# 🔹 Find the version with alias 'staging'
staging_versions = [v for v in versions if "staging" in (v.aliases or [])]

if not staging_versions:
    raise RuntimeError(f"No model version with alias 'staging' found for {MODEL_NAME}")

staging = staging_versions[-1]  # choose the latest if multiple

# Promote to production
client.set_registered_model_alias(
    name=MODEL_NAME,
    alias="production",
    version=staging.version,
)

print(f"Promoted {MODEL_NAME} v{staging.version} → @production")
