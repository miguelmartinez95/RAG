from mlflow.tracking import MlflowClient
import yaml
import os

CONFIG_PATH = os.getenv("MODEL_CONFIG_PATH", "/app/config/model_config.yaml")

with open(CONFIG_PATH) as f:
    config = yaml.safe_load(f)

MODEL_NAME = config["generation_model"]["label"]

client = MlflowClient()

# Find model version with @staging alias
versions = client.search_model_versions(f"name='{MODEL_NAME}'")

staging = next(
    v for v in versions if "staging" in (v.aliases or [])
)

client.set_registered_model_alias(
    name=MODEL_NAME,
    alias="production",
    version=staging.version,
)

print(
    f"Promoted {MODEL_NAME} v{staging.version} → @production"
)
