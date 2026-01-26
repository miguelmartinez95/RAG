import time
from mlflow.tracking import MlflowClient
import yaml
import os

CONFIG_PATH = os.getenv("MODEL_CONFIG_PATH", "/app/config/model_config.yaml")
with open(CONFIG_PATH) as f:
    config = yaml.safe_load(f)

MODEL_NAME = config["generation_model"]["label"]
client = MlflowClient()

# Retry loop to wait for alias propagation
for attempt in range(10):
    versions = client.search_model_versions(f"name='{MODEL_NAME}'")
    staging_versions = [v for v in versions if "staging" in (v.aliases or [])]

    if staging_versions:
        break

    print(f"Attempt {attempt + 1}: no staging version found yet, retrying in 2s...")
    time.sleep(2)
else:
    raise RuntimeError(f"No model version with alias 'staging' found for {MODEL_NAME} after retries")

# Take the latest if multiple exist
staging = sorted(staging_versions, key=lambda v: v.version)[-1]

# Promote to production
client.set_registered_model_alias(
    name=MODEL_NAME,
    alias="production",
    version=staging.version
)

print(f"Promoted {MODEL_NAME} v{staging.version} → @production")
