import mlflow
from transformers import pipeline
from mlflow.tracking import MlflowClient
from .ensure_state import ensure_rag_state
import yaml
import os
import logging
import sys
from pathlib import Path
import tempfile
import shutil

# ---------------- Logging ----------------
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(
    logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
)
handler.stream.reconfigure(encoding="utf-8", errors="replace")
logging.basicConfig(level=logging.INFO, handlers=[handler])
logger = logging.getLogger(__name__)

# ---------------- Config ----------------
CONFIG_PATH = os.getenv("MODEL_CONFIG_PATH", "/app/configmap/model_config.yaml")
with open(CONFIG_PATH) as f:
    config = yaml.safe_load(f)

MLFLOW_URI = config["mlflow_uri"]
MODEL_NAME = config["generation_model"]["label"]
BOOTSTRAP_MODEL_PATH = os.getenv("GENERATION_MODEL_PATH", "/models/generation_model")
MODEL_STAGE = os.getenv("MODEL_STAGE", "Production")  # "Production" by default

mlflow.set_tracking_uri(MLFLOW_URI)
client = MlflowClient()

# ---------------- Generator Singleton ----------------
class GeneratorModel:
    _instance = None
    _failed = False

    @classmethod
    def get_models(cls):
        if cls._instance is not None:
            return cls._instance
        if cls._failed:
            raise RuntimeError("Generator previously failed to load")

        try:
            # 🔹 Try MLflow alias first
            model_uri = f"models:/{MODEL_NAME}@{MODEL_STAGE.lower()}"
            pipeline_obj = None
            try:
                logger.info(f"Trying to load pipeline from MLflow: {model_uri}")
                pipeline_obj = mlflow.transformers.load_model(model_uri)
            except Exception as e:
                logger.warning(f"No model in MLflow {MODEL_STAGE} alias, using bootstrap path: {BOOTSTRAP_MODEL_PATH} ({e})")

            if pipeline_obj:
                pipe = pipeline_obj  # ready-to-use pipeline
                temp_dir = None
            else:
                # 🔹 Fallback: bootstrap local model
                bootstrap_path = Path(BOOTSTRAP_MODEL_PATH)
                if not bootstrap_path.exists():
                    raise FileNotFoundError(f"Bootstrap path does not exist: {bootstrap_path}")

                temp_dir = tempfile.TemporaryDirectory()
                tmp_path = Path(temp_dir.name) / "model"
                shutil.copytree(bootstrap_path, tmp_path)

                pipe = pipeline("text-generation", model=str(tmp_path), device=-1)  # CPU; device=0 for GPU

            cls._instance = {
                "pipeline": pipe,
                "_temp_dir": temp_dir,
            }

            logger.info("Generator model loaded successfully")
            return cls._instance

        except Exception as e:
            cls._failed = True
            logger.error(f"Failed to initialize generator: {e}")
            raise

    @classmethod
    def is_ready(cls) -> bool:
        return cls._instance is not None and not cls._failed


def generate_answer(state):
    state = ensure_rag_state(state)
    pipe = GeneratorModel.get_models()["pipeline"]

    MAX_CONTEXT_CHARS = 500
    context = (state.context or "")[:MAX_CONTEXT_CHARS]
    if not context.strip():
        state.answer = "I don't know"
        return state

    prompt = f"""You are a factual assistant.
Answer the question ONLY using the context below.
Use ONLY the provided context.
Do not add external knowledge.
If the answer is not in the context, say "I don't know".

Context:
{context}

Question:
{state.query}

Answer:"""

    outputs = pipe(
        prompt,
        max_new_tokens=100,
        do_sample=True,
        temperature=0.2,
        top_p=0.7,
        repetition_penalty=1.1,
        return_full_text=False,
    )

    state.answer = outputs[0].get("generated_text", "").strip()
    logger.info(f"GENERATOR: {state.answer}")
    return state
