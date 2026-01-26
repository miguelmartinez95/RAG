import mlflow
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from mlflow.tracking import MlflowClient
from .ensure_state import ensure_rag_state
import yaml
import os
import logging
import sys
from pathlib import Path
import tempfile
import shutil


handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(
    logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )
)

# 🔥 Force UTF-8 + safe replacement
handler.stream.reconfigure(encoding="utf-8", errors="replace")

logging.basicConfig(
    level=logging.INFO,
    handlers=[handler],
)
logger = logging.getLogger(__name__)

# ConfigMap-mounted path
CONFIG_PATH = os.getenv("MODEL_CONFIG_PATH", "/app/configmap/model_config.yaml")

MODEL_STAGE = os.getenv("MODEL_STAGE", "Production")


with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)


MLFLOW_URI = config["mlflow_uri"]
MODEL_NAME = config["generation_model"]["label"]
BOOTSTRAP_MODEL_PATH = os.getenv("GENERATION_MODEL_PATH", "/models/generation_model")
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
            # 1️⃣ Check if a model exists in MLflow alias 'production'
            production_versions = client.get_latest_versions(MODEL_NAME, stages=[])
            prod_model_uri = None
            for v in production_versions:
                if "production" in [a.lower() for a in v.aliases]:
                    prod_model_uri = f"models:/{MODEL_NAME}@production"
                    break

            if prod_model_uri:
                logger.info(f"Loading model from MLflow alias 'production': {prod_model_uri}")
                loaded = mlflow.transformers.load_model(prod_model_uri)
                model = loaded.model
                tokenizer = loaded.tokenizer

            else:
                # 2️⃣ Fallback to local bootstrap
                logger.warning(f"No model in MLflow production alias, using bootstrap path: {BOOTSTRAP_MODEL_PATH}")
                bootstrap_path = Path(BOOTSTRAP_MODEL_PATH)
                if not bootstrap_path.exists():
                    raise FileNotFoundError(f"Bootstrap path does not exist: {bootstrap_path}")

                # 3️⃣ Copy to temporary folder (avoid Windows HF repo issues)
                temp_dir = tempfile.TemporaryDirectory()
                tmp_model_path = Path(temp_dir.name) / "model"
                shutil.copytree(bootstrap_path, tmp_model_path)

                tokenizer = AutoTokenizer.from_pretrained(tmp_model_path, local_files_only=True, trust_remote_code=False)
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token

                model = AutoModelForCausalLM.from_pretrained(tmp_model_path, local_files_only=True, trust_remote_code=False)
            pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, truncation=True)

            cls._instance = {
                "model": model,
                "tokenizer": tokenizer,
                "pipeline": pipe,
                "_temp_dir": temp_dir if not prod_model_uri else None,  # keep temp dir alive if used
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

    model_obj = GeneratorModel.get_models()
    pipe = model_obj["pipeline"]
    tokenizer = model_obj["tokenizer"]

    query = state.query
    #context = state.context
    MAX_CONTEXT_CHARS = 500  # start small

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
    {query}

    Answer:"""

    # Use HF pipeline for automatic decoding, batching, device placement
    outputs = pipe(
    prompt,
    max_new_tokens=100,      # reduce further
    do_sample=True,
    temperature=0.2,
    top_p=0.7,
    repetition_penalty=1.1,
    pad_token_id=tokenizer.eos_token_id,
    return_full_text=False,
    )

    state.answer = str(outputs[0].get("generated_text", "")).strip()

    logger.info(f"GENERATOR: {state.answer}")


    return state  # return the state for LangGraph
