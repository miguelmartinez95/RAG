import mlflow
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from mlflow.tracking import MlflowClient
from .ensure_state import ensure_rag_state
import yaml
import os
import logging
import sys
from pathlib import Path
import shutil
import tempfile

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

with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

MLFLOW_URI = config["mlflow_uri"]
MODEL_NAME = config["generation_model"]["label"]
BOOTSTRAP_MODEL_PATH = os.getenv(
    "GENERATION_MODEL_PATH",
    "/models/generation_model"
)

mlflow.set_tracking_uri(MLFLOW_URI)
client = MlflowClient()


class GeneratorModel:
    _instance = None
    _failed = False   # 👈 ADD THIS

    @classmethod
    def get_models(cls):
        if cls._instance is not None:
            return cls._instance

        if cls._failed:
            raise RuntimeError("Generator model previously failed to load")

        try:
            try:
                model_uri = f"models:/{MODEL_NAME}/Production"
                local_path = mlflow.artifacts.download_artifacts(model_uri)
                logger.info(f"Loaded model from MLflow: {local_path}")
            except Exception as e:
                local_path = BOOTSTRAP_MODEL_PATH
                logger.warning(
                    f"MLflow model unavailable, using bootstrap model at {local_path}: {e}"
                )

            # 2️⃣ Check for 'model' subfolder, fallback to root
            model_path_candidate = Path(local_path) / "model"
            if model_path_candidate.exists():
                model_path = model_path_candidate
            else:
                model_path = Path(local_path)
            logger.info(f"Using model path: {model_path}")

            # 3️⃣ Copy to a temporary folder to avoid HF repo validation issues (Windows)
            temp_dir = tempfile.TemporaryDirectory()
            tmp_model_path = Path(temp_dir.name) / "model"
            shutil.copytree(model_path, tmp_model_path)

            # 4️⃣ Load tokenizer & model
            tokenizer = AutoTokenizer.from_pretrained(tmp_model_path, local_files_only=True, trust_remote_code=False)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            model = AutoModelForCausalLM.from_pretrained(tmp_model_path, local_files_only=True, trust_remote_code=False)

            if tokenizer.vocab_size != model.config.vocab_size:
                raise RuntimeError(
                    f"Tokenizer/model vocab mismatch: "
                    f"{tokenizer.vocab_size} vs {model.config.vocab_size}"
                )

            cls._instance = {
                "model": model,
                "tokenizer": tokenizer,
                "pipeline": pipeline(
                    "text-generation",
                    model=model,
                    tokenizer=tokenizer,
                    truncation=True
                ),
                "_temp_dir": temp_dir  # keep temp dir alive for HF
            }
            return cls._instance

        except Exception as e:
            cls._failed = True
            logger.info(f"Generator failed to initialize: {e}")
            raise

    @classmethod
    def preload(cls):
        """Force model loading at startup"""
        logger.info("Preloading generator model...")
        cls.get_models()
        logger.info("Generator model loaded")

    # ✅ PUBLIC API (no protected access)
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
