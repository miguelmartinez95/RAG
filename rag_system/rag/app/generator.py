import mlflow
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from mlflow.tracking import MlflowClient
import yaml
import os

# ConfigMap-mounted path
CONFIG_PATH = os.getenv("MODEL_CONFIG_PATH", "/app/configmap/model_config.yaml")

with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

MLFLOW_URI = config["mlflow_uri"]
MODEL_NAME = config["generation_model"]["label"]
BOOTSTRAP_MODEL_PATH = "/models/generation_model"

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
                latest_prod = client.get_latest_versions(
                    name=MODEL_NAME, stages=["Production"]
                )[0]
                model_uri = f"models:/{MODEL_NAME}/{latest_prod.version}"
                local_path = mlflow.artifacts.download_artifacts(model_uri)
                print(f"Loaded model from MLflow: {model_uri}")
            except Exception as e:
                local_path = BOOTSTRAP_MODEL_PATH
                print(f"MLflow unavailable, using bootstrap model: {e}")

            tokenizer = AutoTokenizer.from_pretrained(local_path)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            model = AutoModelForCausalLM.from_pretrained(local_path, trust_remote_code=True)

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
                )
            }
            return cls._instance

        except Exception as e:
            cls._failed = True
            print(f"Generator failed to initialize: {e}")
            raise

    @classmethod
    def preload(cls):
        """Force model loading at startup"""
        print("Preloading generator model...")
        cls.get_models()
        print("Generator model loaded")

    # ✅ PUBLIC API (no protected access)
    @classmethod
    def is_ready(cls) -> bool:
        return cls._instance is not None and not cls._failed


def generate_answer(state):
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
    max_new_tokens=64,      # reduce further
    do_sample=True,
    temperature=0.2,
    top_p=0.9,
    repetition_penalty=1.1,
    pad_token_id=tokenizer.eos_token_id,
    return_full_text=False,
    )

    state.answer = str(outputs[0].get("generated_text", "")).strip()

    print(f"GENERATOR: {state.answer}")

    return state  # return the state for LangGraph
