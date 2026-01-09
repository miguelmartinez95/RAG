from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os
import yaml
import torch

with open("config/rag_config.yaml", "r") as f:
    config = yaml.safe_load(f)

EVALUATOR_OUTPUT_DIR = "/models/evaluator_model"

class Evaluator:
    _instance = None

    @classmethod
    def get_model(cls):
        if cls._instance is None:
            if not os.path.exists(EVALUATOR_OUTPUT_DIR):
                raise RuntimeError(f"Evaluator model not found at {EVALUATOR_OUTPUT_DIR}")

            tokenizer = AutoTokenizer.from_pretrained(EVALUATOR_OUTPUT_DIR)
            model = AutoModelForSequenceClassification.from_pretrained(
                EVALUATOR_OUTPUT_DIR,
                torch_dtype=torch.float32,
            )

            cls._instance = {"model": model, "tokenizer": tokenizer}
        return cls._instance


def evaluate(state):
    """
    LangGraph evaluator node for DeBERTa.
    Input: state.query, state.context, state.answer
    Output: {"score": float} between 0 and 1
    """
    model_obj = Evaluator.get_model()
    model = model_obj["model"]
    tokenizer = model_obj["tokenizer"]

    # DeBERTa expects a premise/hypothesis pair
    inputs = tokenizer(
        state.context,      # context = premise
        state.answer,       # generated answer = hypothesis
        return_tensors="pt",
        truncation=True,
        padding=True
    )

    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)

    # MNLI labels: [contradiction, neutral, entailment]
    score = probs[0, 2].item()  # entailment probability as evaluation score
    state.score = score

    return state

