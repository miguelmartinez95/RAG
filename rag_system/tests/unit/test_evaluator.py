from rag.app.evaluator import evaluate, Evaluator

class DummyTokenizer:
    def __call__(self, text, return_tensors=None):
        return {"input_ids": [1]}
    def decode(self, ids, skip_special_tokens=True):
        return "7"

class DummyModel:
    def generate(self, **kwargs):
        return [[1]]

def test_evaluate(monkeypatch):
    monkeypatch.setattr(
        Evaluator,
        "get_models",
        lambda: {
            "model": DummyModel(),
            "tokenizer": DummyTokenizer()
        }
    )

    state = {
        "query": "What is AI?",
        "context": "AI is artificial intelligence",
        "answer": "AI is intelligence"
    }

    result = evaluate(state)

    assert "score" in result
    assert result["score"] == 7.0
