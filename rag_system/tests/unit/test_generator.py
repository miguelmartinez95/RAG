import pytest
from rag.app.generator import generate_answer, GeneratorModel

class DummyTokenizer:
    def __call__(self, text, return_tensors=None):
        return {"input_ids": [1, 2, 3]}
    def decode(self, ids, skip_special_tokens=True):
        return "This is a test answer"

class DummyModel:
    def generate(self, **kwargs):
        return [[1, 2, 3]]

@pytest.fixture(autouse=True)
def mock_generator_model(monkeypatch):
    monkeypatch.setattr(
        GeneratorModel,
        "get_models",
        lambda: {
            "model": DummyModel(),
            "tokenizer": DummyTokenizer()
        }
    )

def test_generate_answer():
    state = {"query": "Test question", "context": "Test context"}
    result = generate_answer(state)

    assert "answer" in result
    assert isinstance(result["answer"], str)
    assert "test answer" in result["answer"].lower()
