import gradio as gr
import requests
import os
import json

RAG_API_URL = os.getenv("RAG_API_URL", "http://rag-bot:80/rag")

#def query_rag_stream(query: str):
#    answer_text = ""
#    context_text = ""
#    score_value = None
#
#    with requests.post(
#        RAG_API_URL,
#        json={"query": query},
#        stream=True,
#        timeout=600,
#    ) as r:
#        r.raise_for_status()
#
#        for line in r.iter_lines(decode_unicode=True):
#            if not line:
#                continue
#
#            msg = json.loads(line)
#
#            if msg["type"] == "meta":
#                context_text = msg.get("context", "")
#                score_value = msg.get("score", None)
#
#                # Yield once so context/score appear early
#                yield answer_text, score_value, context_text
#
#            elif msg["type"] == "token":
#                answer_text += msg["value"]
#                yield answer_text, score_value, context_text
#
#            elif msg["type"] == "done":
#                break

def query_rag(query: str):
    payload = {"query": query}

    response = requests.post(
        RAG_API_URL,
        json=payload,
        timeout=300  # ⬅️ 3 minutes
    )
    response.raise_for_status()

    data = response.json()

    return (
        data["answer"],
        data["score"],
        "\n\n".join(
            f"[{d['source']}]\n{d['content']}"
            for d in data["documents"]
        )
    )

with gr.Blocks(title="RAG Assistant") as demo:
    gr.Markdown("# 📚 RAG Question Answering")

    query = gr.Textbox(
        label="Ask a question",
        placeholder="What is the document about?"
    )

    answer = gr.Textbox(label="Answer")
    score = gr.Number(label="Evaluation Score")
    context = gr.Textbox(label="Retrieved Context", lines=10)

    submit = gr.Button("Ask")

    submit.click(
        fn=query_rag,
        inputs=query,
        outputs=[answer, score, context]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
