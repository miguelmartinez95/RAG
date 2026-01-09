FROM python:3.11-slim

WORKDIR /app

RUN pip install gradio requests pyyaml

COPY rag/ui/gradio_app.py .

ENV RAG_API_URL=http://rag-bot:80/rag

EXPOSE 7860

CMD ["python", "gradio_app.py"]
