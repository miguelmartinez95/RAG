#!/bin/bash
set -e
exec uvicorn rag.app.main:app --host 0.0.0.0 --port 8080 --workers 1
