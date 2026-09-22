FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8000 \
    MODEL_URI=/app/model \
    CIF_LC_MODEL_URI=/app/model_lc \
    CIF_ENV=production

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
COPY deploy/model /app/model
COPY deploy/model_lending_club /app/model_lc
COPY deploy/huggingface/run.py ./run.py

RUN pip install --no-cache-dir .

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/v1/health').status == 200 else 1)"

CMD ["sh", "-c", "uvicorn run:app --host 0.0.0.0 --port ${PORT}"]
