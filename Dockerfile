FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN pip install uv

ENV UV_PROJECT_ENVIRONMENT=/app/.venv
ENV HF_HOME=/app/.cache/huggingface
ENV TORCH_HOME=/app/.cache/torch
ENV HF_TOKEN=""

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen

COPY app/ ./app/

RUN mkdir -p /app/.cache/huggingface /app/.cache/torch

RUN useradd -m -u 1001 appuser && chown -R appuser /app
USER appuser

EXPOSE 8000

ENV PATH="/app/.venv/bin:$PATH"
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
