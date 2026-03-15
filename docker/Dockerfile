FROM python:3.11-slim-bookworm

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY medguard/ ./medguard/

# Default install: core + Anthropic + OpenAI integrations
# For NLP tier: docker build --build-arg EXTRAS=anthropic,openai,nlp
ARG EXTRAS="anthropic,openai"
RUN pip install --no-cache-dir ".[${EXTRAS}]"

# Optional: download spaCy model if nlp extras included
# RUN python -m spacy download en_core_web_sm || true

ENV MEDGUARD_CONFIG=/config/config.json
ENV MEDGUARD_API__HOST=0.0.0.0
ENV MEDGUARD_API__PORT=8080

RUN mkdir -p /config /cache
VOLUME ["/config", "/cache"]

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8080/v1/health || exit 1

CMD ["python", "-m", "medguard", "serve"]
