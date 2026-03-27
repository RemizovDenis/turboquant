# ============================================================
# Stage 1: Builder
# ============================================================
ARG CUDA=false

FROM python:3.11-slim AS builder

WORKDIR /build
COPY pyproject.toml README.md ./
COPY turboquant/ turboquant/

RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir build \
 && python -m build --wheel --outdir /build/dist

# ============================================================
# Stage 2: Runtime
# ============================================================
FROM python:3.11-slim AS runtime

ARG CUDA=false

# Install CUDA runtime libs conditionally
RUN if [ "$CUDA" = "true" ]; then \
      apt-get update && apt-get install -y --no-install-recommends \
        libcudnn8 libcublas-12-0 \
      && rm -rf /var/lib/apt/lists/*; \
    fi

# Create non-root user
RUN groupadd -r turboquant && useradd -r -g turboquant -m turboquant

WORKDIR /app

# Install the wheel
COPY --from=builder /build/dist/*.whl /tmp/
RUN pip install --no-cache-dir /tmp/*.whl \
 && rm -rf /tmp/*.whl

# Copy .env.example as reference
COPY .env.example /app/.env.example

USER turboquant

# Default: run the Ollama proxy
EXPOSE 11435

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:11435/tq/status')" || exit 1

CMD ["turboquant-proxy"]
