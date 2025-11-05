# Stage 1: Builder - Install build dependencies
FROM python:3.11-slim as builder

# Set environment variables
ENV \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    POETRY_VERSION=1.7.1 \
    PIP_DEFAULT_TIMEOUT=300 \
    PIP_RETRIES=5 \
    PIP_FIND_LINKS=https://pypi.org/simple \
    POETRY_VIRTUALENVS_CREATE=false \
    POETRY_NO_INTERACTION=1 \
    POETRY_NO_DEV=1 \
    PIP_NO_DEPS=0

# Install build dependencies
RUN set -ex \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        libpq-dev \
        curl \
        git \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir "poetry==$POETRY_VERSION"

WORKDIR /app

# Copy only dependency files first for better layer caching
COPY pyproject.toml poetry.lock ./

# Install only production dependencies
RUN set -ex \
    && poetry install --no-interaction --no-ansi --no-root --only main --no-cache --extras "full" \
    # Clean up Python cache and build files
    && find /usr/local -type d -name '__pycache__' -exec rm -rf {} + \
    && find /usr/local -type d -name 'tests' -exec rm -rf {} + \
    && find /usr/local -name '*.pyc' -delete \
    && find /usr/local -name '*.pyo' -delete \
    && find /usr/local -name '*.a' -delete \
    && find /usr/local -name '*.c' -delete \
    # Clean up build tools
    && apt-get remove -y --auto-remove build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && rm -rf /tmp/* /var/tmp/*

# Stage 2: Runtime - Create final image
FROM python:3.11-slim

# Set environment variables
ENV \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONFAULTHANDLER=1 \
    PYTHONHASHSEED=random \
    PYTHONPATH=/app \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=100 \
    PYTHONUTF8=1

# Install runtime dependencies and clean up in one layer
RUN set -ex \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        libgomp1 \
        libpq5 \
        curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && rm -rf /var/cache/apt/archives/* \
    && rm -rf /usr/share/man/* \
    && rm -rf /tmp/* /var/tmp/*

WORKDIR /app

# Create non-root user early for better layer caching
RUN set -ex \
    && useradd -m -u 1000 appuser \
    && mkdir -p /app \
    && chown -R appuser:appuser /app

# Copy only the necessary files from builder
COPY --from=builder --chown=appuser:appuser /usr/local /usr/local

# Copy application code
COPY --chown=appuser:appuser src/ src/
COPY --chown=appuser:appuser entrypoint.sh /entrypoint.sh

# Set proper permissions and switch to non-root user
RUN chmod +x /entrypoint.sh
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose the port the app runs on
EXPOSE 8000

# Command to run the application
ENTRYPOINT ["/entrypoint.sh"]
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]