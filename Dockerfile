# Stage 1: Builder
FROM python:3.11-slim as builder

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    POETRY_VERSION=1.7.1 \
    PIP_DEFAULT_TIMEOUT=100 \
    PIP_RETRIES=5

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry using the official installer
RUN curl -sSL https://install.python-poetry.org | python3 - --version $POETRY_VERSION
ENV PATH="/root/.local/bin:$PATH"

# Copy only the necessary files for dependency installation
WORKDIR /app
COPY pyproject.toml poetry.lock ./

# Configure Poetry and install only production dependencies
RUN poetry config virtualenvs.create false \
    && poetry config virtualenvs.in-project false \
    && poetry config installer.max-workers 4 \
    && poetry install --no-interaction --no-ansi --no-root --only main --no-cache

# Stage 2: Final image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy virtual environment from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code (excluding development files)
COPY src/ src/
COPY entrypoint.sh /entrypoint.sh

# Set proper permissions
RUN chmod +x /entrypoint.sh

# Set the entrypoint
ENTRYPOINT ["/entrypoint.sh"]

EXPOSE 8000

ENTRYPOINT ["/entrypoint.sh"]