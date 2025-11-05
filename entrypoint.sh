#!/bin/bash
set -e

# Run database migrations if needed
# python -m alembic upgrade head

# Start the application
exec uvicorn src.api.main:app --host 0.0.0.0 --port 8000