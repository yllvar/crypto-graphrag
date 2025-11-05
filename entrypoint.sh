#!/bin/bash
set -e

# Enable debugging
set -x

# Wait for database to be ready (if needed)
# Example for PostgreSQL:
# while ! nc -z $DB_HOST $DB_PORT; do
#   echo "Waiting for PostgreSQL..."
#   sleep 1
# done

# Run database migrations if needed
# if [ "$RUN_MIGRATIONS" = "true" ]; then
#   echo "Running database migrations..."
#   python -m alembic upgrade head
# fi

# Start the application
echo "Starting application..."
exec "$@"
exec uvicorn src.api.main:app --host 0.0.0.0 --port 8000