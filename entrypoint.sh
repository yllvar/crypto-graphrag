#!/bin/bash
set -e

# Start the application
exec uvicorn src.api.main:app --host 0.0.0.0 --port 8000
