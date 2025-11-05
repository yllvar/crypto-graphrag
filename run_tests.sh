#!/bin/bash

# Navigate to the project directory
cd "$(dirname "$0")"

# Update dependencies
poetry update

# Install the package in development mode
poetry install

# Set environment variables
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Run the tests with verbose output
poetry run pytest tests/unit/llm/ -v -s

echo "Tests completed. Exit code: $?"
