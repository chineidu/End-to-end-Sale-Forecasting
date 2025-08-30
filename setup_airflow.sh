#!/bin/bash

# Get the current directory dynamically
CURRENT_DIR=$(pwd)

# Set AIRFLOW_HOME to your project directory
export AIRFLOW_HOME="$CURRENT_DIR/airflow"

# Activate virtual environment
source "$CURRENT_DIR/.venv/bin/activate"

airflow dags reserialize

echo "AIRFLOW_HOME set to: $AIRFLOW_HOME"
echo "Virtual environment activated from: $CURRENT_DIR/.venv"
echo "You can now run airflow commands like:"
echo "  Start all airflow components"
echo "  airflow standalone"
echo "  OR to start the API server"
echo "  airflow api-server --port 8080"
echo "  airflow dags list"
echo ""

echo "To start Airflow UI, run:"
echo "  airflow standalone"
echo "Then visit: http://localhost:8080"
echo ""

echo "To view the DAGs folder, run:"
echo "  airflow config get-value core dags_folder"
echo ""