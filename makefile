.PHONY: help type-check format lint lint-fix \
	check all clean format-fix ci-check \
	lint-verbose start-airflow stop-airflow

# Use bash with strict error handling
.SHELLFLAGS := -ec

# Default target when just running "make"
all: format-fix

# ===== ENVIRONMENT VARIABLES =====
COMPOSE_FILE := "docker-compose.yml"

help:
	@echo "Ruff Formatting and Linting Makefile"
	@echo ""
	@echo "Usage:"
	@echo "  make type-check       			Run type checking with MyPy"
	@echo "  make format           			Format code using Ruff"
	@echo "  make lint             			Run Ruff linter without fixing issues"
	@echo "  make lint-fix         			Run Ruff linter and fix issues automatically"
	@echo "  make lint-verbose     			Run Ruff linter with verbose output"
	@echo "  make format-fix       			Format code and fix linting issues (one command)"
	@echo "  make check            			Run both formatter and linter without fixing"
	@echo "  make ci-check         			Run all checks for CI (type checking, formatting, linting)"
	@echo "  make airflow-startup         		Run Airflow startup script"
	@echo "  make start-airflow         			Start Airflow components"
	@echo "  make check-airflow-config         		Check Airflow configuration and DAGs folder"
	@echo "  make stop-airflow          			Stop Airflow components"
	@echo "  make all              			Same as format-fix (default)"
	@echo "  make clean            			Clean all cache files"
	@echo "  make help             			Show this help message"

# Type checking with MyPy
type-check:
	@echo "Running type checking with MyPy..."
	@bash -c "uv run -m mypy ."
	@echo "Type checking completed."

# Format code with Ruff
format:
	@echo "Formatting code with Ruff..."
	@bash -c "uv run -m ruff format ."

# Lint code with Ruff (no fixing)
lint:
	@echo "Linting code with Ruff..."
	@bash -c "uv run -m ruff check ."

# Lint code with Ruff and fix issues
lint-fix:
	@echo "Linting code with Ruff and applying fixes..."
	@bash -c "uv run -m ruff check --fix ."

# Lint code with Ruff (verbose output)
lint-verbose:
	@echo "Running verbose linting..."
	@bash -c "uv run -m ruff check --verbose ."

# Format and fix in a single command
format-fix: type-check format lint-fix

# Run format and lint without fixing (good for CI)
check:
	@echo "Running full code check (format and lint)..."
	@bash -c "uv run -m ruff format --check ."
	@bash -c "uv run -m ruff check ."

# Complete CI check with type checking
ci-check: type-check check

# ==== Airflow commands ====
airflow-startup:
	@echo "Running Airflow startup script..."
	@bash -c "source ./setup_airflow.sh"

start-airflow: airflow-startup
	@echo "Running Airflow standalone..."
	@bash -c "export AIRFLOW_HOME=$$(pwd)/airflow && source .venv/bin/activate && airflow standalone"

check-airflow-config:
	@echo "Checking Airflow configuration..."
	@bash -c "export AIRFLOW_HOME=$$(pwd)/airflow && source .venv/bin/activate && echo 'AIRFLOW_HOME:' \$$AIRFLOW_HOME && airflow config get-value core dags_folder"

stop-airflow:
	@echo "Stopping Airflow..."
	@bash -c "pkill -f 'airflow' || echo 'Airflow not running'"

# Clean cache files
clean:
	@echo "Cleaning cache files..."
	rm -rf .mypy_cache .ruff_cache __pycache__ .pytest_cache
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	@echo "Cache cleaned."