import logging
import os

import httpx

logger = logging.getLogger(__name__)


def get_mlflow_endpoint() -> str:
    """
    Find MLflow endpoint, trying environment variable first, then common endpoints

    Parameters
    ----------
    None

    Returns
    -------
    str
        MLflow endpoint as a string
    """
    # Check environment variable first
    if env_uri := os.getenv("MLFLOW_TRACKING_URI"):
        return env_uri

    # Check if we're in a container
    in_container = os.path.exists("/.dockerenv") or "AIRFLOW__CORE__EXECUTOR" in os.environ

    # Define endpoints based on environment
    endpoints = (
        ["http://mlflow:5001", "http://host.docker.internal:5001", "http://172.17.0.1:5001", "http://localhost:5001"]
        if in_container
        else ["http://localhost:5001", "http://127.0.0.1:5001", "http://host.docker.internal:5001"]
    )

    # Test each endpoint
    for endpoint in endpoints:
        try:
            response = httpx.get(endpoint, timeout=2.0)
            response.raise_for_status()
            logger.info(f"MLflow accessible at: {endpoint}")
            return endpoint

        except Exception as e:
            logger.debug(f"MLflow not accessible at {endpoint}: {e}")

    # Return default if none work
    default = "http://mlflow:5001" if in_container else "http://localhost:5001"
    logger.warning(f"Could not connect to MLflow, using default: {default}")
    return default


def get_minio_endpoint() -> str:
    """
    Find MinIO endpoint, trying environment variable first, then common endpoints

    Parameters
    ----------
    None

    Returns
    -------
    str
        MinIO endpoint as a string
    """
    # Check environment variable first
    if env_uri := os.getenv("MLFLOW_S3_ENDPOINT_URL"):
        return env_uri

    # Check if we're in a container
    in_container = os.path.exists("/.dockerenv") or "AIRFLOW__CORE__EXECUTOR" in os.environ

    # Define endpoints based on environment
    endpoints = (
        ["http://minio:9000", "http://host.docker.internal:9000", "http://172.17.0.1:9000", "http://localhost:9000"]
        if in_container
        else ["http://localhost:9000", "http://127.0.0.1:9000", "http://host.docker.internal:9000"]
    )

    # Test each endpoint
    for endpoint in endpoints:
        try:
            response = httpx.get(endpoint, timeout=2.0)
            response.raise_for_status()
            logger.info(f"MinIO accessible at: {endpoint}")
            return endpoint

        except Exception as e:
            logger.debug(f"MinIO not accessible at {endpoint}: {e}")

    # Return default if none work
    default = "http://minio:9000" if in_container else "http://localhost:9000"
    logger.warning(f"Could not connect to MinIO, using default: {default}")
    return default
