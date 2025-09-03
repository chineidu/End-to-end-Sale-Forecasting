"""This module provides utilities for managing MLflow artifacts in S3/MinIO.
Inspired by: https://github.com/airscholar/astro-salesforecast/blob/main/include/utils/mlflow_utils.py
"""

import os
from datetime import datetime
from typing import Any, Optional

import joblib
import mlflow
import mlflow.pyfunc
import pandas as pd
import yaml  # type: ignore
from mlflow.pyfunc import PyFuncModel
from mlflow.tracking import MlflowClient  # type: ignore

from src import create_logger
from src.config import app_config, app_settings
from src.utilities.service_discovery import get_mlflow_endpoint

logger = create_logger("mlflow_utils")


class MLflowManager:
    def __init__(self) -> None:
        mlflow_config = app_config.mlflow
        self.tracking_uri = app_settings.mlflow_tracking_uri or get_mlflow_endpoint()

        self.experiment_name = mlflow_config.experiment_name
        self.registry_name = mlflow_config.registry_name

        mlflow.set_tracking_uri(self.tracking_uri)

        try:
            mlflow.set_experiment(self.experiment_name)

        except Exception as e:
            logger.warning(f"Failed to set experiment {self.experiment_name}: {e}")

        self.client = MlflowClient(tracking_uri=self.tracking_uri)

    def start_run(self, run_name: Optional[str] = None, tags: Optional[dict[str, str]] = None) -> str:
        """
        Start a new MLflow run.

        Parameters
        ----------
        run_name : str, optional
            Name of the run to start. If not provided, a default name will be generated.
        tags : dict[str, str], optional
            Tags to be added to the run.

        Returns
        -------
        str
            ID of the started run.
        """
        if run_name is None:
            run_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        run = mlflow.start_run(run_name=run_name, tags=tags)  # type: ignore
        logger.info(f"Started MLflow run: {run.info.run_id}")
        return run.info.run_id

    def log_params(self, params: dict[str, Any]) -> None:
        """
        Log parameters to MLflow.

        Parameters
        ----------
        params : dict[str, Any]
            Parameters to log.
        """
        for key, value in params.items():
            mlflow.log_param(key, value)  # type: ignore

    def log_metrics(self, metrics: dict[str, float], step: Optional[int] = None) -> None:
        """Log metrics to MLflow."""
        for key, value in metrics.items():
            mlflow.log_metric(key, value, step=step)  # type: ignore

    def log_model(
        self,
        model: Any,
        model_name: str,
        input_example: pd.DataFrame | None = None,  # noqa: ARG002
        signature: Any | None = None,  # noqa: ARG002
        registered_model_name: str | None = None,  # noqa: ARG002
    ) -> None:
        """
        Log model to MLflow with compatibility for different versions.

        Parameters
        ----------
        model : Any
            Model to log.
        model_name : str
            Name of the model to log.
        input_example : pd.DataFrame, optional
            Example input to be used for logging model signature.
        signature : Any, optional
            Model signature to be used for logging.
        registered_model_name : str, optional
            Name of the registered model to log.

        Notes
        -----
        Falls back to saving models as artifacts if MLflow model logging fails.
        """
        try:
            # Save model to a temporary file first
            import tempfile

            with tempfile.TemporaryDirectory() as tmpdir:
                model_path = os.path.join(tmpdir, f"{model_name}_model.pkl")
                metadata_path = os.path.join(tmpdir, f"{model_name}_metadata.yaml")

                # Save model
                joblib.dump(model, model_path)

                # Create metadata
                metadata = {
                    "model_type": model_name,
                    "framework": type(model).__module__,
                    "class": type(model).__name__,
                    "timestamp": datetime.now().isoformat(),
                }
                with open(metadata_path, "w") as f:
                    yaml.dump(metadata, f)

                # Log artifacts
                mlflow.log_artifact(model_path, artifact_path=f"models/{model_name}")  # type: ignore
                mlflow.log_artifact(metadata_path, artifact_path=f"models/{model_name}")  # type: ignore

                logger.info(f"Successfully saved {model_name} model as artifact")

        except Exception as e:
            logger.error(f"Failed to log model {model_name}: {e}")

    def log_artifacts(self, artifact_path: str) -> None:
        """
        Log artifacts to MLflow.

        Parameters
        ----------
        artifact_path : str
            Local path to the artifacts to log.
        """
        mlflow.log_artifacts(artifact_path)  # type: ignore

    def log_figure(self, figure: Any, artifact_file: str) -> None:
        """
        Log a figure to MLflow.

        Parameters
        ----------
        figure : Any
            Figure to log. Can be a matplotlib figure, plotly figure, bokeh
            figure, or altair chart.
        artifact_file : str
            Local file path used to write the figure to. The file will be saved
            to the MLflow artifact directory.

        Notes
        -----
        See the `mlflow.log_figure` documentation for more information.
        """
        mlflow.log_figure(figure, artifact_file)  # type: ignore

    def end_run(self, status: str = "FINISHED") -> None:
        """
        End the current MLflow run.

        Parameters
        ----------
        status : str
            Status to end the run with. Defaults to "FINISHED".

        Notes
        -----
        If the run is successful, the artifacts will be synced to S3.
        """
        # Get run ID before ending
        run = mlflow.active_run()
        run_id = run.info.run_id if run else None

        mlflow.end_run(status=status)  # type: ignore
        logger.info("Ended MLflow run")

        # Sync artifacts to S3 after run ends
        if run_id and status == "FINISHED":
            try:
                from src.utilities.mlflow_s3_utils import MLflowS3Manager

                s3_manager = MLflowS3Manager()
                s3_manager.sync_mlflow_artifacts_to_s3(run_id)
                logger.info(f"Synced artifacts to S3 for run {run_id}")

            except Exception as e:
                logger.warning(f"Failed to sync artifacts to S3: {e}")

    def get_best_model(self, metric: str = "rmse", ascending: bool = True) -> dict[str, Any]:
        """
        Get the best model based on a specified metric from the experiment

        Parameters
        ----------
        metric : str
            Metric to use for selecting the best model. Defaults to "rmse".
        ascending : bool
            Whether to sort the runs in ascending or descending order based on the specified metric.
            Defaults to True.

        Returns
        -------
        dict[str, Any]
            A dictionary containing the best run's ID, metrics, and parameters.
        """
        experiment = mlflow.get_experiment_by_name(self.experiment_name)  # type: ignore
        runs = mlflow.search_runs(  # type: ignore
            experiment_ids=[experiment.experiment_id],  # type: ignore
            order_by=[f"metrics.{metric} {'ASC' if ascending else 'DESC'}"],
            max_results=1,
        )

        if len(runs) == 0:
            raise ValueError("No runs found in the experiment")

        best_run = runs.iloc[0]  # type: ignore
        return {
            "run_id": best_run["run_id"],
            "metrics": {col.replace("metrics.", ""): val for col, val in best_run.items() if col.startswith("metrics.")},
            "params": {col.replace("params.", ""): val for col, val in best_run.items() if col.startswith("params.")},
        }

    def load_model(self, model_uri: str) -> PyFuncModel | Any:
        """
        Load model from MLflow or from artifacts.

        Parameters
        ----------
        model_uri : str
            URI of the model to load.

        Returns
        -------
        PyFuncModel | Any
            Loaded model.
        """
        try:
            return mlflow.pyfunc.load_model(model_uri)
        except Exception:
            # Try loading from artifacts
            if "runs:/" in model_uri:
                run_id = model_uri.split("/")[1]
                artifact_path = "/".join(model_uri.split("/")[2:])
                local_path = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path=f"{artifact_path}_model.pkl")  # type: ignore
                return joblib.load(local_path)
            raise ValueError(f"Cannot load model from {model_uri}") from None

    def register_model(self, run_id: str, model_name: str, artifact_path: str) -> str:
        """
        Register model if possible, otherwise return run_id as version.

        Parameters
        ----------
        run_id : str
            MLflow run ID containing the model to register.
        model_name : str
            Name to give the registered model.
        artifact_path : str
            Path to the model artifact within the run.

        Returns
        -------
        str
            Version of the registered model, or the run_id if registration failed.
        """
        try:
            model_uri = f"runs:/{run_id}/{artifact_path}"
            model_version = mlflow.register_model(model_uri, f"{self.registry_name}_{model_name}")  # type: ignore
            return model_version.version
        except Exception:
            logger.warning("Model registration not available, using run_id as version")
            return run_id

    def transition_model_stage(self, model_name: str, version: str, stage: str) -> None:
        """
        Transition model to a new stage.

        Parameters
        ----------
        model_name : str
            Name of the model to transition.
        version : str
            Version of the model to transition.
        stage : str
            Name of the stage to transition to.
        """
        try:
            self.client.transition_model_version_stage(
                name=f"{self.registry_name}_{model_name}", version=version, stage=stage
            )
        except Exception:
            logger.warning("Model stage transition not available")

    def get_latest_model_version(self, model_name: str, stage: str | None = None) -> dict[str, Any]:
        """
        Get the latest model version from the registry.

        If the model is not found, fall back to finding the best run.

        Parameters
        ----------
        model_name : str
            Name of the model to retrieve.
        stage : str | None
            Optional - stage to filter the model versions by.

        Returns
        -------
        dict[str, Any]
            A dictionary containing the version, stage, run_id and source of the model.
        """
        try:
            filter_string = f"name='{self.registry_name}_{model_name}'"
            if stage:
                filter_string += f" AND current_stage='{stage}'"

            versions = self.client.search_model_versions(filter_string)
            if not versions:
                raise ValueError(f"No model versions found for {model_name}")

            # Find the latest model version
            latest_version = max(versions, key=lambda x: int(x.version))
            return {
                "version": latest_version.version,
                "stage": latest_version.current_stage,
                "run_id": latest_version.run_id,
                "source": latest_version.source,
            }
        except Exception:
            logger.debug("No model versions found in the registry, falling back to finding the best run")
            # Fallback to finding the best run
            best_model = self.get_best_model()

            return {
                "version": best_model["run_id"],
                "stage": "None",
                "run_id": best_model["run_id"],
                "source": f"runs:/{best_model['run_id']}/models",
            }
