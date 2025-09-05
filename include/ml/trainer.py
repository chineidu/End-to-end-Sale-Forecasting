"""This module provides utilities for model training.
Inspired by: https://github.com/airscholar/astro-salesforecast/blob/main/include/ml_models/train_models.py
"""

import base64
import json
import os
import tempfile
from datetime import datetime
from typing import Any

import joblib
import lightgbm as lgb
import mlflow
import numpy as np
import polars as pl
import polars.selectors as cs
import xgboost
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder, StandardScaler

from include import PACKAGE_PATH, create_logger
from include.config import app_config
from include.ml.diagnostics import diagnose_model_performance
from include.ml.ensemble_model import EnsembleModel
from include.ml.visualization import ModelVisualizer
from include.utilities.feature_engineering import FeatureEngineer
from include.utilities.mlflow_s3_utils import MLflowS3Manager
from include.utilities.mlflow_utils import MLflowManager
from include.utilities.s3_verification import log_s3_verification_results, verify_s3_artifacts

logger = create_logger(__name__)


class ModelTrainer:
    def __init__(self) -> None:
        """
        Initializes a ModelTrainer object.

        The ModelTrainer object is used to train and diagnose the performance of
        machine learning models.

        Parameters
        ----------
        None

        Attributes
        ----------
        model_config : ModelsConfig
            MLflow models configuration
        training_config : TrainingConfig
            Training configuration
        mlflow_manager : MLflowManager
            Manager for MLflow
        feature_engineer : FeatureEngineer
            Feature engineering utilities
        models : dict[str, Any]
            Trained models
        scalers : dict[str, StandardScaler]
            Scalers used for feature scaling
        label_encoders : dict[str, LabelEncoder]
            Label encoders used for categorical encoding
        """
        self.model_config = app_config.models
        self.training_config = app_config.training
        self.mlflow_manager = MLflowManager()
        self.feature_engineer = FeatureEngineer()
        self.models: dict[str, Any] = {}
        self.scalers: dict[str, StandardScaler] = {}
        self.label_encoders: dict[str, LabelEncoder] = {}

    @property
    def run_name(self) -> str:
        """
        Property to get the run name.

        Returns
        -------
        str
            Run name.
        """
        return self.mlflow_manager.get_run_id(None)

    def prepare_data(
        self,
        df: pl.DataFrame,
        target_col: str = "sales",
        group_cols: list[str] | None = None,
        categorical_cols: list[str] | None = None,
    ) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
        """
        Prepare data for training and validation.

        Parameters
        ----------
        df : pl.DataFrame
            Input DataFrame.
        target_col : str, optional
            Target column name. Defaults to "sales".
        group_cols : list[str] | None, optional
            Columns to group by when creating lag and rolling features. Defaults to None.
        categorical_cols : list[str] | None, optional
            Columns to create interaction features for. Defaults to None.

        Returns
        -------
        tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]
            Tuple of (train_df, val_df, test_df) DataFrames.
        """
        required_cols: list[str] = ["date", target_col]
        if group_cols:
            required_cols.extend(group_cols)

        missing_cols: list[str] = list(set(required_cols) - set(df.columns))
        if missing_cols:
            raise ValueError(f"Missing required columns for training: {missing_cols}")

        # Feature engineering
        df_features: pl.DataFrame = self.feature_engineer.create_all_features(
            df,
            target_col=target_col,
            date_col="date",
            group_cols=group_cols,
            categorical_cols=categorical_cols,
        )
        # Sort the data chronologically (for time series)
        df_features = df_features.sort(by=["date"], descending=False)

        # Split data: Use the most recent data for validation (mimick real-world scenario)
        train_size: int = int(len(df_features) * (1 - self.training_config.test_size - self.training_config.validation_size))
        validation_size: int = int(len(df_features) * self.training_config.validation_size)

        # Drop rows with NaN in target column
        train_df: pl.DataFrame = df_features[:train_size].drop_nulls(subset=[target_col])
        val_df: pl.DataFrame = df_features[train_size : train_size + validation_size].drop_nulls(subset=[target_col])
        test_df: pl.DataFrame = df_features[train_size + validation_size :].drop_nulls(subset=[target_col])
        extras: dict[str, Any] = {"train_size": train_size, "validation_size": validation_size, "test_size": len(test_df)}
        logger.info(f"Data split - {json.dumps(extras)}")

        return train_df, val_df, test_df

    def preprocess_features(
        self,
        train_df: pl.DataFrame,
        val_df: pl.DataFrame,
        test_df: pl.DataFrame,
        target_col: str,
        excluded_cols: list[str] = ["date"],  # noqa: B006
    ) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.Series, pl.Series, pl.Series]:
        # Separate features and target
        """
        Preprocess features for training and evaluation.

        This function separates features and target from the input DataFrame, encodes
        categorical features using LabelEncoder, and scales the features using StandardScaler.

        Parameters
        ----------
        train_df : pl.DataFrame
            Training data.
        val_df : pl.DataFrame
            Validation data.
        test_df : pl.DataFrame
            Testing data.
        target_col : str
            Target column name.
        excluded_cols : list[str]
            Columns to exclude from the feature set. Defaults to ["date"].

        Returns
        -------
        X_train_scaled : pl.DataFrame
            Scaled training features.
        X_val_scaled : pl.DataFrame
            Scaled validation features.
        X_test_scaled : pl.DataFrame
            Scaled testing features.
        y_train : pl.Series
            Training target.
        y_val : pl.Series
            Validation target.
        y_test : pl.Series
            Testing target.
        """
        X_train = train_df.drop([target_col] + excluded_cols)
        X_val = val_df.drop([target_col] + excluded_cols)
        X_test = test_df.drop([target_col] + excluded_cols)

        y_train = train_df[target_col]
        y_val = val_df[target_col]
        y_test = test_df[target_col]

        # Encode categorical features
        # Note: sklearn's LabelEncoder raises on unseen labels. We map unseen labels to -1
        # so validation/test data won't break the pipeline. We keep encoders in
        # self.label_encoders to reuse during inference.
        cat_cols: list[str] = X_train.select(cs.string()).columns
        for var in cat_cols:
            train_values = X_train[var].to_list()

            if var not in self.label_encoders:
                le = LabelEncoder()
                encoded_train = le.fit_transform(train_values)
                self.label_encoders[var] = le
            else:
                le = self.label_encoders[var]
                # Transform training values with existing encoder; unknowns here are unlikely
                # but map them to -1 defensively
                try:
                    encoded_train = le.transform(train_values)
                except ValueError:
                    mapping = {c: i for i, c in enumerate(le.classes_)}
                    encoded_train = [mapping.get(v, -1) for v in train_values]

            # Build mapping for known classes to safely encode val/test (unknown -> -1)
            mapping = {c: i for i, c in enumerate(self.label_encoders[var].classes_)}

            def _encode_list(values: list[str], mapping: dict[str, int] = mapping) -> list[int]:
                """Encode a list of categorical values using an existing mapping."""
                return [mapping.get(v, -1) for v in values]

            val_values = X_val[var].to_list()
            test_values = X_test[var].to_list()

            encoded_val = _encode_list(val_values)
            encoded_test = _encode_list(test_values)

            # Attach encoded columns back as small-int (Int8). -1 reserved for unknowns.
            X_train = X_train.with_columns(pl.Series(var, values=encoded_train, dtype=pl.Int8))
            X_val = X_val.with_columns(pl.Series(var, values=encoded_val, dtype=pl.Int8))
            X_test = X_test.with_columns(pl.Series(var, values=encoded_test, dtype=pl.Int8))

        # Track feature columns used for modeling (post-encoding, pre-scaling)
        self.feature_cols: list[str] = X_train.columns  # type: ignore[attr-defined]

        # Scale the features
        scaler = StandardScaler()
        X_train_scaled: pl.DataFrame = pl.DataFrame(scaler.fit_transform(X_train), schema=X_train.columns)
        X_val_scaled: pl.DataFrame = pl.DataFrame(scaler.transform(X_val), schema=X_val.columns)
        X_test_scaled: pl.DataFrame = pl.DataFrame(scaler.transform(X_test), schema=X_test.columns)

        self.scalers["standard"] = scaler

        return (X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test)

    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
        """
        Calculate and return a dictionary with the following metrics:
        - root mean squared error (rmse)
        - mean absolute error (mae)
        - mean absolute percentage error (mape)
        - R-squared (r2)

        Parameters
        ----------
        y_true : np.ndarray
            True labels.
        y_pred : np.ndarray
            Predicted labels.

        Returns
        -------
        metrics : dict[str, float]
            Dictionary with the calculated metrics.
        """
        return {
            "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
            "mae": mean_absolute_error(y_true, y_pred),
            "mape": np.mean(np.abs((y_true - y_pred) / y_true)) * 100,
            "r2": r2_score(y_true, y_pred),
        }

    def train_xgboost(
        self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray
    ) -> xgboost.XGBRegressor:  # type: ignore
        """
        Train an XGBoost model.

        Parameters
        ----------
        X_train : np.ndarray
            Training features.
        y_train : np.ndarray
            Training target.
        X_val : np.ndarray
            Validation features.
        y_val : np.ndarray
            Validation target.

        Returns
        -------
        model : xgboost.XGBRegressor
            Trained model.

        """
        logger.info("Training XGBoost model")

        best_params = self.model_config.xgboost.params
        best_params["early_stopping_rounds"] = 50
        model = xgboost.XGBRegressor(**best_params)  # type: ignore
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=True)

        self.models["xgboost"] = model
        return model

    def train_lightgbm(
        self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray
    ) -> lgb.LGBMRegressor:  # type: ignore
        """
        Train a LightGBM model.

        Parameters
        ----------
        X_train : np.ndarray
            Training features.
        y_train : np.ndarray
            Training target.
        X_val : np.ndarray
            Validation features.
        y_val : np.ndarray
            Validation target.

        Returns
        -------
        model : lgb.LGBMRegressor
            Trained model.
        """
        logger.info("Training LightGBM model")

        best_params = self.model_config["lightgbm"]["params"]
        model = lgb.LGBMRegressor(**best_params)  # type: ignore
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(50)])

        self.models["lightgbm"] = model
        return model

    def train_all_models(
        self,
        train_df: pl.DataFrame,
        val_df: pl.DataFrame,
        test_df: pl.DataFrame,
        target_col: str = "sales",
    ) -> dict[str, dict[str, Any]]:
        """
        Train multiple models (XGBoost, LightGBM, and Prophet (optional)) on the given data, and
        log the results to MLflow. The models are trained on the training data, and evaluated on
        the validation data. The best model is selected based on the validation R2 score, and the
        ensemble model is created using the best model.

        Parameters
        ----------
        train_df : pl.DataFrame
            Training data.
        val_df : pl.DataFrame
            Validation data.
        test_df : pl.DataFrame
            Testing data.
        target_col : str, optional
            Target column name, by default "sales".

        Returns
        -------
        results : dict[str, dict[str, Any]]
            A dictionary containing the results of the model training, including the trained models,
            the metrics, and the predictions.
        """
        results: dict[str, dict[str, Any]] = {}

        # Start MLflow run
        _ = self.mlflow_manager.start_run(
            run_name=f"{app_config.mlflow.experiment_name}_training_{datetime.now().isoformat(timespec='seconds')}",
            tags={"model_type": "ensemble"},
        )

        try:
            # Preprocess data
            X_train_df, X_val_df, X_test_df, y_train_series, y_val_series, y_test_series = self.preprocess_features(
                train_df, val_df, test_df, target_col
            )
            # Convert to Arrays
            X_train: np.ndarray = X_train_df.to_numpy()
            X_val: np.ndarray = X_val_df.to_numpy()
            X_test: np.ndarray = X_test_df.to_numpy()
            y_train: np.ndarray = y_train_series.to_numpy()
            y_val: np.ndarray = y_val_series.to_numpy()
            y_test: np.ndarray = y_test_series.to_numpy()

            # Log data stats
            self.mlflow_manager.log_params(
                {
                    "train_size": len(train_df),
                    "val_size": len(val_df),
                    "test_size": len(test_df),
                    "n_features": X_train.shape[1],
                }
            )

            # Train XGBoost
            xgb_model = self.train_xgboost(X_train, y_train, X_val, y_val)
            xgb_pred = xgb_model.predict(X_test)
            xgb_metrics = self.calculate_metrics(y_test, xgb_pred)

            self.mlflow_manager.log_metrics({f"xgboost_{k}": v for k, v in xgb_metrics.items()})
            self.mlflow_manager.log_model(xgb_model, "xgboost", input_example=X_train_df.head())

            # Log feature importance
            feature_importance: pl.DataFrame = (
                pl.DataFrame({"feature": self.feature_cols, "importance": xgb_model.feature_importances_})
                .sort("importance", descending=True)
                .head(20)
            )

            logger.info(f"Top XGBoost features:\n{feature_importance.to_dicts()}")
            self.mlflow_manager.log_params(
                {
                    f"xgb_top_feature_{idx}": f"{row[0]} ({row[1]:.4f})"
                    for idx, row in enumerate(feature_importance.iter_rows(), start=1)
                }
            )

            results["xgboost"] = {"model": xgb_model, "metrics": xgb_metrics, "predictions": xgb_pred}

            try:
                lgb_model = self.train_lightgbm(X_train, y_train, X_val, y_val)
                lgb_pred = lgb_model.predict(X_test)
                lgb_metrics = self.calculate_metrics(y_test, lgb_pred)

                self.mlflow_manager.log_metrics({f"lightgbm_{k}": v for k, v in lgb_metrics.items()})
                self.mlflow_manager.log_model(lgb_model, "lightgbm", input_example=X_train_df.head())

                # Log feature importance for LightGBM
                lgb_importance: pl.DataFrame = (
                    pl.DataFrame({"feature": self.feature_cols, "importance": lgb_model.feature_importances_})
                    .sort("importance", descending=True)
                    .head(20)
                )
                logger.info(f"Top LightGBM features:\n{lgb_importance.to_dicts()}")
                self.mlflow_manager.log_params(
                    {
                        f"lgb_top_feature_{idx}": f"{row[0]} ({row[1]:.4f})"
                        for idx, row in enumerate(lgb_importance.iter_rows(), start=1)
                    }
                )

                results["lightgbm"] = {"model": lgb_model, "metrics": lgb_metrics, "predictions": lgb_pred}
            except Exception as lgb_err:
                logger.warning(f"Skipping LightGBM due to error: {lgb_err}")

            # Weighted ensemble based on individual model performance (using validation R2)
            # Ensemble: if LightGBM is present, use weighted; otherwise fall back to XGBoost only
            xgb_val_pred = xgb_model.predict(X_val)
            xgb_val_r2 = r2_score(y_val, xgb_val_pred)

            if "lightgbm" in results:
                lgb_val_pred = results["lightgbm"]["model"].predict(X_val)
                lgb_val_r2 = r2_score(y_val, lgb_val_pred)

                # Calculate weights with a minimum weight constraint
                min_weight = 0.2
                xgb_weight = max(min_weight, xgb_val_r2 / (xgb_val_r2 + lgb_val_r2))
                lgb_weight = max(min_weight, lgb_val_r2 / (xgb_val_r2 + lgb_val_r2))
                total_weight = xgb_weight + lgb_weight
                xgb_weight /= total_weight
                lgb_weight /= total_weight
                logger.info(f"Ensemble weights - XGBoost: {xgb_weight:.3f}, LightGBM: {lgb_weight:.3f}")

                ensemble_weights = {"xgboost": xgb_weight, "lightgbm": lgb_weight}
                ensemble_pred = xgb_weight * xgb_pred + lgb_weight * results["lightgbm"]["predictions"]
                ensemble_models = {"xgboost": xgb_model, "lightgbm": results["lightgbm"]["model"]}

            else:
                logger.info("LightGBM not available; using XGBoost-only ensemble")
                ensemble_weights = {"xgboost": 1.0}
                ensemble_pred = xgb_pred
                ensemble_models = {"xgboost": xgb_model}

            if "prophet" in results:
                ensemble_models["prophet"] = results["prophet"]["model"]
                ensemble_weights = {"xgboost": 1 / 3, "lightgbm": 1 / 3, "prophet": 1 / 3}

            ensemble_model = EnsembleModel(ensemble_models, ensemble_weights)

            # Save ensemble model
            self.models["ensemble"] = ensemble_model

            ensemble_metrics = self.calculate_metrics(y_test, ensemble_pred)

            self.mlflow_manager.log_metrics({f"ensemble_{k}": v for k, v in ensemble_metrics.items()})
            self.mlflow_manager.log_model(ensemble_model, "ensemble", input_example=None)

            results["ensemble"] = {"model": ensemble_model, "metrics": ensemble_metrics, "predictions": ensemble_pred}

            # Run diagnostics
            logger.info("Running model diagnostics...")
            test_predictions = {
                "xgboost": xgb_pred if "xgboost" in results else None,
                "lightgbm": lgb_pred if "lightgbm" in results else None,
                "ensemble": ensemble_pred,
            }

            diagnosis = diagnose_model_performance(train_df, val_df, test_df, test_predictions, target_col)

            logger.info("Diagnostic recommendations:")
            for rec in diagnosis["recommendations"]:
                logger.warning(f"- {rec}")

            # Generate visualizations
            logger.info("Generating model comparison visualizations...")
            try:
                self._generate_and_log_visualizations(results, test_df, target_col)
            except Exception as viz_error:
                logger.error(f"Visualization generation failed: {viz_error}", exc_info=True)

            # Save artifacts
            self.save_artifacts()

            # Get current run ID for verification
            current_run_id = mlflow.active_run().info.run_id  # type: ignore

            self.mlflow_manager.end_run()

            logger.info("Syncing artifacts to S3...")
            try:
                s3_manager = MLflowS3Manager()
                s3_manager.sync_mlflow_artifacts_to_s3(current_run_id)
                logger.info("✅ Successfully synced artifacts to S3")

                # Verify S3 artifacts after sync
                logger.info("Verifying S3 artifact storage...")
                verification_results = verify_s3_artifacts(
                    run_id=current_run_id,
                    expected_artifacts=[
                        "models/",
                        "scalers.pkl",
                        "encoders.pkl",
                        "feature_cols.pkl",
                        "visualizations/",
                        "reports/",
                    ],
                )
                log_s3_verification_results(verification_results)

                if not verification_results["success"]:
                    logger.warning("S3 artifact verification failed after sync")
            except Exception as e:
                logger.error(f"❌ Failed to sync artifacts to S3: {e}")

        except Exception as e:
            self.mlflow_manager.end_run(status="FAILED")
            raise e

        return results

    def _generate_and_log_visualizations(self, results: dict[str, Any], test_df: pl.DataFrame) -> None:
        """Generate and log model comparison visualizations to MLflow"""
        try:
            logger.info("Starting visualization generation...")
            visualizer = ModelVisualizer()

            # Extract metrics
            metrics_dict: dict[str, Any] = {}
            for model_name, model_results in results.items():
                if "metrics" in model_results:
                    metrics_dict[model_name] = model_results["metrics"]

            # Prepare predictions data
            predictions_dict: dict[str, Any] = {}
            for model_name, model_results in results.items():
                if "predictions" in model_results and model_results["predictions"] is not None:
                    pred_df: pl.DataFrame = test_df.select(["date"]).clone()
                    pred_df = pred_df.with_columns(pl.Series("predictions", values=model_results["predictions"]))
                    predictions_dict[model_name] = pred_df

            # Extract feature importance if available
            feature_importance_dict = {}
            for model_name, model_results in results.items():
                if model_name in ["xgboost", "lightgbm"] and "model" in model_results:
                    model: Any = model_results["model"]
                    if hasattr(model, "feature_importances_"):
                        importance_df = pl.DataFrame(
                            {"feature": self.feature_cols, "importance": model.feature_importances_}
                        ).sort("importance", descending=True)
                        feature_importance_dict[model_name] = importance_df

            # Create temporary directory for visualizations
            with tempfile.TemporaryDirectory() as temp_dir:
                logger.info(f"Creating visualizations in temporary directory: {temp_dir}")

                # Generate all visualizations
                saved_files = visualizer.create_comprehensive_report(
                    metrics_dict=metrics_dict,
                    predictions_dict=predictions_dict,
                    actual_data=test_df,
                    feature_importance_dict=feature_importance_dict if feature_importance_dict else None,
                    save_dir=temp_dir,
                )

                logger.info(f"Generated {len(saved_files)} visualization files: {list(saved_files.keys())}")

                # Log each visualization to MLflow
                for viz_name, file_path in saved_files.items():
                    if os.path.exists(file_path):
                        mlflow.log_artifact(file_path, "visualizations")  # type: ignore
                        logger.info(f"Logged visualization: {viz_name} from {file_path}")
                    else:
                        logger.warning(f"Visualization file not found: {file_path}")

                # Also create a combined HTML report
                self._create_combined_html_report(saved_files, temp_dir)

                # Log the combined report
                combined_report = os.path.join(temp_dir, "model_comparison_report.html")
                if os.path.exists(combined_report):
                    mlflow.log_artifact(combined_report, "reports")  # type: ignore
                    logger.info("Logged combined HTML report")

        except Exception as e:
            # Don't fail the entire run
            logger.error(f"Failed to generate visualizations: {e}")

    def _create_combined_html_report(self, saved_files: dict[str, str], save_dir: str) -> None:
        """
        Create a combined HTML report with all visualizations.

        The HTML report includes all visualizations in a single page for easy comparison.
        """

        html_content: str = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Model Comparison Report</title>
            <style>
                body {
                    font-family: Arial, sans-serif;
                    margin: 20px;
                    background-color: #f5f5f5;
                }
                h1, h2 {
                    color: #333;
                }
                .section {
                    background-color: white;
                    padding: 20px;
                    margin-bottom: 20px;
                    border-radius: 8px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }
                .timestamp {
                    color: #666;
                    font-size: 14px;
                }
                iframe {
                    width: 100%;
                    height: 800px;
                    border: 1px solid #ddd;
                    border-radius: 4px;
                    margin-top: 10px;
                }
                img {
                    max-width: 100%;
                    height: auto;
                    border-radius: 4px;
                    margin-top: 10px;
                }
            </style>
        </head>
        <body>
            <h1>Sales Forecast Model Comparison Report</h1>
            <p class="timestamp">Generated on: {timestamp}</p>
        """

        html_content: str = html_content.format(timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

        # Add each visualization section
        sections: list[tuple[str, str]] = [
            ("metrics_comparison", "Model Performance Metrics"),
            ("predictions_comparison", "Predictions Comparison"),
            ("residuals_analysis", "Residuals Analysis"),
            ("error_distribution", "Error Distribution"),
            ("feature_importance", "Feature Importance"),
            ("summary", "Summary Statistics"),
        ]

        for key, title in sections:
            if key in saved_files:
                html_content += f'<div class="section"><h2>{title}</h2>'

                # All files are now PNG - base64 encode them
                with open(saved_files[key], "rb") as f:
                    img_data = base64.b64encode(f.read()).decode()
                html_content += f'<img src="data:image/png;base64,{img_data}" alt="{title}">'

                html_content += "</div>"

        html_content += """
        </body>
        </html>
        """

        # Save the combined report
        with open(os.path.join(save_dir, "model_comparison_report.html"), "w") as f:
            f.write(html_content)

    def save_artifacts(self) -> None:
        """
        Saves artifacts to disk in the expected format for MLflow.

        Saves the following artifacts:
        - scalers.pkl: Joblib dump of the scalers used for feature scaling
        - encoders.pkl: Joblib dump of the encoders used for categorical encoding
        - feature_cols.pkl: Joblib dump of the feature column names
        - models/xgboost/xgboost_model.pkl: Joblib dump of the XGBoost model
        - models/lightgbm/lightgbm_model.pkl: Joblib dump of the LightGBM model
        - models/ensemble/ensemble_model.pkl: Joblib dump of the ensemble model

        Also logs the artifacts to MLflow.
        """
        joblib.dump(self.scalers, f"{PACKAGE_PATH}/artifacts/scalers.pkl")
        joblib.dump(self.label_encoders, f"{PACKAGE_PATH}/artifacts/encoders.pkl")
        joblib.dump(self.feature_cols, f"{PACKAGE_PATH}/artifacts/feature_cols.pkl")

        # Save individual models in the expected format
        os.makedirs(f"{PACKAGE_PATH}/artifacts/models/xgboost", exist_ok=True)
        os.makedirs(f"{PACKAGE_PATH}/artifacts/models/lightgbm", exist_ok=True)
        os.makedirs(f"{PACKAGE_PATH}/artifacts/models/ensemble", exist_ok=True)

        if "xgboost" in self.models:
            joblib.dump(self.models["xgboost"], f"{PACKAGE_PATH}/artifacts/models/xgboost/xgboost_model.pkl")

        if "lightgbm" in self.models:
            joblib.dump(self.models["lightgbm"], f"{PACKAGE_PATH}/artifacts/models/lightgbm/lightgbm_model.pkl")

        if "ensemble" in self.models:
            joblib.dump(self.models["ensemble"], f"{PACKAGE_PATH}/artifacts/models/ensemble/ensemble_model.pkl")

        self.mlflow_manager.log_artifacts(f"{PACKAGE_PATH}/artifacts")

        logger.info("Artifacts saved successfully")
