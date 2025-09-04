from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel, Field

from include import PACKAGE_PATH
from include.schemas.input_schema import Float


class MLFlowConfig(BaseModel):
    tracking_uri: str = Field(..., description="MLflow tracking URI")
    experiment_name: str = Field("sales_forecasting", description="MLflow experiment name")
    registry_name: str = Field("sales_forecast_models", description="MLflow registry name")


class XGBoostConfig(BaseModel):
    params: dict[str, Any] = Field(..., description="XGBoost model parameters")


class LightGBMConfig(BaseModel):
    params: dict[str, Any] = Field(..., description="LightGBM model parameters")


class ProphetConfig(BaseModel):
    enabled: bool = Field(False, description="Enable Prophet model")
    params: dict[str, Any] = Field(..., description="Prophet model parameters")


class ModelsConfig(BaseModel):
    xgboost: XGBoostConfig = Field(..., description="XGBoost model configuration")
    lightgbm: LightGBMConfig = Field(..., description="LightGBM model configuration")
    prophet: ProphetConfig = Field(..., description="Prophet model configuration")


class FeaturesConfig(BaseModel):
    date_features: list[str] = Field(..., description="List of date-related features")
    lag_features: list[int] = Field(..., description="List of lag features")
    rolling_features: dict[str, list[int | str]] = Field(..., description="Rolling features configuration")


class ValidationConfig(BaseModel):
    required_columns: list[str] = Field(..., description="List of required columns for validation")
    data_types: dict[str, str] = Field(..., description="Expected data types for each column")
    value_ranges: dict[str, dict[str, Float | int]] = Field(..., description="Valid value ranges for each column")


class TrainingConfig(BaseModel):
    test_size: Float = Field(..., description="Proportion of the dataset to include in the test split")
    validation_size: Float = Field(..., description="Proportion of the dataset to include in the validation split")
    cv_folds: int = Field(..., description="Number of cross-validation folds")
    metrics: list[str] = Field(..., description="List of evaluation metrics")


class InferenceConfig(BaseModel):
    batch_size: int = Field(..., description="Batch size for inference")
    prediction_horizon: int = Field(..., description="Prediction horizon for forecasting")
    confidence_intervals: list[Float] = Field(..., description="Confidence intervals for predictions")


class MonitoringConfig(BaseModel):
    drift_detection: dict[str, str | Float | bool] = Field(..., description="Drift detection configuration")
    performance_monitoring: dict[str, bool | dict[str, Float | int]] = Field(
        ..., description="Performance monitoring configuration"
    )


class Config(BaseModel):
    mlflow: MLFlowConfig = Field(..., description="MLflow configuration")
    models: ModelsConfig = Field(..., description="Models configuration")
    features: FeaturesConfig = Field(..., description="Features configuration")
    validation: ValidationConfig = Field(..., description="Validation configuration")
    training: TrainingConfig = Field(..., description="Training configuration")
    inference: InferenceConfig = Field(..., description="Inference configuration")
    monitoring: MonitoringConfig = Field(..., description="Monitoring configuration")


class AppConfig(BaseModel):
    config: Config = Field(..., description="Application configuration")


config_path: Path = PACKAGE_PATH / "include/config/config.yaml"
config: DictConfig = OmegaConf.load(config_path).config
# Resolve all the variables
resolved_cfg = OmegaConf.to_container(config, resolve=True)
# Validate the config
app_config: Config = AppConfig(config={**dict(resolved_cfg)}).config  # type: ignore
