"""This module provides utilities for model diagnostics.
Inspired by: https://github.com/airscholar/astro-salesforecast/blob/main/include/ml_models/diagnostics.py
"""

from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import polars.selectors as cs

from src import create_logger

logger = create_logger(__name__)


def diagnose_model_performance(
    train_df: pl.DataFrame,
    val_df: pl.DataFrame,
    test_df: pl.DataFrame,
    predictions: dict[str, np.ndarray],
    target_col: str = "sales",
) -> dict[str, Any]:
    """
    Diagnose why models are underperforming.

    This function performs the following checks:

    1. **Data quality**:
        - Checks for outliers in target variable.
        - Checks for perfect correlations between features.
    2. **Distribution shift**:
        - Checks if there's a significant distribution shift in the target variable.
    3. **Prediction analysis**:
        - Checks the distribution of predictions.
        - Checks for extreme predictions (very low or high).
        - Calculates residuals and checks for outliers.
    4. **Feature importance check**:
        - Checks if there are too many features.
    5. **Data leakage check**:
        - Checks if there are perfect correlations between features and target.
    6. **Sample size check**:
        - Checks if the training set is small.
    7. **Target variable analysis**:
        - Checks if there are many zero sales.

    Parameters
    ----------
    train_df : pl.DataFrame
        The training data.
    val_df : pl.DataFrame
        The validation data.
    test_df : pl.DataFrame
        The test data.
    predictions : dict[str, np.ndarray]
        The predictions of each model.
    target_col : str, optional
        The target variable. Defaults to "sales".

    Returns
    -------
    dict[str, Any]
        A dictionary containing the results of the diagnosis.
    """

    diagnosis: dict[str, Any] = {
        "data_quality": {},
        "distribution_shift": {},
        "prediction_analysis": {},
        "recommendations": [],
    }

    # 1. Check data quality
    logger.info("Checking data quality...")

    # Check for outliers in target
    y_train = train_df[target_col]
    y_val = val_df[target_col]
    y_test = test_df[target_col]

    train_outliers = detect_outliers(y_train)
    val_outliers = detect_outliers(y_val)
    test_outliers = detect_outliers(y_test)

    diagnosis["data_quality"]["train_outliers"] = train_outliers
    diagnosis["data_quality"]["val_outliers"] = val_outliers
    diagnosis["data_quality"]["test_outliers"] = test_outliers

    # 2. Check for distribution shift
    logger.info("Checking for distribution shift...")

    train_mean: float = y_train.mean()  # type: ignore
    val_mean: float = y_val.mean()  # type: ignore
    test_mean: float = y_test.mean()  # type: ignore
    train_std: float = y_train.std()  # type: ignore
    val_std: float = y_val.std()  # type: ignore
    test_std: float = y_test.std()  # type: ignore

    diagnosis["distribution_shift"]["train_stats"] = {"mean": train_mean, "std": train_std}
    diagnosis["distribution_shift"]["val_stats"] = {"mean": val_mean, "std": val_std}
    diagnosis["distribution_shift"]["test_stats"] = {"mean": test_mean, "std": test_std}

    # Check if there's significant shift
    mean_shift_val: float = abs(val_mean - train_mean) / train_mean
    mean_shift_test: float = abs(test_mean - train_mean) / train_mean

    if mean_shift_val > 0.2:
        diagnosis["recommendations"].append(
            f"Significant distribution shift in validation set (mean shift: {mean_shift_val:.1%})"
        )
    if mean_shift_test > 0.2:
        diagnosis["recommendations"].append(
            f"Significant distribution shift in test set (mean shift: {mean_shift_test:.1%})"
        )

    # 3. Analyze predictions
    logger.info("Analyzing predictions...")

    for model_name, pred in predictions.items():
        if pred is not None:
            # Check prediction distribution
            pred_mean = pred.mean()
            pred_std = pred.std()

            # Check for extreme predictions
            extreme_low: int = (pred < y_test.min() * 0.5).sum()  # type: ignore
            extreme_high: int = (pred > y_test.max() * 1.5).sum()  # type: ignore

            # Calculate residuals
            residuals = y_test - pred

            diagnosis["prediction_analysis"][model_name] = {
                "pred_mean": pred_mean,
                "pred_std": pred_std,
                "extreme_low_count": extreme_low,
                "extreme_high_count": extreme_high,
                "residual_mean": residuals.mean(),
                "residual_std": residuals.std(),
                "mape": np.mean(np.abs(residuals / y_test)) * 100,
            }

    # 4. Feature importance check
    feature_cols = [col for col in train_df.columns if col != target_col and col != "date"]
    diagnosis["data_quality"]["n_features"] = len(feature_cols)

    if len(feature_cols) > 50:
        diagnosis["recommendations"].append(
            f"High number of features ({len(feature_cols)}). Consider more aggressive feature selection."
        )

    # 5. Check for data leakage
    # Look for perfect correlations
    numeric_cols: list[str] = train_df.select(cs.numeric()).columns
    numeric_cols = [col for col in numeric_cols if col != target_col]

    if len(numeric_cols) > 0:
        correlations = train_df[numeric_cols].to_pandas().corrwith(train_df[target_col])
        high_corr = correlations[abs(correlations) > 0.95]

        if len(high_corr) > 0:
            diagnosis["recommendations"].append(
                f"Potential data leakage: {len(high_corr)} features have >95% correlation with target"
            )
            diagnosis["data_quality"]["high_correlation_features"] = high_corr.to_dict()

    # 6. Sample size check
    if len(train_df) < 1000:
        diagnosis["recommendations"].append(
            f"Small training set ({len(train_df)} samples). Consider generating more data."
        )

    # 7. Target variable analysis
    target_zeros = (y_train == 0).sum()
    if target_zeros > len(y_train) * 0.1:
        diagnosis["recommendations"].append(
            f"Many zero sales ({target_zeros} in training). Consider log transformation or zero-inflated models."
        )

    return diagnosis


def detect_outliers(data: pl.Series, method: str = Literal["iqr"]) -> dict[str, Any]:
    """
    Detect outliers in data using the Interquartile Range (IQR) method.

    Parameters
    ----------
    data : pl.Series
        The input data.
    method : str, optional
        The method to use for outlier detection. Currently, only "iqr" is supported.

    Returns
    -------
    dict[str, Any]
        A dictionary containing the outlier statistics.
    """

    if method == "iqr":
        # Calculate the first quartile (Q1) and third quartile (Q3)
        Q1: float = data.quantile(0.25)  # type: ignore
        Q3: float = data.quantile(0.75)  # type: ignore

        # Calculate the Interquartile Range (IQR)
        IQR: float = Q3 - Q1

        # Calculate the lower and upper bounds
        lower_bound: float = Q1 - 1.5 * IQR
        upper_bound: float = Q3 + 1.5 * IQR

        # Find outliers that are outside the bounds
        outliers: pl.Series = data.filter((data < lower_bound) | (data > upper_bound))

        # Return the outlier statistics
        return {
            "count": len(outliers),
            "percentage": round(len(outliers) / len(data) * 100, 2),
            "lower_bound": lower_bound,
            "upper_bound": upper_bound,
            "min_outlier": outliers.min() if len(outliers) > 0 else None,
            "max_outlier": outliers.max() if len(outliers) > 0 else None,
        }

    # If the method is not supported, return an empty dictionary
    return {}


def plot_diagnostic_charts(
    train_df: pl.DataFrame,
    val_df: pl.DataFrame,
    test_df: pl.DataFrame,
    predictions: dict[str, np.ndarray],
    target_col: str = "sales",
    save_path: str | None = None,
) -> plt.Figure:
    """
    Plot diagnostic charts for a regression problem.

    Parameters
    ----------
    train_df : pl.DataFrame
        Training data.
    val_df : pl.DataFrame
        Validation data.
    test_df : pl.DataFrame
        Testing data.
    predictions : dict[str, np.ndarray]
        Predictions from different models.
    target_col : str, optional
        Name of the target column, by default "sales".
    save_path : str | None, optional
        Path to save the figure, by default None.

    Returns
    -------
    plt.Figure
        The figure containing the diagnostic charts.
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # 1. Target distribution across splits
    ax = axes[0, 0]
    ax.hist(train_df[target_col], bins=50, alpha=0.5, label="Train", density=True)
    ax.hist(val_df[target_col], bins=50, alpha=0.5, label="Val", density=True)
    ax.hist(test_df[target_col], bins=50, alpha=0.5, label="Test", density=True)
    ax.set_title("Target Distribution Across Splits")
    ax.set_xlabel(target_col)
    ax.legend()

    # 2. Predictions vs Actual
    ax = axes[0, 1]
    y_test = test_df[target_col]
    for model_name, pred in predictions.items():
        if pred is not None:
            ax.scatter(y_test, pred, alpha=0.5, label=model_name)
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--", label="Perfect")
    ax.set_title("Predictions vs Actual")
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.legend()

    # 3. Residual distribution
    ax = axes[1, 0]
    for model_name, pred in predictions.items():
        if pred is not None:
            residuals = y_test - pred
            ax.hist(residuals, bins=50, alpha=0.5, label=model_name, density=True)
    ax.set_title("Residual Distribution")
    ax.set_xlabel("Residuals")
    ax.legend()

    # 4. Time series of actual vs predicted
    ax = axes[1, 1]
    if "date" in test_df.columns:
        dates = test_df["date"]
        ax.plot(dates, y_test, "k-", label="Actual", linewidth=2)
        for model_name, pred in predictions.items():
            if pred is not None:
                ax.plot(dates, pred, "--", label=model_name, alpha=0.7)
        ax.set_title("Time Series: Actual vs Predicted")
        ax.set_xlabel("Date")
        ax.legend()
        ax.tick_params(axis="x", rotation=45)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)

    return fig
