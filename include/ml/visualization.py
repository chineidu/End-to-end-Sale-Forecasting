"""This module provides visualization utilities for comparing machine learning model performance.
Copied from: https://github.com/airscholar/astro-salesforecast/blob/main/include/ml_models/model_visualization.py
"""

import os
from typing import Any

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import polars as pl
from scipy import stats

from include import PACKAGE_PATH, create_logger

logger = create_logger(__name__)


class ModelVisualizer:
    """Create comprehensive visualizations for model comparison and analysis"""

    def __init__(self, style: str = "seaborn-v0_8-darkgrid") -> None:
        """Initialize the visualizer with plotting style"""
        try:
            plt.style.use(style)
        except Exception:
            plt.style.use("seaborn-v0_8")

        self.colors = {
            "xgboost": "#FF6B6B",
            "lightgbm": "#A7D7D4",
            "prophet": "#99D145",
            "ensemble": "#198050",
            "actual": "#17222E",
        }

    def create_metrics_comparison_chart(
        self, metrics_dict: dict[str, dict[str, float]], save_path: str | None = None
    ) -> plt.Figure:
        """Create a comparison chart for model metrics.

        Parameters
        ----------
        metrics_dict : dict[str, dict[str, float]]
            Dictionary with model names as keys and another dictionary with metric names as keys and their values as values.
        save_path : str | None
            Path to save the figure. If None, the figure is not saved.

        Returns
        -------
        plt.Figure
            The figure containing the comparison chart.
        """
        # Prepare data
        models: list[str] = list(metrics_dict.keys())

        # Create matplotlib figure
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle("Model Performance Metrics Comparison", fontsize=16)

        # Define metrics to plot
        metrics_to_plot: list[tuple[str, str, bool, Any]] = [
            ("rmse", "RMSE", True, axes[0, 0]),
            ("mae", "MAE", True, axes[0, 1]),
            ("mape", "MAPE (%)", True, axes[1, 0]),
            ("r2", "R² Score", False, axes[1, 1]),  # Higher is better for R²
        ]

        for metric, title, lower_better, ax in metrics_to_plot:
            values: list[float] = [metrics_dict[model].get(metric, 0) for model in models]
            colors: list[str] = [self.colors.get(model.lower(), "#95A5A6") for model in models]

            # Create bar chart
            bars = ax.bar(models, values, color=colors, alpha=0.7)

            # Add value labels on bars
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2.0, height, f"{value:.3f}", ha="center", va="bottom")

            # Highlight best model
            if lower_better:
                best_idx: int = values.index(min(values))
            else:
                best_idx = values.index(max(values))

            bars[best_idx].set_edgecolor("green")
            bars[best_idx].set_linewidth(3)

            ax.set_title(f"{title} Comparison")
            ax.set_ylabel(title)
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, max(values) * 1.15)  # Add space for labels

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            plt.close()
            logger.info(f"Saved metrics comparison chart to {save_path}")

        return fig

    def create_predictions_comparison_chart(
        self,
        predictions_dict: dict[str, pd.DataFrame] | dict[str, pl.DataFrame],
        actual_data: pd.DataFrame | pl.DataFrame,
        date_col: str = "date",
        target_col: str = "sales",
        save_path: str | None = None,
    ) -> plt.Figure:
        """
        Create a comparison chart of model predictions.

        Parameters
        ----------
        predictions_dict : dict[str, pd.DataFrame] | dict[str, pl.DataFrame]
            Dictionary with model names as keys and predictions as values.
        actual_data : pd.DataFrame | pl.DataFrame
            Actual data to compare predictions with.
        date_col : str, optional
            Column name for the date, by default "date".
        target_col : str, optional
            Column name for the target variable, by default "sales".
        save_path : str | None
            Path to save the figure, by default None.

        Returns
        -------
        plt.Figure
            The figure containing the comparison chart.
        """
        # Convert polars dataframes to pandas dataframes for plotting
        if isinstance(actual_data, pl.DataFrame):
            actual_data = actual_data.to_pandas()

        if isinstance(predictions_dict, dict):
            # Convert polars DataFrames to pandas, leave pandas DataFrames unchanged
            predictions_dict = {
                k: (v.to_pandas() if isinstance(v, pl.DataFrame) else v) for k, v in predictions_dict.items()
            }

        fig, ax = plt.subplots(figsize=(14, 8))

        # Plot actual data
        ax.plot(
            actual_data[date_col],
            actual_data[target_col],
            color=self.colors["actual"],
            linewidth=3,
            label="Actual",
            alpha=0.8,
        )

        # Plot predictions for each model
        for model_name, pred_df in predictions_dict.items():
            color: str = self.colors.get(model_name.lower(), "#95A5A6")

            ax.plot(
                pred_df[date_col],
                pred_df["prediction"],
                color=color,
                linewidth=2,
                label=f"{model_name} Prediction",
                alpha=0.7,
            )

            # Plot confidence intervals if available
            if "prediction_lower" in pred_df.columns and "prediction_upper" in pred_df.columns:
                ax.fill_between(
                    pred_df[date_col], pred_df["prediction_lower"], pred_df["prediction_upper"], color=color, alpha=0.1
                )

        ax.set_title("Model Predictions Comparison", fontsize=16)
        ax.set_xlabel("Date", fontsize=12)
        ax.set_ylabel(target_col.capitalize(), fontsize=12)
        ax.legend(loc="upper left", framealpha=0.8)
        ax.grid(True, alpha=0.3)

        # Format x-axis dates
        fig.autofmt_xdate()

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            plt.close()
            logger.info(f"Saved predictions comparison chart to {save_path}")

        return fig

    def create_residuals_analysis(
        self,
        predictions_dict: dict[str, pd.DataFrame] | dict[str, pl.DataFrame],
        actual_data: pd.DataFrame | pl.DataFrame,
        target_col: str = "sales",
        save_path: str | None = None,
    ) -> plt.Figure:
        """
        Create residuals analysis plots.

        Parameters
        ----------
        predictions_dict : dict[str, pd.DataFrame] | dict[str, pl.DataFrame]
            Dictionary of model predictions.
        actual_data : pd.DataFrame | pl.DataFrame
            Actual data.
        target_col : str, optional
            Name of the target column, by default "sales".
        save_path : str | None, optional
            Path to save the figure, by default None.

        Returns
        -------
        plt.Figure
            The figure containing the residuals analysis plots.
        """
        if isinstance(actual_data, pl.DataFrame):
            actual_data = actual_data.to_pandas()

        if isinstance(predictions_dict, dict):
            predictions_dict = {
                k: (v.to_pandas() if isinstance(v, pl.DataFrame) else v) for k, v in predictions_dict.items()
            }

        # Calculate residuals for each model
        residuals_data: dict[str, pd.Series] = {}
        merged_data: dict[str, pd.DataFrame] = {}  # Keep track of merged dataframes

        for model_name, pred_df in predictions_dict.items():
            # Ensure dates are comparable - coerce to datetime
            try:
                actual_dates = actual_data["date"].astype("datetime64[ns]")
            except Exception:
                actual_dates = pd.to_datetime(actual_data["date"])

            try:
                pred_dates = pred_df["date"].astype("datetime64[ns]")
            except Exception:
                pred_dates = pd.to_datetime(pred_df["date"])

            actual_subset = pd.DataFrame({"date": actual_dates, target_col: actual_data[target_col].values})
            pred_subset = pd.DataFrame({"date": pred_dates, "prediction": pred_df["prediction"].values})

            # Merge predictions with actual data
            merged = pd.merge(actual_subset, pred_subset, on="date", how="inner")
            residuals = merged[target_col] - merged["prediction"]

            # Logging for debugging empty residuals
            logger.info(
                "Residuals merge for %s: merged_shape=%s, residuals_len=%s, residuals_na=%s",
                model_name,
                merged.shape,
                len(residuals),
                int(residuals.isna().sum()),
            )

            residuals_data[model_name] = residuals
            merged_data[model_name] = merged  # Store the merged dataframe

        # Filter out models with empty or invalid residuals
        valid_residuals: dict[str, pd.Series[Any]] = {
            k: v for k, v in residuals_data.items() if len(v) > 0 and not v.isna().all()
        }

        logger.info("Valid residuals keys: %s", {k: len(v) for k, v in residuals_data.items()})

        if not valid_residuals:
            logger.warning("No valid residuals data found for boxplot")
            # Create empty plot
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            fig.suptitle("Residuals Analysis - No Data Available", fontsize=16)
            for ax in axes.flat:
                ax.text(0.5, 0.5, "No residuals data available", ha="center", va="center", transform=ax.transAxes)
            plt.tight_layout()

            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches="tight")
                plt.close()
            return fig

        # Create matplotlib subplots
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle("Residuals Analysis", fontsize=16)

        # 1. Box plot of residuals
        ax1 = axes[0, 0]
        box_data = [v.dropna().values for v in valid_residuals.values()]  # Convert to numpy arrays and drop NaN
        box_labels: list[str] = list(valid_residuals.keys())
        box_colors: list[str] = [self.colors.get(model.lower(), "#95A5A6") for model in valid_residuals.keys()]

        # Ensure all arrays have the same length by padding with NaN if necessary
        max_len = max(len(arr) for arr in box_data)
        box_data_padded: list[np.ndarray] = []
        for arr in box_data:
            if len(arr) < max_len:
                # Pad with NaN values
                padded = np.full(max_len, np.nan)
                padded[: len(arr)] = arr
                box_data_padded.append(padded)
            else:
                box_data_padded.append(arr)

        bp = ax1.boxplot(box_data_padded, labels=box_labels, patch_artist=True)
        for patch, color in zip(bp["boxes"], box_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax1.set_title("Residuals Distribution")
        ax1.set_ylabel("Residuals")
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=0, color="red", linestyle="--", alpha=0.5)

        # 2. Residuals vs Predicted (for first model)
        ax2 = axes[0, 1]
        first_model = list(valid_residuals.keys())[0]
        first_pred: pd.DataFrame = predictions_dict[first_model]
        first_residuals: pd.Series = valid_residuals[first_model]

        # Ensure we have matching lengths
        min_len = min(len(first_pred), len(first_residuals))
        pred_values: np.ndarray = first_pred["prediction"].values[:min_len]
        resid_values: np.ndarray = first_residuals.values[:min_len]

        ax2.scatter(pred_values, resid_values, color=self.colors.get(first_model.lower(), "#95A5A6"), alpha=0.6, s=30)
        ax2.axhline(y=0, color="red", linestyle="--")
        ax2.set_title(f"Residuals vs Predicted ({first_model})")
        ax2.set_xlabel("Predicted Values")
        ax2.set_ylabel("Residuals")
        ax2.grid(True, alpha=0.3)

        # 3. Residuals over time
        ax3 = axes[1, 0]
        for model_name in valid_residuals.keys():
            if model_name in merged_data:
                # Use the dates from merged data to ensure alignment
                dates: pd.Series = merged_data[model_name]["date"]
                residuals: pd.Series = valid_residuals[model_name]

                ax3.plot(dates, residuals, color=self.colors.get(model_name.lower(), "#95A5A6"), label=model_name, alpha=0.7)
            else:
                # Fallback for backward compatibility
                residuals = valid_residuals[model_name]
                pred_df: pd.DataFrame = predictions_dict[model_name]
                min_len: int = min(len(pred_df), len(residuals))
                dates = pred_df["date"].iloc[:min_len]
                resid_values: np.ndarray = residuals.iloc[:min_len] if hasattr(residuals, "iloc") else residuals[:min_len]

                ax3.plot(
                    dates, resid_values, color=self.colors.get(model_name.lower(), "#95A5A6"), label=model_name, alpha=0.7
                )

        ax3.axhline(y=0, color="red", linestyle="--", alpha=0.5)
        ax3.set_title("Residuals Over Time")
        ax3.set_xlabel("Date")
        ax3.set_ylabel("Residuals")
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        fig.autofmt_xdate()

        # 4. Q-Q plot (for first model)
        ax4 = axes[1, 1]

        # Use the residuals array directly
        resid_array = first_residuals.values if hasattr(first_residuals, "values") else first_residuals
        theoretical_quantiles = stats.probplot(resid_array, dist="norm", fit=False)[0]

        ax4.scatter(
            theoretical_quantiles, sorted(resid_array), color=self.colors.get(first_model.lower(), "#95A5A6"), alpha=0.6
        )

        # Add diagonal reference line
        min_val = min(theoretical_quantiles.min(), resid_array.min())
        max_val = max(theoretical_quantiles.max(), resid_array.max())
        ax4.plot([min_val, max_val], [min_val, max_val], "r--")

        ax4.set_title(f"Q-Q Plot ({first_model})")
        ax4.set_xlabel("Theoretical Quantiles")
        ax4.set_ylabel("Sample Quantiles")
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            plt.close()
            logger.info(f"Saved residuals analysis chart to {save_path}")

        return fig

    def create_feature_importance_chart(
        self,
        feature_importance_dict: dict[str, pd.DataFrame] | dict[str, pl.DataFrame],
        top_n: int = 20,
        save_path: str | None = None,
    ) -> plt.Figure:
        """
        Create feature importance comparison chart

        Parameters
        ----------
        feature_importance_dict : dict[str, pd.DataFrame] | dict[str, pl.DataFrame]
            Dictionary containing feature importance dataframes for each model
        top_n : int, optional
            Number of top features to display, by default 20
        save_path : str | None, optional
            Path to save the figure, by default None

        Returns
        -------
        plt.Figure
            The figure containing the feature importance comparison chart
        """
        if isinstance(feature_importance_dict, dict):
            feature_importance_dict = {k: v.to_pandas() for k, v in feature_importance_dict.items()}

        n_models = len(feature_importance_dict)
        fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 8), sharey=False)

        # Handle single model case
        if n_models == 1:
            axes: list[Any] = [axes]

        for idx, (model_name, importance_df) in enumerate(feature_importance_dict.items()):
            ax = axes[idx]

            # Get top N features
            top_features: pd.DataFrame = importance_df.nlargest(top_n, "importance")

            # Create horizontal bar chart
            y_pos = np.arange(len(top_features))
            ax.barh(
                y_pos,
                top_features["importance"],
                color=self.colors.get(model_name.lower(), "#95A5A6"),
                alpha=0.7,
            )

            # Add value labels
            for i, v in enumerate(top_features["importance"]):
                ax.text(v, i, f" {v:.3f}", va="center")

            ax.set_yticks(y_pos)
            ax.set_yticklabels(top_features["feature"])
            ax.set_xlabel("Importance")
            ax.set_title(f"{model_name} - Top {top_n} Features")
            ax.grid(True, alpha=0.3, axis="x")

            if idx == 0:
                ax.set_ylabel("Features")

        fig.suptitle(f"Top {top_n} Feature Importance by Model", fontsize=16)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            plt.close()
            logger.info(f"Saved feature importance chart to {save_path}")

        return fig

    def create_error_distribution_chart(
        self,
        predictions_dict: dict[str, pd.DataFrame] | dict[str, pl.DataFrame],
        actual_data: pd.DataFrame | pl.DataFrame,
        target_col: str = "sales",
        save_path: str | None = None,
    ) -> plt.Figure:
        """
        Create error distribution visualization.

        Parameters
        ----------
        predictions_dict : dict[str, pd.DataFrame] | dict[str, pl.DataFrame]
            Dictionary of model predictions.
        actual_data : pd.DataFrame | pl.DataFrame
            Actual data.
        target_col : str, optional
            Name of the target column, by default "sales".
        save_path : str | None, optional
            Path to save the figure, by default None.

        Returns
        -------
        plt.Figure
            The figure containing the error distribution chart.
        """
        if isinstance(actual_data, pl.DataFrame):
            actual_data = actual_data.to_pandas()

        if isinstance(predictions_dict, dict):
            predictions_dict = {
                k: (v.to_pandas() if isinstance(v, pl.DataFrame) else v) for k, v in predictions_dict.items()
            }

        fig, ax = plt.subplots(figsize=(10, 6))

        # Iterate over each model and calculate errors
        for model_name, pred_df in predictions_dict.items():
            # Merge predictions with actual data
            merged: pd.DataFrame = pd.merge(
                actual_data[["date", target_col]],
                pred_df[["date", "prediction"]],
                on="date",
                how="inner",
            )

            # Calculate absolute errors
            errors: pd.Series = (merged[target_col] - merged["prediction"]).abs()

            # Create histogram
            ax.hist(
                errors,
                bins=50,
                alpha=0.7,
                color=self.colors.get(model_name.lower(), "#95A5A6"),
                label=model_name,
                density=True,
            )

        ax.set_title("Absolute Error Distribution by Model", fontsize=16)
        ax.set_xlabel("Absolute Error", fontsize=12)
        ax.set_ylabel("Density", fontsize=12)
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            plt.close()
            logger.info(f"Saved error distribution chart to {save_path}")

        return fig

    def create_comprehensive_report(
        self,
        metrics_dict: dict[str, dict[str, float]],
        predictions_dict: dict[str, pd.DataFrame] | dict[str, pl.DataFrame],
        actual_data: pd.DataFrame | pl.DataFrame,
        feature_importance_dict: dict[str, pd.DataFrame] | None = None,
        save_dir: str = "/tmp/model_comparison_charts",
    ) -> dict[str, str]:
        """
        Generate all comparison charts and save them

        Parameters
        ----------
        metrics_dict : dict[str, dict[str, float]]
            A dictionary containing the metrics for each model.
        predictions_dict : dict[str, pd.DataFrame] | dict[str, pl.DataFrame]
            A dictionary containing the predictions for each model.
        actual_data : pd.DataFrame | pl.DataFrame
            The actual data.
        feature_importance_dict : dict[str, pd.DataFrame] | None, optional
            A dictionary containing the feature importance for each model. Defaults to None.
        save_dir : str, optional
            The directory to save the charts. Defaults to "/tmp/model_comparison_charts".

        Returns
        -------
        dict[str, str]
            A dictionary containing the paths to the saved charts.
        """

        if isinstance(actual_data, pl.DataFrame):
            actual_data = actual_data.to_pandas()

        if isinstance(predictions_dict, dict):
            predictions_dict = {
                k: (v.to_pandas() if isinstance(v, pl.DataFrame) else v) for k, v in predictions_dict.items()
            }

        os.makedirs(save_dir, exist_ok=True)

        saved_files: dict[str, str] = {}

        # 1. Metrics comparison
        self.create_metrics_comparison_chart(metrics_dict, save_path=os.path.join(save_dir, "metrics_comparison.png"))
        saved_files["metrics_comparison"] = os.path.join(save_dir, "metrics_comparison.png")

        # 2. Predictions comparison
        self.create_predictions_comparison_chart(
            predictions_dict, actual_data, save_path=os.path.join(save_dir, "predictions_comparison.png")
        )
        saved_files["predictions_comparison"] = os.path.join(save_dir, "predictions_comparison.png")

        # 3. Residuals analysis
        self.create_residuals_analysis(
            predictions_dict, actual_data, save_path=os.path.join(save_dir, "residuals_analysis.png")
        )
        saved_files["residuals_analysis"] = os.path.join(save_dir, "residuals_analysis.png")

        # 4. Error distribution
        self.create_error_distribution_chart(
            predictions_dict, actual_data, save_path=os.path.join(save_dir, "error_distribution.png")
        )
        saved_files["error_distribution"] = os.path.join(save_dir, "error_distribution.png")

        # 5. Feature importance (if available)
        if feature_importance_dict:
            self.create_feature_importance_chart(
                feature_importance_dict, save_path=os.path.join(save_dir, "feature_importance.png")
            )
            saved_files["feature_importance"] = os.path.join(save_dir, "feature_importance.png")

        # Create summary matplotlib figure
        self._create_summary_figure(metrics_dict, save_dir)
        saved_files["summary"] = os.path.join(save_dir, "model_comparison_summary.png")

        logger.info(f"Generated {len(saved_files)} visualization files in {save_dir}")
        return saved_files

    def _create_summary_figure(self, metrics_dict: dict[str, dict[str, float]], save_dir: str) -> None:
        """Create a summary figure using matplotlib

        This function creates a summary figure with a 2x2 grid of subplots.
        Each subplot shows a comparison of the model performances for a
        specific metric (RMSE, MAE, MAPE, R2).

        Parameters
        ----------
        metrics_dict : dict[str, dict[str, float]]
            A dictionary containing the metrics for each model.
        save_dir : str
            The directory to save the figure.

        Returns
        -------
        None
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle("Model Performance Summary", fontsize=16)

        models: list[str] = list(metrics_dict.keys())
        metrics: list[str] = ["rmse", "mae", "mape", "r2"]

        for _, (ax, metric) in enumerate(zip(axes.flat, metrics)):
            values: list[float] = [metrics_dict[model].get(metric, 0) for model in models]
            colors: list[str] = [self.colors.get(model.lower(), "#95A5A6") for model in models]

            bars = ax.bar(models, values, color=colors, alpha=0.7)

            # Add value labels on bars
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height,
                    f"{value:.3f}",
                    ha="center",
                    va="bottom",
                )

            ax.set_title(f"{metric.upper()} Comparison")
            ax.set_ylabel(metric.upper())
            ax.grid(True, alpha=0.3)

            # Highlight best model
            if metric == "r2":  # Higher is better
                best_idx: int = values.index(max(values))
            else:  # Lower is better
                best_idx = values.index(min(values))
            bars[best_idx].set_edgecolor("green")
            bars[best_idx].set_linewidth(3)

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "model_comparison_summary.png"), dpi=300, bbox_inches="tight")
        plt.close()


def generate_model_comparison_report(run_id: str, test_data: pd.DataFrame | pl.DataFrame) -> dict[str, str]:
    """
    Generate comparison report from MLflow run

    Parameters
    ----------
    run_id : str
        MLflow run ID
    test_data : pd.DataFrame | pl.DataFrame
        Test data with ground truth

    Returns
    -------
    dict[str, str]
        Dictionary of saved file paths
    """
    if isinstance(test_data, pl.DataFrame):
        test_data = test_data.to_pandas()

    visualizer = ModelVisualizer()

    client = mlflow.tracking.MlflowClient()  # type: ignore
    run = client.get_run(run_id)

    # Extract metrics
    metrics_dict: dict[str, dict[str, float]] = {}
    for model in ["xgboost", "lightgbm", "ensemble"]:
        model_metrics: dict[str, float] = {}
        for metric in ["rmse", "mae", "mape", "r2"]:
            metric_key = f"{model}_{metric}"
            if metric_key in run.data.metrics:
                model_metrics[metric] = run.data.metrics[metric_key]
        if model_metrics:
            metrics_dict[model] = model_metrics

    # Generate dummy predictions for visualization
    # In real scenario, load actual predictions from artifacts
    predictions_dict: dict[str, pd.DataFrame] = {}
    rng = np.random.default_rng()
    for model in metrics_dict.keys():
        pred_df: pd.DataFrame = test_data[["date"]].copy()
        # Add some noise to create different predictions
        noise: np.ndarray = rng.normal(0, 5, len(test_data))
        pred_df["prediction"] = test_data["sales"] + noise
        predictions_dict[model] = pred_df

    # Generate visualizations
    saved_files: dict[str, str] = visualizer.create_comprehensive_report(metrics_dict, predictions_dict, test_data)

    # Log visualizations to MLflow
    for name, path in saved_files.items():
        mlflow.log_artifact(path, f"{PACKAGE_PATH}/artifacts/visualizations/{name}")  # type: ignore

    return saved_files
