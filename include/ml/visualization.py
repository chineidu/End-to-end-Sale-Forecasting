"""This module provides visualization utilities for comparing machine learning model performance.
Copied from: https://github.com/airscholar/astro-salesforecast/blob/main/include/ml_models/model_visualization.py
"""

import os

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import polars as pl
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

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

    def create_performance_dashboard(
        self, metrics_dict: dict[str, dict[str, float]], save_path: str | None = None
    ) -> plt.Figure:
        """Create a comprehensive performance dashboard combining metrics and rankings.

        Parameters
        ----------
        metrics_dict : dict[str, dict[str, float]]
            Dictionary with model names as keys and metrics as values.
        save_path : str | None
            Path to save the figure.

        Returns
        -------
        plt.Figure
            The figure containing the performance dashboard.
        """
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

        models = list(metrics_dict.keys())

        # Individual metric plots (2x2 grid in upper section)
        metrics_info = [
            ("rmse", "RMSE", True, 0, 0),
            ("mae", "MAE", True, 0, 1),
            ("mape", "MAPE (%)", True, 1, 0),
            ("r2", "R² Score", False, 1, 1),
        ]

        for metric, title, lower_better, row, col in metrics_info:
            ax = fig.add_subplot(gs[row, col])
            values = [metrics_dict[model].get(metric, 0) for model in models]
            colors = [self.colors.get(model.lower(), "#95A5A6") for model in models]

            bars = ax.bar(models, values, color=colors, alpha=0.7)

            # Add value labels
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2.0, height, f"{value:.3f}", ha="center", va="bottom", fontsize=10)

            # Highlight best model
            best_idx = values.index(min(values) if lower_better else max(values))
            bars[best_idx].set_edgecolor("green")
            bars[best_idx].set_linewidth(3)

            ax.set_title(f"{title}")
            ax.grid(True, alpha=0.3)
            ax.tick_params(axis="x", rotation=45)

        # Model ranking summary (right side)
        ax_ranking = fig.add_subplot(gs[0:2, 2])
        self._create_model_ranking(metrics_dict, ax_ranking)

        # Overall performance radar/spider chart (bottom)
        ax_radar = fig.add_subplot(gs[2, :])
        self._create_normalized_performance_comparison(metrics_dict, ax_radar)

        fig.suptitle("Model Performance Dashboard", fontsize=18, y=0.95)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            plt.close()
            logger.info(f"Saved performance dashboard to {save_path}")

        return fig

    def create_prediction_quality_analysis(
        self,
        predictions_dict: dict[str, pd.DataFrame] | dict[str, pl.DataFrame],
        actual_data: pd.DataFrame | pl.DataFrame,
        date_col: str = "date",
        target_col: str = "sales",
        save_path: str | None = None,
    ) -> plt.Figure:
        """Create prediction quality analysis combining scatter plots and time series.

        Parameters
        ----------
        predictions_dict : dict[str, pd.DataFrame] | dict[str, pl.DataFrame]
            Dictionary with model predictions.
        actual_data : pd.DataFrame | pl.DataFrame
            Actual data.
        date_col : str, optional
            Date column name, by default "date".
        target_col : str, optional
            Target column name, by default "sales".
        save_path : str | None, optional
            Path to save figure, by default None.

        Returns
        -------
        plt.Figure
            The figure containing prediction quality analysis.
        """
        # Convert to pandas if needed
        if isinstance(actual_data, pl.DataFrame):
            actual_data = actual_data.to_pandas()

        if isinstance(predictions_dict, dict):
            predictions_dict = {
                k: (v.to_pandas() if isinstance(v, pl.DataFrame) else v) for k, v in predictions_dict.items()
            }

        n_models = len(predictions_dict)
        fig, axes = plt.subplots(2, n_models, figsize=(5 * n_models, 10))

        if n_models == 1:
            axes = axes.reshape(-1, 1)

        for idx, (model_name, pred_df) in enumerate(predictions_dict.items()):
            # Merge data
            merged = pd.merge(
                actual_data[[date_col, target_col]], pred_df[[date_col, "prediction"]], on=date_col, how="inner"
            )

            color = self.colors.get(model_name.lower(), "#95A5A6")

            # Top row: Actual vs Predicted scatter
            ax_scatter = axes[0, idx]
            ax_scatter.scatter(merged[target_col], merged["prediction"], color=color, alpha=0.6, s=30)

            # Perfect prediction line
            min_val = min(merged[target_col].min(), merged["prediction"].min())
            max_val = max(merged[target_col].max(), merged["prediction"].max())
            ax_scatter.plot([min_val, max_val], [min_val, max_val], "r--", alpha=0.8)

            # Calculate R²
            r2 = r2_score(merged[target_col], merged["prediction"])
            ax_scatter.text(
                0.05,
                0.95,
                f"R² = {r2:.3f}",
                transform=ax_scatter.transAxes,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
            )

            ax_scatter.set_xlabel(f"Actual {target_col.title()}")
            ax_scatter.set_ylabel("Predicted")
            ax_scatter.set_title(f"{model_name} - Actual vs Predicted")
            ax_scatter.grid(True, alpha=0.3)

            # Bottom row: Time series comparison
            ax_time = axes[1, idx]

            # Sample data if too many points (for readability)
            if len(merged) > 500:
                sample_idx = np.linspace(0, len(merged) - 1, 500, dtype=int)
                plot_data = merged.iloc[sample_idx]
            else:
                plot_data = merged

            ax_time.plot(
                plot_data[date_col],
                plot_data[target_col],
                color=self.colors["actual"],
                linewidth=2,
                label="Actual",
                alpha=0.8,
            )
            ax_time.plot(
                plot_data[date_col],
                plot_data["prediction"],
                color=color,
                linewidth=2,
                label=f"{model_name} Prediction",
                alpha=0.7,
            )

            # Add confidence intervals if available
            if "prediction_lower" in pred_df.columns and "prediction_upper" in pred_df.columns:
                ax_time.fill_between(
                    plot_data[date_col],
                    plot_data["prediction_lower"] if "prediction_lower" in plot_data.columns else plot_data["prediction"],
                    plot_data["prediction_upper"] if "prediction_upper" in plot_data.columns else plot_data["prediction"],
                    color=color,
                    alpha=0.2,
                )

            ax_time.set_xlabel("Date")
            ax_time.set_ylabel(target_col.title())
            ax_time.set_title(f"{model_name} - Time Series")
            ax_time.legend()
            ax_time.grid(True, alpha=0.3)
            fig.autofmt_xdate()

        fig.suptitle("Prediction Quality Analysis", fontsize=16)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            plt.close()
            logger.info(f"Saved prediction quality analysis to {save_path}")

        return fig

    def create_residuals_diagnostic_panel(
        self,
        predictions_dict: dict[str, pd.DataFrame] | dict[str, pl.DataFrame],
        actual_data: pd.DataFrame | pl.DataFrame,
        target_col: str = "sales",
        save_path: str | None = None,
    ) -> plt.Figure:
        """Create comprehensive residuals diagnostic panel.

        Parameters
        ----------
        predictions_dict : dict[str, pd.DataFrame] | dict[str, pl.DataFrame]
            Dictionary of model predictions.
        actual_data : pd.DataFrame | pl.DataFrame
            Actual data.
        target_col : str, optional
            Target column name, by default "sales".
        save_path : str | None, optional
            Path to save figure, by default None.

        Returns
        -------
        plt.Figure
            The figure containing residuals diagnostics.
        """
        if isinstance(actual_data, pl.DataFrame):
            actual_data = actual_data.to_pandas()

        if isinstance(predictions_dict, dict):
            predictions_dict = {
                k: (v.to_pandas() if isinstance(v, pl.DataFrame) else v) for k, v in predictions_dict.items()
            }

        # Calculate residuals
        residuals_data = {}
        for model_name, pred_df in predictions_dict.items():
            merged = pd.merge(actual_data[["date", target_col]], pred_df[["date", "prediction"]], on="date", how="inner")
            residuals = merged[target_col] - merged["prediction"]
            residuals_data[model_name] = {
                "residuals": residuals,
                "predictions": merged["prediction"],
                "actual": merged[target_col],
                "dates": merged["date"],
            }

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle("Residuals Diagnostic Panel", fontsize=16)

        # 1. Residuals distribution comparison
        ax1 = axes[0, 0]
        for model_name, data in residuals_data.items():
            residuals = data["residuals"].dropna()
            if len(residuals) > 0:
                ax1.hist(
                    residuals,
                    bins=30,
                    alpha=0.6,
                    density=True,
                    color=self.colors.get(model_name.lower(), "#95A5A6"),
                    label=model_name,
                )
        ax1.axvline(x=0, color="red", linestyle="--", alpha=0.7)
        ax1.set_title("Residuals Distribution")
        ax1.set_xlabel("Residuals")
        ax1.set_ylabel("Density")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. Residuals vs Fitted
        ax2 = axes[0, 1]
        for model_name, data in residuals_data.items():
            ax2.scatter(
                data["predictions"],
                data["residuals"],
                color=self.colors.get(model_name.lower(), "#95A5A6"),
                alpha=0.6,
                s=20,
                label=model_name,
            )
        ax2.axhline(y=0, color="red", linestyle="--", alpha=0.7)
        ax2.set_title("Residuals vs Fitted Values")
        ax2.set_xlabel("Fitted Values")
        ax2.set_ylabel("Residuals")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 3. Scale-Location plot (sqrt of standardized residuals vs fitted)
        ax3 = axes[0, 2]
        for model_name, data in residuals_data.items():
            residuals = data["residuals"]
            predictions = data["predictions"]
            std_residuals = np.sqrt(np.abs(residuals / residuals.std()))
            ax3.scatter(
                predictions,
                std_residuals,
                color=self.colors.get(model_name.lower(), "#95A5A6"),
                alpha=0.6,
                s=20,
                label=model_name,
            )
        ax3.set_title("Scale-Location Plot")
        ax3.set_xlabel("Fitted Values")
        ax3.set_ylabel("√|Standardized Residuals|")
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # 4. Residuals over time
        ax4 = axes[1, 0]
        for model_name, data in residuals_data.items():
            ax4.plot(
                data["dates"],
                data["residuals"],
                color=self.colors.get(model_name.lower(), "#95A5A6"),
                alpha=0.7,
                label=model_name,
                linewidth=1,
            )
        ax4.axhline(y=0, color="red", linestyle="--", alpha=0.7)
        ax4.set_title("Residuals Over Time")
        ax4.set_xlabel("Date")
        ax4.set_ylabel("Residuals")
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        fig.autofmt_xdate()

        # 5. Q-Q Plot (Normal probability plot)
        ax5 = axes[1, 1]
        for model_name, data in residuals_data.items():
            residuals = data["residuals"].dropna().values
            if len(residuals) > 0:
                stats.probplot(residuals, dist="norm", plot=ax5)
                ax5.get_lines()[-1].set_color(self.colors.get(model_name.lower(), "#95A5A6"))
                ax5.get_lines()[-1].set_label(model_name)
        ax5.set_title("Normal Q-Q Plot")
        ax5.grid(True, alpha=0.3)
        ax5.legend()

        # 6. Residuals boxplot comparison
        ax6 = axes[1, 2]
        box_data = []
        box_labels = []
        box_colors = []
        for model_name, data in residuals_data.items():
            residuals = data["residuals"].dropna().values
            if len(residuals) > 0:
                box_data.append(residuals)
                box_labels.append(model_name)
                box_colors.append(self.colors.get(model_name.lower(), "#95A5A6"))

        if box_data:
            bp = ax6.boxplot(box_data, labels=box_labels, patch_artist=True)
            for patch, color in zip(bp["boxes"], box_colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
        ax6.axhline(y=0, color="red", linestyle="--", alpha=0.7)
        ax6.set_title("Residuals Distribution")
        ax6.set_ylabel("Residuals")
        ax6.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            plt.close()
            logger.info(f"Saved residuals diagnostic panel to {save_path}")

        return fig

    def create_feature_importance_comparison(
        self,
        feature_importance_dict: dict[str, pd.DataFrame] | dict[str, pl.DataFrame],
        top_n: int = 15,
        save_path: str | None = None,
    ) -> plt.Figure:
        """Create enhanced feature importance comparison with consistency analysis.

        Parameters
        ----------
        feature_importance_dict : dict[str, pd.DataFrame] | dict[str, pl.DataFrame]
            Dictionary containing feature importance dataframes.
        top_n : int, optional
            Number of top features to display, by default 15.
        save_path : str | None, optional
            Path to save figure, by default None.

        Returns
        -------
        plt.Figure
            The figure containing feature importance comparison.
        """
        if isinstance(feature_importance_dict, dict):
            feature_importance_dict = {
                k: v.to_pandas() if isinstance(v, pl.DataFrame) else v for k, v in feature_importance_dict.items()
            }

        n_models = len(feature_importance_dict)
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(2, n_models + 1, height_ratios=[2, 1], width_ratios=[1] * n_models + [1.2])

        # Individual model importance plots (top row)
        for idx, (model_name, importance_df) in enumerate(feature_importance_dict.items()):
            ax = fig.add_subplot(gs[0, idx])

            top_features = importance_df.nlargest(top_n, "importance")
            y_pos = np.arange(len(top_features))

            bars = ax.barh(
                y_pos, top_features["importance"], color=self.colors.get(model_name.lower(), "#95A5A6"), alpha=0.7
            )

            # Add value labels
            for i, v in enumerate(top_features["importance"]):
                ax.text(v + max(top_features["importance"]) * 0.01, i, f"{v:.3f}", va="center", fontsize=9)

            ax.set_yticks(y_pos)
            ax.set_yticklabels(top_features["feature"], fontsize=9)
            ax.set_xlabel("Importance")
            ax.set_title(f"{model_name}\nTop {top_n} Features")
            ax.grid(True, alpha=0.3, axis="x")

        # Feature consistency analysis (top right)
        ax_consistency = fig.add_subplot(gs[0, n_models])
        self._create_feature_consistency_plot(feature_importance_dict, ax_consistency, top_n)

        # Feature importance heatmap (bottom)
        ax_heatmap = fig.add_subplot(gs[1, :])
        self._create_importance_heatmap(feature_importance_dict, ax_heatmap, top_n)

        fig.suptitle(f"Feature Importance Analysis - Top {top_n} Features", fontsize=16)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            plt.close()
            logger.info(f"Saved feature importance comparison to {save_path}")

        return fig

    def create_model_stability_analysis(
        self,
        predictions_dict: dict[str, pd.DataFrame] | dict[str, pl.DataFrame],
        actual_data: pd.DataFrame | pl.DataFrame,
        target_col: str = "sales",
        window_size: int = 30,
        save_path: str | None = None,
    ) -> plt.Figure:
        """Create model stability analysis showing performance over time windows.

        Parameters
        ----------
        predictions_dict : dict[str, pd.DataFrame] | dict[str, pl.DataFrame]
            Dictionary of model predictions.
        actual_data : pd.DataFrame | pl.DataFrame
            Actual data.
        target_col : str, optional
            Target column name, by default "sales".
        window_size : int, optional
            Rolling window size for stability analysis, by default 30.
        save_path : str | None, optional
            Path to save figure, by default None.

        Returns
        -------
        plt.Figure
            The figure containing stability analysis.
        """
        if isinstance(actual_data, pl.DataFrame):
            actual_data = actual_data.to_pandas()

        if isinstance(predictions_dict, dict):
            predictions_dict = {
                k: (v.to_pandas() if isinstance(v, pl.DataFrame) else v) for k, v in predictions_dict.items()
            }

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f"Model Stability Analysis (Rolling Window: {window_size})", fontsize=16)

        stability_data = {}

        # Calculate rolling metrics for each model
        for model_name, pred_df in predictions_dict.items():
            merged = pd.merge(actual_data[["date", target_col]], pred_df[["date", "prediction"]], on="date", how="inner")
            merged = merged.sort_values("date").reset_index(drop=True)

            # Calculate rolling metrics
            rolling_mae = []
            rolling_rmse = []
            rolling_r2 = []
            dates = []

            for i in range(window_size, len(merged)):
                window_actual = merged[target_col].iloc[i - window_size : i]
                window_pred = merged["prediction"].iloc[i - window_size : i]

                mae = mean_absolute_error(window_actual, window_pred)
                rmse = np.sqrt(mean_squared_error(window_actual, window_pred))
                r2 = r2_score(window_actual, window_pred)

                rolling_mae.append(mae)
                rolling_rmse.append(rmse)
                rolling_r2.append(r2)
                dates.append(merged["date"].iloc[i])

            stability_data[model_name] = {"dates": dates, "mae": rolling_mae, "rmse": rolling_rmse, "r2": rolling_r2}

        # Plot rolling metrics
        metrics = ["mae", "rmse", "r2"]
        titles = ["Rolling MAE", "Rolling RMSE", "Rolling R²"]

        for idx, (metric, title) in enumerate(zip(metrics[:3], titles)):
            ax = axes[idx // 2, idx % 2]

            for model_name, data in stability_data.items():
                ax.plot(
                    data["dates"],
                    data[metric],
                    color=self.colors.get(model_name.lower(), "#95A5A6"),
                    label=model_name,
                    linewidth=2,
                    alpha=0.8,
                )

            ax.set_title(title)
            ax.set_xlabel("Date")
            ax.set_ylabel(metric.upper())
            ax.legend()
            ax.grid(True, alpha=0.3)
            fig.autofmt_xdate()

        # Model stability ranking (bottom right)
        ax_stability = axes[1, 1]
        self._create_stability_ranking(stability_data, ax_stability)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            plt.close()
            logger.info(f"Saved stability analysis to {save_path}")

        return fig

    def _create_model_ranking(self, metrics_dict: dict[str, dict[str, float]], ax: plt.Axes) -> None:
        """Create model ranking visualization."""
        models = list(metrics_dict.keys())

        # Calculate ranks for each metric (1 = best)
        ranking_scores = {}
        for model in models:
            score = 0
            for metric in ["rmse", "mae", "mape"]:  # Lower is better
                values = [metrics_dict[m].get(metric, float("inf")) for m in models]
                rank = sorted(values).index(metrics_dict[model].get(metric, float("inf"))) + 1
                score += rank

            # R² - higher is better
            r2_values = [metrics_dict[m].get("r2", -float("inf")) for m in models]
            r2_rank = len(models) - sorted(r2_values).index(metrics_dict[model].get("r2", -float("inf")))
            score += r2_rank

            ranking_scores[model] = score / 4  # Average rank

        # Sort by ranking score
        sorted_models = sorted(ranking_scores.items(), key=lambda x: x[1])

        models_sorted = [x[0] for x in sorted_models]
        scores_sorted = [x[1] for x in sorted_models]
        colors = [self.colors.get(model.lower(), "#95A5A6") for model in models_sorted]

        bars = ax.barh(range(len(models_sorted)), scores_sorted, color=colors, alpha=0.7)

        # Add score labels
        for i, (bar, score) in enumerate(zip(bars, scores_sorted)):
            ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height() / 2, f"{score:.2f}", va="center", fontsize=10)

        ax.set_yticks(range(len(models_sorted)))
        ax.set_yticklabels(models_sorted)
        ax.set_xlabel("Average Rank")
        ax.set_title("Overall Model Ranking\n(Lower is Better)")
        ax.grid(True, alpha=0.3, axis="x")
        ax.invert_yaxis()

    def _create_normalized_performance_comparison(self, metrics_dict: dict[str, dict[str, float]], ax: plt.Axes) -> None:
        """Create normalized performance comparison."""
        models = list(metrics_dict.keys())
        metrics = ["rmse", "mae", "mape", "r2"]

        # Normalize metrics (0-1 scale)
        normalized_data = {}
        for metric in metrics:
            values = [metrics_dict[model].get(metric, 0) for model in models]
            if metric == "r2":  # Higher is better
                min_val, max_val = min(values), max(values)
                normalized_values = [(v - min_val) / (max_val - min_val) if max_val != min_val else 0.5 for v in values]
            else:  # Lower is better - invert
                min_val, max_val = min(values), max(values)
                normalized_values = [1 - (v - min_val) / (max_val - min_val) if max_val != min_val else 0.5 for v in values]

            normalized_data[metric] = normalized_values

        # Create grouped bar chart
        x = np.arange(len(models))
        width = 0.2

        for i, metric in enumerate(metrics):
            offset = (i - 1.5) * width
            bars = ax.bar(x + offset, normalized_data[metric], width, label=metric.upper(), alpha=0.8)

        ax.set_xlabel("Models")
        ax.set_ylabel("Normalized Performance (0-1, Higher is Better)")
        ax.set_title("Normalized Performance Comparison")
        ax.set_xticks(x)
        ax.set_xticklabels(models)
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")
        ax.set_ylim(0, 1.1)

    def _create_feature_consistency_plot(
        self, feature_importance_dict: dict[str, pd.DataFrame], ax: plt.Axes, top_n: int
    ) -> None:
        """Create feature consistency analysis."""
        # Get all unique features from all models
        all_features = set()
        for df in feature_importance_dict.values():
            all_features.update(df["feature"].tolist())

        # Calculate how many models each feature appears in top_n
        feature_counts = {}
        for feature in all_features:
            count = 0
            for model_name, df in feature_importance_dict.items():
                top_features = df.nlargest(top_n, "importance")["feature"].tolist()
                if feature in top_features:
                    count += 1
            if count > 1:  # Only show features that appear in multiple models
                feature_counts[feature] = count

        if feature_counts:
            features = list(feature_counts.keys())
            counts = list(feature_counts.values())

            bars = ax.barh(range(len(features)), counts, color="#4A90E2", alpha=0.7)

            ax.set_yticks(range(len(features)))
            ax.set_yticklabels(features, fontsize=9)
            ax.set_xlabel("Number of Models")
            ax.set_title("Feature Consistency\n(Across Models)")
            ax.grid(True, alpha=0.3, axis="x")
            ax.set_xlim(0, len(feature_importance_dict))
        else:
            ax.text(0.5, 0.5, "No consistent features\nacross models", ha="center", va="center", transform=ax.transAxes)
            ax.set_title("Feature Consistency")

    def _create_importance_heatmap(self, feature_importance_dict: dict[str, pd.DataFrame], ax: plt.Axes, top_n: int) -> None:
        """Create feature importance heatmap."""
        # Get top features across all models
        all_features = set()
        for df in feature_importance_dict.values():
            top_features = df.nlargest(top_n, "importance")["feature"].tolist()
            all_features.update(top_features)

        # Create importance matrix
        models = list(feature_importance_dict.keys())
        features = list(all_features)
        importance_matrix = np.zeros((len(features), len(models)))

        for j, model in enumerate(models):
            df = feature_importance_dict[model]
            for i, feature in enumerate(features):
                importance_row = df[df["feature"] == feature]
                if not importance_row.empty:
                    importance_matrix[i, j] = importance_row["importance"].iloc[0]

        # Normalize by row (feature) for better visualization
        importance_matrix_norm = importance_matrix / (importance_matrix.max(axis=1, keepdims=True) + 1e-8)

        # Create heatmap
        im = ax.imshow(importance_matrix_norm, cmap="RdYlBu_r", aspect="auto")

        # Set labels
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(models, rotation=45)
        ax.set_yticks(range(len(features)))
        ax.set_yticklabels(features, fontsize=8)

        # Add colorbar
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Normalized Importance")

        ax.set_title("Feature Importance Heatmap")

    def _create_stability_ranking(self, stability_data: dict[str, dict], ax: plt.Axes) -> None:
        """Create model stability ranking based on variance of rolling metrics."""
        stability_scores = {}

        for model_name, data in stability_data.items():
            # Calculate coefficient of variation (std/mean) for each metric
            mae_cv = np.std(data["mae"]) / (np.mean(data["mae"]) + 1e-8)
            rmse_cv = np.std(data["rmse"]) / (np.mean(data["rmse"]) + 1e-8)
            r2_cv = np.std(data["r2"]) / (np.mean(np.abs(data["r2"])) + 1e-8)

            # Average CV (lower is more stable)
            stability_scores[model_name] = (mae_cv + rmse_cv + r2_cv) / 3

        # Sort by stability (lower CV = more stable)
        sorted_models = sorted(stability_scores.items(), key=lambda x: x[1])

        models_sorted = [x[0] for x in sorted_models]
        scores_sorted = [x[1] for x in sorted_models]
        colors = [self.colors.get(model.lower(), "#95A5A6") for model in models_sorted]

        bars = ax.barh(range(len(models_sorted)), scores_sorted, color=colors, alpha=0.7)

        # Add score labels
        for i, (bar, score) in enumerate(zip(bars, scores_sorted)):
            ax.text(
                bar.get_width() + max(scores_sorted) * 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{score:.3f}",
                va="center",
                fontsize=9,
            )

        ax.set_yticks(range(len(models_sorted)))
        ax.set_yticklabels(models_sorted)
        ax.set_xlabel("Coefficient of Variation")
        ax.set_title("Model Stability Ranking\n(Lower = More Stable)")
        ax.grid(True, alpha=0.3, axis="x")
        ax.invert_yaxis()

    def create_comprehensive_report(
        self,
        metrics_dict: dict[str, dict[str, float]],
        predictions_dict: dict[str, pd.DataFrame] | dict[str, pl.DataFrame],
        actual_data: pd.DataFrame | pl.DataFrame,
        feature_importance_dict: dict[str, pd.DataFrame] | None = None,
        save_dir: str = "/tmp/model_comparison_charts",
        window_size: int = 30,
    ) -> dict[str, str]:
        """Generate all comparison charts and save them.

        Parameters
        ----------
        metrics_dict : dict[str, dict[str, float]]
            Dictionary containing metrics for each model.
        predictions_dict : dict[str, pd.DataFrame] | dict[str, pl.DataFrame]
            Dictionary containing predictions for each model.
        actual_data : pd.DataFrame | pl.DataFrame
            The actual data.
        feature_importance_dict : dict[str, pd.DataFrame] | None, optional
            Dictionary containing feature importance for each model, by default None.
        save_dir : str, optional
            Directory to save charts, by default "/tmp/model_comparison_charts".
        window_size : int, optional
            Window size for stability analysis, by default 30.

        Returns
        -------
        dict[str, str]
            Dictionary containing paths to saved charts.
        """
        if isinstance(actual_data, pl.DataFrame):
            actual_data = actual_data.to_pandas()

        if isinstance(predictions_dict, dict):
            predictions_dict = {
                k: (v.to_pandas() if isinstance(v, pl.DataFrame) else v) for k, v in predictions_dict.items()
            }

        os.makedirs(save_dir, exist_ok=True)
        saved_files = {}

        # 1. Performance Dashboard (replaces simple metrics comparison)
        self.create_performance_dashboard(metrics_dict, save_path=os.path.join(save_dir, "performance_dashboard.png"))
        saved_files["performance_dashboard"] = os.path.join(save_dir, "performance_dashboard.png")

        # 2. Prediction Quality Analysis (combines actual vs predicted + time series)
        self.create_prediction_quality_analysis(
            predictions_dict, actual_data, save_path=os.path.join(save_dir, "prediction_quality_analysis.png")
        )
        saved_files["prediction_quality_analysis"] = os.path.join(save_dir, "prediction_quality_analysis.png")

        # 3. Enhanced Residuals Diagnostic Panel
        self.create_residuals_diagnostic_panel(
            predictions_dict, actual_data, save_path=os.path.join(save_dir, "residuals_diagnostic_panel.png")
        )
        saved_files["residuals_diagnostic_panel"] = os.path.join(save_dir, "residuals_diagnostic_panel.png")

        # 4. Model Stability Analysis (new)
        if len(actual_data) > window_size * 2:  # Only if enough data
            self.create_model_stability_analysis(
                predictions_dict,
                actual_data,
                window_size=window_size,
                save_path=os.path.join(save_dir, "model_stability_analysis.png"),
            )
            saved_files["model_stability_analysis"] = os.path.join(save_dir, "model_stability_analysis.png")

        # 5. Enhanced Feature Importance (if available)
        if feature_importance_dict:
            self.create_feature_importance_comparison(
                feature_importance_dict, save_path=os.path.join(save_dir, "feature_importance_comparison.png")
            )
            saved_files["feature_importance_comparison"] = os.path.join(save_dir, "feature_importance_comparison.png")

        logger.info(f"Generated {len(saved_files)} visualization files in {save_dir}")
        return saved_files


def generate_model_comparison_report(run_id: str, test_data: pd.DataFrame | pl.DataFrame) -> dict[str, str]:
    """Generate comparison report from MLflow run.

    Parameters
    ----------
    run_id : str
        MLflow run ID.
    test_data : pd.DataFrame | pl.DataFrame
        Test data with ground truth.

    Returns
    -------
    dict[str, str]
        Dictionary of saved file paths.
    """
    if isinstance(test_data, pl.DataFrame):
        test_data = test_data.to_pandas()

    visualizer = ModelVisualizer()

    client = mlflow.tracking.MlflowClient()  # type: ignore
    run = client.get_run(run_id)

    # Extract metrics
    metrics_dict = {}
    for model in ["xgboost", "lightgbm", "ensemble"]:
        model_metrics = {}
        for metric in ["rmse", "mae", "mape", "r2"]:
            metric_key = f"{model}_{metric}"
            if metric_key in run.data.metrics:
                model_metrics[metric] = run.data.metrics[metric_key]
        if model_metrics:
            metrics_dict[model] = model_metrics

    # Generate dummy predictions for visualization
    # In real scenario, load actual predictions from artifacts
    predictions_dict = {}
    rng = np.random.default_rng()
    for model in metrics_dict.keys():
        pred_df = test_data[["date"]].copy()
        # Add some noise to create different predictions
        noise = rng.normal(0, 5, len(test_data))
        pred_df["prediction"] = test_data["sales"] + noise
        predictions_dict[model] = pred_df

    # Generate visualizations
    saved_files = visualizer.create_comprehensive_report(metrics_dict, predictions_dict, test_data)

    # Log visualizations to MLflow
    for name, path in saved_files.items():
        mlflow.log_artifact(path, f"{PACKAGE_PATH}/artifacts/visualizations/{name}")  # type: ignore

    return saved_files
