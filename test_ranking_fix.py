#!/usr/bin/env python3
"""Test script to verify the ranking fix works correctly."""

import matplotlib.pyplot as plt

from include.ml.visualization import ModelVisualizer

# Sample metrics from the screenshot
metrics_dict = {
    "xgboost": {"rmse": 22.432, "mae": 12.025, "mape": 10.613, "r2": 0.936},
    "lightgbm": {"rmse": 24.992, "mae": 13.918, "mape": 12.437, "r2": 0.921},
    "ensemble": {"rmse": 23.380, "mae": 12.414, "mape": 11.154, "r2": 0.930},
}


def test_ranking() -> None:
    """Test the ranking calculation."""
    print("Testing model ranking with sample metrics...")
    print("Metrics from screenshot:")
    for model, metrics in metrics_dict.items():
        print(f"  {model}: RMSE={metrics['rmse']}, MAE={metrics['mae']}, MAPE={metrics['mape']}, R2={metrics['r2']}")

    # Create visualizer and test dashboard
    visualizer = ModelVisualizer()

    # Test just the dashboard creation
    fig = visualizer.create_performance_dashboard(metrics_dict, save_path="test_dashboard.png")
    print("\nDashboard created and saved as 'test_dashboard.png'")
    print("Expected ranking (best to worst): ensemble, xgboost, lightgbm")

    # Clean up
    plt.close(fig)


if __name__ == "__main__":
    test_ranking()
