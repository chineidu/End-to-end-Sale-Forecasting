import os
import shutil
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from airflow.decorators import dag, task
from airflow.providers.standard.operators.bash import BashOperator

# Make local project imports (include/...) resolvable when parsing the DAG
# DAG file path: <project_root>/airflow/dags/ml_dag.py
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# NOTE: Avoid importing project modules at parse time; import them inside tasks instead

default_args = {
    "owner": "data-team",
    "depends_on_past": False,
    "start_date": datetime(2025, 7, 22),
    # Disable failure emails to avoid template errors in Airflow 3 and reduce noise
    "email_on_failure": False,
    "email_on_retry": False,
    "email": ["admin@example.com"],
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}


@dag(
    schedule="@weekly",
    start_date=datetime(2025, 7, 22),
    catchup=False,
    default_args=default_args,
    description="Train sales forecasting models",
    tags=["ml", "training", "sales"],
)
def sales_forecast_training() -> None:
    @task()
    def extract_data_task() -> dict[str, Any]:
        from include.utilities.data_gen import RealisticSalesDataGenerator

        data_output_dir = "data/"
        # Clean previous run outputs to avoid mixing stale/corrupt files
        try:
            if os.path.exists(data_output_dir):
                shutil.rmtree(data_output_dir)
        except Exception as e:
            print(f"Warning: could not clean previous data directory {data_output_dir}: {e}")
        generator = RealisticSalesDataGenerator(start_date="2021-01-01", end_date="2021-02-28")
        print("Generating realistic sales data...")

        file_paths: dict[str, list[str]] = generator.generate_sales_data(output_dir=data_output_dir)
        total_files: int = sum(len(paths) for paths in file_paths.values())
        print(f"Generated {total_files} files:")

        for data_type, paths in file_paths.items():
            print(f"  - {data_type}: {len(paths)} files")
        return {
            "data_output_dir": data_output_dir,
            "file_paths": file_paths,
            "total_files": total_files,
        }

    @task()
    def validate_data_task(extract_result: dict[str, Any]) -> dict[str, Any]:
        file_paths: dict[str, list[str]] = extract_result["file_paths"]
        total_rows: int = 0
        issues_found: list[str] = []
        N: int = 10  # Number of files to sample for validation

        print(f"Validating {len(file_paths['sales'])} sales files...")
        for i, sales_file in enumerate(file_paths["sales"][:N]):
            try:
                if not os.path.isfile(sales_file):
                    issues_found.append(f"Not a file: {sales_file}")
                    continue
                if os.path.getsize(sales_file) == 0:
                    issues_found.append(f"Zero-byte file: {sales_file}")
                    continue
                df: pd.DataFrame = pd.read_parquet(sales_file, engine="pyarrow")
            except Exception as e:
                issues_found.append(f"Failed to read {sales_file}: {e}")
                continue

            if i == 0:
                # Print the columns from the first file
                print(f"Sales data columns: {df.columns.tolist()}")
            if df.empty:
                issues_found.append(f"Empty file: {sales_file}")
                continue

            required_cols: list[str] = [
                "date",
                "store_id",
                "product_id",
                "quantity_sold",
                "revenue",
            ]
            missing_cols: set[str] = set(required_cols) - set(df.columns)

            if missing_cols:
                issues_found.append(f"Missing columns in {sales_file}: {missing_cols}")
            total_rows += len(df)

            if df["quantity_sold"].min() < 0:
                issues_found.append(f"Negative quantities in {sales_file}")
            if df["revenue"].min() < 0:
                issues_found.append(f"Negative revenue in {sales_file}")

        for data_type in ["promotions", "store_events", "customer_traffic"]:
            if data_type in file_paths and file_paths[data_type]:
                sample_file: str = file_paths[data_type][0]
                try:
                    if os.path.isfile(sample_file) and os.path.getsize(sample_file) > 0:
                        df = pd.read_parquet(sample_file, engine="pyarrow")
                        print(f"{data_type} data shape: {df.shape}")
                        print(f"{data_type} columns: {df.columns.tolist()}")
                    else:
                        issues_found.append(f"Invalid {data_type} file: {sample_file}")
                except Exception as e:
                    issues_found.append(f"Failed to read {data_type!r} file {sample_file!r}: {e}")

        validation_summary: dict[str, Any] = {
            "total_files_validated": len(file_paths["sales"][:10]),
            "total_rows": total_rows,
            "num_issues_found": len(issues_found),
            "issues": issues_found[:N],
        }
        if issues_found:
            print(f"Validation completed with {len(issues_found)} issues:")
            for issue in issues_found[:N]:
                print(f"  - {issue}")
        else:
            print(f"Validation passed! Total rows: {total_rows}")

        return validation_summary

    @task()
    def train_models_task(extract_result):
        # Local import to avoid heavy dependencies at DAG parse time
        import polars as pl

        from include.ml.trainer import ModelTrainer

        file_paths: dict[str, list[str]] = extract_result["file_paths"]
        print("Loading sales data from multiple files...")
        sales_dfs: list[pl.DataFrame] = []
        max_files: int = 50
        skipped_sales: int = 0

        for i, sales_file in enumerate(file_paths["sales"][:max_files]):
            try:
                df = pd.read_parquet(sales_file, engine="pyarrow")
                sales_dfs.append(df)
            except Exception as e:
                skipped_sales += 1
                print(f"  Skipping unreadable sales file {sales_file}: {e}")
                continue
            if (i + 1) % 10 == 0:
                print(f"  Loaded {i + 1} files...")
        if not sales_dfs:
            raise ValueError("No readable sales parquet files were loaded; aborting training")

        sales_df = pd.concat(sales_dfs, ignore_index=True)
        print(f"Combined sales data shape: {sales_df.shape}")
        daily_sales = (
            sales_df.groupby(["date", "store_id", "product_id", "category"])
            .agg(
                {
                    "quantity_sold": "sum",
                    "revenue": "sum",
                    "cost": "sum",
                    "profit": "sum",
                    "discount_percent": "mean",
                    "unit_price": "mean",
                }
            )
            .reset_index()
        )
        daily_sales = daily_sales.rename(columns={"revenue": "sales"})
        if file_paths.get("promotions"):
            try:
                promo_df = pd.read_parquet(file_paths["promotions"][0], engine="pyarrow")
                promo_summary = promo_df.groupby(["date", "product_id"])["discount_percent"].max().reset_index()
                promo_summary["has_promotion"] = 1
                daily_sales = daily_sales.merge(
                    promo_summary[["date", "product_id", "has_promotion"]],
                    on=["date", "product_id"],
                    how="left",
                )
                daily_sales["has_promotion"] = daily_sales["has_promotion"].fillna(0)
            except Exception as e:
                print(f"Skipping promotions merge due to error: {e}")
        if file_paths.get("customer_traffic"):
            traffic_dfs = []
            skipped_traffic = 0
            for traffic_file in file_paths["customer_traffic"][:10]:
                try:
                    traffic_dfs.append(pd.read_parquet(traffic_file, engine="pyarrow"))
                except Exception as e:
                    skipped_traffic += 1
                    print(f"  Skipping unreadable traffic file {traffic_file}: {e}")
            if traffic_dfs:
                traffic_df = pd.concat(traffic_dfs, ignore_index=True)
                traffic_summary = (
                    traffic_df.groupby(["date", "store_id"])
                    .agg({"customer_traffic": "sum", "is_holiday": "max"})
                    .reset_index()
                )
                daily_sales = daily_sales.merge(traffic_summary, on=["date", "store_id"], how="left")
            else:
                print("No readable traffic files; skipping merge")
        print(f"Final training data shape: {daily_sales.shape}")
        print(f"Columns: {daily_sales.columns.tolist()}")
        trainer = ModelTrainer()
        store_daily_sales = (
            daily_sales.groupby(["date", "store_id"])
            .agg(
                {
                    "sales": "sum",
                    "quantity_sold": "sum",
                    "profit": "sum",
                    "has_promotion": "mean",
                    "customer_traffic": "first",
                    "is_holiday": "first",
                }
            )
            .reset_index()
        )
        store_daily_sales["date"] = pd.to_datetime(store_daily_sales["date"])
        store_daily_sales_pl = pl.from_pandas(store_daily_sales)
        train_df, val_df, test_df = trainer.prepare_data(
            store_daily_sales_pl,
            target_col="sales",
            group_cols=["store_id"],
            categorical_cols=["store_id"],
        )
        print(f"Train shape: {train_df.shape}, Val shape: {val_df.shape}, Test shape: {test_df.shape}")
        results = trainer.train_all_models(train_df, val_df, test_df, target_col="sales")
        for model_name, model_results in results.items():
            if "metrics" in model_results:
                print(f"\n{model_name} metrics:")
                for metric, value in model_results["metrics"].items():
                    print(f"  {metric}: {value:.4f}")
        print("\nVisualization charts have been generated and saved to MLflow/MinIO")
        print("Charts include:")
        print("  - Model metrics comparison")
        print("  - Predictions vs actual values")
        print("  - Residuals analysis")
        print("  - Error distribution")
        print("  - Feature importance comparison")
        serializable_results = {}
        for model_name, model_results in results.items():
            serializable_results[model_name] = {"metrics": model_results.get("metrics", {})}

        import mlflow

        current_run = mlflow.active_run()
        current_run_id = current_run.info.run_id if current_run else None
        return {
            "training_results": serializable_results,
            "mlflow_run_id": current_run_id,
        }

    @task()
    def evaluate_models_task(training_result):
        from include.utilities.mlflow_utils import MLflowManager

        results = training_result["training_results"]
        best_model_name = None
        best_rmse = float("inf")
        for model_name, model_results in results.items():
            if "metrics" in model_results and "rmse" in model_results["metrics"]:
                if model_results["metrics"]["rmse"] < best_rmse:
                    best_rmse = model_results["metrics"]["rmse"]
                    best_model_name = model_name
        print(f"Best model: {best_model_name} with RMSE: {best_rmse:.4f}")
        # Find the run with best ensemble RMSE from the experiment
        mlflow_manager = MLflowManager()
        best_run = mlflow_manager.get_best_model(metric="ensemble_rmse", ascending=True)
        return {"best_model": best_model_name, "best_run_id": best_run["run_id"]}

    @task()
    def register_best_model_task(evaluation_result):
        # Local import to avoid heavy dependencies at DAG parse time
        from include.utilities.mlflow_utils import MLflowManager

        evaluation_result["best_model"]
        run_id = evaluation_result["best_run_id"]
        mlflow_manager = MLflowManager()
        model_versions = {}
        for model_name in ["xgboost", "lightgbm"]:
            version = mlflow_manager.register_model(run_id, model_name, model_name)
            model_versions[model_name] = version
            print(f"Registered {model_name} version: {version}")
        return model_versions

    @task()
    def transition_to_production_task(model_versions):
        # Local import to avoid heavy dependencies at DAG parse time
        from include.utilities.mlflow_utils import MLflowManager

        mlflow_manager = MLflowManager()
        for model_name, version in model_versions.items():
            mlflow_manager.transition_model_stage(model_name, version, "Production")
            print(f"Transitioned {model_name} v{version} to Production")
        return "Models transitioned to production"

    @task()
    def generate_performance_report_task(training_result, validation_summary):
        results = training_result["training_results"]
        report = {
            "timestamp": datetime.now().isoformat(),
            "data_summary": {
                "total_rows": (validation_summary.get("total_rows", 0) if validation_summary else 0),
                "files_validated": (validation_summary.get("total_files_validated", 0) if validation_summary else 0),
                "issues_found": (validation_summary.get("issues_found", 0) if validation_summary else 0),
                "issues": (validation_summary.get("issues", []) if validation_summary else []),
            },
            "model_performance": {},
        }
        if results:
            for model_name, model_results in results.items():
                if "metrics" in model_results:
                    report["model_performance"][model_name] = model_results["metrics"]
        import json

        with open("/tmp/performance_report.json", "w") as f:
            json.dump(report, f, indent=2)
        print("Performance report generated")
        print(f"Models trained: {list(report['model_performance'].keys())}")
        return report

    # Task dependencies using function calls
    extract_result = extract_data_task()
    validation_summary = validate_data_task(extract_result)
    training_result = train_models_task(extract_result)
    evaluation_result = evaluate_models_task(training_result)
    model_versions = register_best_model_task(evaluation_result)
    _ = transition_to_production_task(model_versions)
    report = generate_performance_report_task(training_result, validation_summary)
    cleanup = BashOperator(
        task_id="cleanup",
        bash_command="rm -rf /tmp/sales_data /tmp/performance_report.json || true",
    )
    report >> cleanup


sales_forecast_training_dag = sales_forecast_training()
