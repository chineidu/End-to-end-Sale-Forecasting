import json

import pendulum

from airflow.sdk import dag, task


@dag(
    schedule=None,
    start_date=pendulum.datetime(2025, 8, 30, tz="UTC"),
    catchup=False,
    tags=["example"],
)
def simple_tutorial_taskflow_api() -> None:
    """A simple tutorial demonstrating the TaskFlow API."""

    @task()
    def extract() -> dict[str, float]:
        data_str: str = '{"1001": 234.57, "1002": 753.05, "1003": 652.43, "1004": 307.89}'
        return json.loads(data_str)

    @task(
        # When multiple_outputs=True is enabled, a task returns a dictionary. Each key in this dictionary corresponds to
        # a separate output, which can be accessed individually by downstream tasks. This is helpful when a single task
        # generates several distinct results that need to be handled separately in later steps.
        multiple_outputs=True,
    )
    def transform(data: dict[str, float]) -> dict[str, float]:
        total_value: float = sum(data.values())
        return {"total": total_value}

    @task()
    def load(data: dict[str, float]) -> None:
        print(f"Total value: {data['total']:.2f}")

    # Build the flow
    data = extract()
    transformed_data = transform(data)
    load(transformed_data)

simple_tutorial_taskflow_api_dag = simple_tutorial_taskflow_api()
