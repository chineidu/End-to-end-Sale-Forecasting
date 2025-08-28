"""
Simple DAG for Airflow 3.x
This DAG demonstrates basic tasks using only core operators.
"""

from datetime import datetime, timedelta

from airflow.sdk import DAG


def print_hello(**context):
    """Simple function to print hello"""
    print("Hello from Airflow 3.x!")
    print(f"Execution date: {context['ds']}")
    return "Hello World"


def print_date(**context):
    """Function to print current date"""
    from datetime import datetime

    current_date = datetime.now()
    print(f"Current date and time: {current_date}")
    return str(current_date)


def calculate_sum(**context):
    """Function to calculate a simple sum"""
    result = 10 + 20
    print(f"10 + 20 = {result}")
    return result


# Define the DAG using context manager
with DAG(
    "simple_dag",
    default_args={
        "owner": "airflow",
        "depends_on_past": False,
        "start_date": datetime(2023, 1, 1),
        "email_on_failure": False,
        "email_on_retry": False,
        "retries": 1,
        "retry_delay": timedelta(minutes=5),
    },
    description="A simple DAG for Airflow 3.x",
    schedule=timedelta(days=1),
    start_date=datetime(2023, 1, 1),
    catchup=False,
    tags={"example", "simple"},
) as dag:
    # For now, let's create a very simple task structure
    # We'll use Python callables directly in the DAG
    pass
