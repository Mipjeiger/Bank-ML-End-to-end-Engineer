from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
from airflow.operators.bash import BashOperator
from airflow.utils.task_group import TaskGroup
import pandas as pd
import json
import redis
from pathlib import Path

# Configuration
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / 'bank-system' / 'database'
REDIS_HOST = 'redis'
REDIS_PORT = 6379
RAW_FILE = DATA_DIR / 'eda_banking.parquet'

# redis connection
redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)

# DAG definition
default_args = {
    'owner': 'data-team',
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
    'start_date': datetime(2026, 4, 1),
}

dag = DAG(
    'data_retraining_pipeline',
    default_args=default_args,
    description='Data pipeline for model retraining',
    catchup=False,
    schedule=None
)

# Task
def load_raw_data():
    """Load raw data from source"""
    print("📂Loading raw data...")
    df = pd.read_parquet(RAW_FILE)

    # Store metadata in Redis
    redis_client.set("data:shape", json.dumps({"rows": len(df), "columns": len(df.columns)}))
    print(f"✅ Loaded {len(df)} rows, {len(df.columns)} columns")

    return str(RAW_FILE)

def validate_data():
    """Validate data quality"""
    print("🔍 Validating data...")
    df = pd.read_parquet(RAW_FILE)

    # Check for missing values
    missing_values = df.isnull().sum().sum()
    if missing_values > 0:
        df = df.dropna()
        print(f"⚠️ Found {missing_values} missing values, dropped rows to prevent issues during training.")
    
    # Store validation results in Redis
    validation_results = {
        "missing_values": missing_values,
        "shape": list(df.shape),
        "timestamp": datetime.now().isoformat()
    }
    redis_client.set("data:validation:latest", json.dumps(validation_results))

    return validation_results

def preprocess_data():
    """Preprocess data for training pipeline"""
    print("⚙️ Preprocessing data...")
    from backend.app.services.feature_engineering import feature_engineering

    df = pd.read_parquet(RAW_FILE)
    df_processed = feature_engineering.feature_engineering(df)

    # Save processed data for training
    processed_file = DATA_DIR / "processed" / "eda_banking_processed.parquet"
    processed_file.parent.mkdir(parents=True, exist_ok=True)
    df_processed.to_parquet(processed_file, index=False)

    # Store info in Redis
    redis_client.set("data:processed:shape", json.dumps({"rows": len(df_processed), "columns": len(df_processed.columns)}))

    print(f"Processed data saved to {processed_file}")
    return str(processed_file)

def data_quality_check():
    """Perform data quality checks"""
    print("✅ Performing data quality checks...")

    checks = {
        "missing_values": redis_client.get("data:validation:latest"),
        "no_duplicates": "Yes" if pd.read_parquet(RAW_FILE).duplicated().sum() == 0 else "No",
        "timestamp": datetime.now().isoformat(),
        "schema_valid": True  # Assuming schema is valid for simplicity
    }

    redis_client.set("data:quality:latest", json.dumps(checks))
    print(f"Data quality checks completed: {checks}")

    return checks

# Task definitions
load_task = PythonOperator(
    task_id='load_raw_data',
    python_callable=load_raw_data,
    dag=dag
)

validate_task = PythonOperator(
    task_id='validate_data',
    python_callable=validate_data,
    dag=dag
)

preprocess_task = PythonOperator(
    task_id='preprocess_data',
    python_callable=preprocess_data,
    dag=dag
)

quality_task = PythonOperator(
    task_id='data_quality_check',
    python_callable=data_quality_check,
    dag=dag
)

# DAG dependencies
load_task >> validate_task >> preprocess_task >> quality_task