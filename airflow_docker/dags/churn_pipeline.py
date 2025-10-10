from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import sys
import os

# add project path so airflow can import scripts
sys.path.append(r'C:\Users\Maxwe\OneDrive\Documents\Classification\Churn Prediction\Churn-Prediction')

# import existing models
from src.EDA import run_eda
from src.model_evaluation import model_evaluation
from src.preprocessing import preprocessing
from src.model_training import model_training

# default arguments for all tasks
default_args = {
    'owner' : 'Maxwell',
    'depends_on_past' : False,
    'retries' : 2,
    'retry_delay' : timedelta(minutes=3)
}

# define DAG 
with DAG(
    dag_id='customer churn prediction model',
    default_args=default_args,
    description='Automated ML Pipeline for Customer Churn Prediction',
    schedule_interval='@weekly',
    start_date=datetime(2025,10,10),
    catchup=False
) as dag:
    
    # define each airflow task
    eda_task = PythonOperator(
        task_id='data_cleaning_and_feature_engineering',
        python_callable=run_eda
    )

    preprocess_task = PythonOperator(
        task_id='train_test_split_and_preprocessing',
        python_callable=preprocessing
    )

    model_training_task = PythonOperator(
        task_id='model_training',
        python_callable=model_training
    )

    model_eval_task = PythonOperator(
        task_id= 'model_evaluation',
        python_callable=model_evaluation
    )

    # set task dependencies
    eda_task >> preprocess_task >> model_training_task >> model_eval_task