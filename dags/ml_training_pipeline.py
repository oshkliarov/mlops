import json
import os
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import BranchPythonOperator
from airflow.operators.empty import EmptyOperator

# Базові налаштування нашого DAG
default_args = {
    'owner': 'mlops',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'retries': 1,
    'retry_delay': timedelta(minutes=2),
}

# Створюємо сам об'єкт пайплайну
dag = DAG(
    'ml_training_pipeline',
    default_args=default_args,
    description='Continuous Training Pipeline for Traffic Signs',
    schedule_interval=None,  # Запускаємо вручну або через тригер
    catchup=False,
)

# 1. Підготовка даних (виконуємо bash-команду)
task_prepare = BashOperator(
    task_id='prepare_data',
    # Створюємо папки і запускаємо твій скрипт
    bash_command='mkdir -p data/raw && cp -r data/test_raw/* data/raw/ && python src/prepare.py',
    dag=dag,
)

# 2. Тренування моделі
task_train = BashOperator(
    task_id='train_model',
    bash_command='python src/optimize.py',
    dag=dag,
)


# 3. Логіка розгалуження (Quality Gate)
def check_accuracy(**kwargs):
    # Читаємо метрики, які згенерував optimize.py
    try:
        with open('metrics.json', 'r') as f:
            metrics = json.load(f)
        # Отримуємо точність
        accuracy = float(metrics.get('accuracy', 0))
    except Exception as e:
        print(f"Error reading metrics: {e}")
        accuracy = 0.0

    print(f"Current model accuracy: {accuracy}")

    # Якщо точність >= 0.50 (як ми робили для мок-даних), йдемо на реєстрацію
    if accuracy >= 0.50:
        return 'register_model'
    else:
        return 'stop_pipeline'


branch_task = BranchPythonOperator(
    task_id='evaluate_model',
    python_callable=check_accuracy,
    dag=dag,
)

# 4. Реєстрація моделі (гілка успіху)
task_register = BashOperator(
    task_id='register_model',
    bash_command='echo "Accuracy is good! Registering model in MLflow Model Registry..."',
    dag=dag,
)

# 5. Зупинка пайплайну (гілка провалу)
task_stop = EmptyOperator(
    task_id='stop_pipeline',
    dag=dag,
)

# Будуємо послідовність (граф) виконання:
task_prepare >> task_train >> branch_task
branch_task >> task_register
branch_task >> task_stop
