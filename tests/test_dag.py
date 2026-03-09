import os
import pytest
from airflow.models import DagBag


def test_dag_import():
    """Перевірка, чи всі DAG-файли завантажуються без помилок імпорту"""
    dag_folder = os.path.join(os.path.dirname(__file__), '..', 'dags')
    dagbag = DagBag(dag_folder=dag_folder, include_examples=False)

    assert len(dagbag.import_errors) == 0, f"DAG import errors: {dagbag.import_errors}"
