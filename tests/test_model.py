import os
import json
import numpy as np


def test_data_schema_basic():
    """Pre-train: Перевірка наявності та форми підготовлених даних"""
    # Перевіряємо, чи існують файли
    train_path = "data/prepared/train_data.npz"
    assert os.path.exists(train_path), f"Data not found: {train_path}"

    # Перевіряємо, чи є всередині 'X' та 'y'
    data = np.load(train_path)
    assert 'X' in data.files, "Train data is missing 'X' arrays"
    assert 'y' in data.files, "Train data is missing 'y' labels"

    X = data['X']
    y = data['y']
    # Перевіряємо, чи X і y однакової довжини
    assert X.shape[0] == y.shape[0], "Inconsistent number of samples between X and y"
    # Перевіряємо, чи є достатньо даних
    assert X.shape[0] >= 50, "Too few samples for training"


def test_artifacts_exist():
    """Post-train: Перевірка створення артефактів"""
    assert os.path.exists("models/best_model.pkl") or os.path.exists("model.pkl"), "Model file not found"
    assert os.path.exists("metrics.json"), "metrics.json not found"
    assert os.path.exists("confusion_matrix.png"), "confusion_matrix.png not found"


def test_quality_gate_accuracy():
    """Post-train: Перевірка, чи модель достатньо якісна (Quality Gate)"""
    threshold = float(os.getenv("ACCURACY_THRESHOLD", "0.70"))

    with open("metrics.json", "r", encoding="utf-8") as f:
        metrics = json.load(f)

    acc = float(metrics.get("accuracy", 0))
    assert acc >= threshold, f"Quality Gate Failed! Accuracy: {acc:.4f} < {threshold:.2f}"
