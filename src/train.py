import argparse
import os
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

mlflow.set_tracking_uri("http://127.0.0.1:8080")
mlflow.set_experiment("Traffic_Signs_DVC_Pipeline")


def load_prepared_data(data_dir):
    train_data = np.load(os.path.join(data_dir, 'train_data.npz'))
    test_data = np.load(os.path.join(data_dir, 'test_data.npz'))
    return train_data['X'], train_data['y'], test_data['X'], test_data['y']


def train(n_estimators, max_depth, data_dir='data/prepared'):
    with mlflow.start_run():
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)

        print("Завантаження підготовлених даних...")
        X_train, y_train, X_test, y_test = load_prepared_data(data_dir)

        print(f"Навчання RandomForest (n_estimators={n_estimators}, max_depth={max_depth})...")
        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        accuracy = accuracy_score(y_test, preds)

        mlflow.log_metric("test_accuracy", accuracy)
        mlflow.sklearn.log_model(model, "model")
        print(f"Готово! Accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--max_depth", type=int, default=10)
    args = parser.parse_args()

    train(args.n_estimators, args.max_depth)
