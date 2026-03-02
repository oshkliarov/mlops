import argparse
import os
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from PIL import Image
from tqdm import tqdm

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("Traffic_Signs_Classification_Final")


def load_data(data_dir, csv_path, img_size=(32, 32), limit=None):
    df = pd.read_csv(csv_path)
    if limit:
        print(f"Використовується лише {limit} зображень для тесту!")
        df = df.sample(n=limit, random_state=42)

    data, labels = [], []
    print("Завантажую зображення...")
    for _, row in tqdm(df.iterrows(), total=df.shape[0]):
        try:
            img_path = os.path.join(data_dir, row['Path'])
            # Конвертуємо у Grayscale (L), щоб зменшити кількість фічів (для кращого графіка)
            image = Image.open(img_path).resize(img_size).convert("L")
            data.append(np.array(image).flatten())
            labels.append(row['ClassId'])
        except Exception:
            pass

    return np.array(data), np.array(labels)


def plot_feature_importance(model, save_path="feature_importance.png"):
    importance = model.feature_importances_
    importance_img = importance.reshape(32, 32)

    plt.figure(figsize=(6, 6))
    plt.imshow(importance_img, cmap='hot')
    plt.title('Feature Importance (Pixel Importance)')
    plt.colorbar()
    plt.savefig(save_path)
    plt.close()
    return save_path


def train(n_estimators, max_depth, fast_mode):
    limit = 2000 if fast_mode else None

    with mlflow.start_run():
        mlflow.set_tag("developer", "Oleksandr")
        mlflow.set_tag("model_type", "RandomForest")
        mlflow.set_tag("dataset", "GTSRB")

        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("fast_mode", fast_mode)

        X, y = load_data('data/raw', 'data/raw/Train.csv', limit=limit)
        X = X / 255.0

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        print(f"Тренування RandomForest (n_estimators={n_estimators}, max_depth={max_depth})...")
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            n_jobs=-1,
            random_state=42,
            verbose=0
        )
        model.fit(X_train, y_train)

        preds_train = model.predict(X_train)
        preds_test = model.predict(X_test)

        train_acc = accuracy_score(y_train, preds_train)
        test_acc = accuracy_score(y_test, preds_test)
        test_f1 = f1_score(y_test, preds_test, average='weighted')

        mlflow.log_metric("train_accuracy", train_acc)
        mlflow.log_metric("test_accuracy", test_acc)
        mlflow.log_metric("test_f1_score", test_f1)

        plot_path = plot_feature_importance(model)
        mlflow.log_artifact(plot_path)
        mlflow.sklearn.log_model(model, "random_forest_model")

        print(f"Готово! Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f} | F1: {test_f1:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--max_depth", type=int, default=10)
    parser.add_argument("--fast", action="store_true")
    args = parser.parse_args()

    train(args.n_estimators, args.max_depth, args.fast)
