import os
import random
from typing import Any, Dict

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import optuna
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import hydra


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def load_prepared_data(data_dir: str):
    """Завантаження .npz файлів, які ми згенерували в ЛР2"""
    abs_dir = to_absolute_path(data_dir)
    train_data = np.load(os.path.join(abs_dir, 'train_data.npz'))
    test_data = np.load(os.path.join(abs_dir, 'test_data.npz'))

    # Явно дістаємо кожну змінну, щоб точно не переплутати!
    X_train = train_data['X']
    y_train = train_data['y']

    X_test = test_data['X']
    y_test = test_data['y']

    print(
        f"Розмірності: X_train={X_train.shape}, y_train={y_train.shape}, X_test={X_test.shape}, y_test={y_test.shape}")

    # Повертаємо у стандартному sklearn-порядку
    return X_train, X_test, y_train, y_test


def build_model(model_type: str, params: Dict[str, Any], seed: int) -> Any:
    if model_type == "random_forest":
        return RandomForestClassifier(random_state=seed, n_jobs=-1, **params)
    raise ValueError(f"Unknown model.type='{model_type}'")


def evaluate(model: Any, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray) -> float:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    # Повертаємо Accuracy (бо у нас мультикласова класифікація 43 знаків)
    return float(accuracy_score(y_test, y_pred))


def suggest_params(trial: optuna.Trial, model_type: str, cfg: DictConfig) -> Dict[str, Any]:
    if model_type == "random_forest":
        space = cfg.hpo.random_forest
        return {
            "n_estimators": trial.suggest_int("n_estimators", space.n_estimators.low, space.n_estimators.high),
            "max_depth": trial.suggest_int("max_depth", space.max_depth.low, space.max_depth.high),
            "min_samples_split": trial.suggest_int("min_samples_split", space.min_samples_split.low,
                                                   space.min_samples_split.high),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", space.min_samples_leaf.low,
                                                  space.min_samples_leaf.high),
        }
    raise ValueError(f"Unknown model.type='{model_type}'.")


def objective_factory(cfg: DictConfig, X_train, X_test, y_train, y_test):
    def objective(trial: optuna.Trial) -> float:
        params = suggest_params(trial, cfg.model.type, cfg)

        # Nested Run: створюємо дочірній запуск (trial) всередині батьківського (Study)
        with mlflow.start_run(nested=True, run_name=f"trial_{trial.number:03d}"):
            mlflow.set_tag("trial_number", trial.number)
            mlflow.set_tag("sampler", cfg.hpo.sampler)
            mlflow.set_tag("seed", cfg.seed)
            mlflow.log_params(params)

            model = build_model(cfg.model.type, params=params, seed=cfg.seed)
            score = evaluate(model, X_train, y_train, X_test, y_test)

            mlflow.log_metric(cfg.hpo.metric, score)
            return score

    return objective


def main(cfg: DictConfig) -> None:
    set_global_seed(cfg.seed)

    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    mlflow.set_experiment(cfg.mlflow.experiment_name)

    # Завантаження даних
    X_train, X_test, y_train, y_test = load_prepared_data(cfg.data.prepared_dir)

    # Створення Sampler'а
    if cfg.hpo.sampler.lower() == "tpe":
        sampler = optuna.samplers.TPESampler(seed=cfg.seed)
    elif cfg.hpo.sampler.lower() == "random":
        sampler = optuna.samplers.RandomSampler(seed=cfg.seed)
    else:
        sampler = optuna.samplers.TPESampler(seed=cfg.seed)

    # Parent Run (Основний запуск HPO)
    with mlflow.start_run(run_name=f"hpo_{cfg.hpo.sampler}") as parent_run:
        mlflow.set_tag("sampler", cfg.hpo.sampler)
        mlflow.set_tag("seed", cfg.seed)
        mlflow.log_dict(OmegaConf.to_container(cfg, resolve=True), "config.json")

        # Запуск Optuna
        print(f"Починаємо пошук гіперпараметрів (Sampler: {cfg.hpo.sampler}, Trials: {cfg.hpo.n_trials})...")
        study = optuna.create_study(direction=cfg.hpo.direction, sampler=sampler)
        objective = objective_factory(cfg, X_train, X_test, y_train, y_test)
        study.optimize(objective, n_trials=cfg.hpo.n_trials)

        # Логування найкращих результатів
        best_trial = study.best_trial
        print(f"\nНайкращий результат: {best_trial.value:.4f}")
        print("Найкращі параметри:", best_trial.params)

        mlflow.log_metric(f"best_{cfg.hpo.metric}", float(best_trial.value))
        mlflow.log_dict(best_trial.params, "best_params.json")

        # Тренування фінальної моделі з найкращими параметрами
        print("Тренування фінальної моделі...")
        best_model = build_model(cfg.model.type, params=best_trial.params, seed=cfg.seed)
        best_model.fit(X_train, y_train)

        # РОБИМО ПЕРЕДБАЧЕННЯ ФІНАЛЬНОЮ МОДЕЛЛЮ ДЛЯ МЕТРИК
        y_pred = best_model.predict(X_test)
        final_accuracy = accuracy_score(y_test, y_pred)

        # 1. Збереження моделі
        os.makedirs("models", exist_ok=True)
        joblib.dump(best_model, "models/best_model.pkl")
        mlflow.log_artifact("models/best_model.pkl")

        # 2. Збереження метрик у JSON (для Quality Gate у GitHub Actions)
        import json
        metrics_dict = {"accuracy": float(final_accuracy)}
        with open("metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics_dict, f, ensure_ascii=False, indent=2)

        # 3. Генерація та збереження Confusion Matrix (для звіту CML)
        import matplotlib.pyplot as plt
        from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(cmap=plt.cm.Blues)
        plt.title("Confusion Matrix (Best Model)")
        plt.savefig("confusion_matrix.png")
        plt.close()  # Закриваємо графік, щоб він не висів у пам'яті

        # Логування в MLflow
        if cfg.mlflow.log_model:
            mlflow.sklearn.log_model(best_model, artifact_path="model")


@hydra.main(version_base=None, config_path="../config", config_name="config")
def hydra_entry(cfg: DictConfig) -> None:
    main(cfg)


if __name__ == "__main__":
    hydra_entry()
