import timeit
import optuna
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from logger import Logger
from typing import Tuple, List, Dict

def evaluate_model(model, X, y, err_for_acc: int = 5, plot_save: bool = False, plot_path: str = None) -> None:
    import numpy as np
    from matplotlib import pyplot as plt

    y_actual = y[y.columns[0]].values.reshape(-1, 1)
    y_predicted = model.predict(X).reshape(-1, 1)

    mse = mean_squared_error(y_actual, y_predicted)
    r2 = r2_score(y_actual, y_predicted)
    mae = mean_absolute_error(y_actual, y_predicted)
    percent_error = mae / np.mean(y_actual) * 100
    diff = (np.abs(y_predicted - y_actual) / y_actual) * 100

    acc = (diff < err_for_acc).mean()
    acc_x2 = (diff < 2 * err_for_acc).mean()
    acc_x3 = (diff < 3 * err_for_acc).mean()

    log_block = (
        f"\n[Оцінка моделі]\n"
        f"------------------------------\n"
        f"MSE (середньоквадратична):    {mse:.5f}\n"
        f"MAE (середня абсолютна):      {mae:.5f}\n"
        f"R^2 (детермінації):           {r2:.5f}\n"
        f"Відносна помилка:             {percent_error:.5f}%\n"
        f"\nТочність прогнозу (відхилення < X%):\n"
        f"  <{err_for_acc}%    — {acc:.5f}\n"
        f"  <{err_for_acc*2}%   — {acc_x2:.5f}\n"
        f"  <{err_for_acc*3}%   — {acc_x3:.5f}\n"
    )

    Logger().log_info(log_block)

    plt.figure(figsize=(10, 6))
    plt.plot(y_actual, label='Фактичні', color='blue', linewidth=2)
    plt.plot(y_predicted, label='Прогнозовані', color='red', linewidth=2)
    plt.title('Фактичні vs Прогнозовані')
    plt.xlabel('Зразки')
    plt.ylabel('Значення цільової змінної')
    plt.legend()
    plt.grid(True)

    if plot_save and plot_path:
        plt.savefig(plot_path)
        Logger().log_info(f"Графік збережено: {plot_path}")
    else:
        plt.show()

def log_feature_importances(model, features: list):
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        ranked = sorted(zip(features, importances), key=lambda x: x[1], reverse=True)
        Logger().log_info("\nВажливість ознак:")
        for f, imp in ranked:
            Logger().log_info(f" - {f}: {imp:.4f}")

def train_random_forest(df: pd.DataFrame, feature_columns: List[str], target_column: str, rf_params: Dict) -> RandomForestRegressor:
    default_rf_params = {
        "n_estimators": 100,
        "max_depth": None,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "max_features": "log2",
        "bootstrap": True
    }
    rf_params = {**default_rf_params, **rf_params}
    X = df[feature_columns]
    y = df[[target_column]]
    model = RandomForestRegressor(**rf_params, random_state=42, n_jobs=-1)
    Logger().log_debug(f"Тренування моделі з параметрами: {rf_params}")
    start = timeit.default_timer()
    model.fit(X, y.values.ravel())
    Logger().log_debug(f"Тренування завершено за {timeit.default_timer() - start:.2f} сек")
    log_feature_importances(model, feature_columns)
    return model

def hyperparameter_search(df: pd.DataFrame, feature_columns: List[str], target_column: str, n_trials: int = 100) -> Tuple[RandomForestRegressor, Dict]:
    X = df[feature_columns]
    y = df[target_column]
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 800, step=50),
            "max_depth": trial.suggest_int("max_depth", 10, 50),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 15),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
            "bootstrap": trial.suggest_categorical("bootstrap", [True, False])
        }
        model = RandomForestRegressor(**params, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train.values.ravel())
        return mean_squared_error(y_valid, model.predict(X_valid))

    Logger().log_debug(f"Запуск Optuna з {n_trials} ітераціями")
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    Logger().log_info(f"Найкращі параметри: {study.best_params}")
    best_model = RandomForestRegressor(**study.best_params, random_state=42, n_jobs=-1)
    best_model.fit(X, y)
    log_feature_importances(best_model, feature_columns)
    return best_model, study.best_params