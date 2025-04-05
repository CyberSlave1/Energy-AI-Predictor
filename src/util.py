import pickle
import json
from logger import Logger

def save_model(path: str, model) -> None:
    try:
        with open(path, 'wb') as f:
            pickle.dump(model, f)
        Logger().log_info(f"Модель збережено у файл: {path}")
    except Exception as e:
        Logger().log_error(f"Не вдалося зберегти модель: {e}")


def load_model(path: str):
    try:
        with open(path, 'rb') as f:
            model = pickle.load(f)
        Logger().log_info(f"Модель завантажено з файлу: {path}")
        return model
    except Exception as e:
        Logger().log_error(f"Не вдалося завантажити модель: {e}")
        raise


def load_params(path: str) -> dict:
    try:
        with open(path, 'r') as f:
            params = json.load(f)
        Logger().log_info(f"Параметри завантажено з файлу: {path}")
        return params
    except Exception as e:
        Logger().log_error(f"Не вдалося завантажити параметри: {e}")
        raise


def save_params(path: str, params: dict) -> None:
    try:
        with open(path, 'w') as f:
            json.dump(params, f, indent=4)
        Logger().log_info(f"Параметри збережено у файл: {path}")
    except Exception as e:
        Logger().log_error(f"Не вдалося зберегти параметри: {e}")


