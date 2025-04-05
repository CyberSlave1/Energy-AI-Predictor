from logger import Logger
import pandas as pd
import constants
import prepare
import train
import util
import argparse
import os
from datetime import datetime
from pathlib import Path
from typing import Tuple
from sklearn.metrics import mean_squared_error
from energy import print_help, print_available_features
from features import get_feature_list
import sys


if any(x in sys.argv for x in ['--help', '-help', '--h', '-h']):
    print_help()
    sys.exit()
if '--show_features' in sys.argv and all(x not in sys.argv for x in ['--mode']):
    print_available_features()
    sys.exit()

def load_and_prepare_df(path_to_df: str, normalize=False, feature_list=None, require_function_col=True) -> pd.DataFrame:
    df = prepare.load_dataframe(path_to_df)
    df[constants.is_weekend_col] = df.apply(prepare.is_holiday_or_weekend, axis=1)
    df[constants.is_prev_weekend_col] = df.apply(prepare.is_last_day_holiday_or_weekend, axis=1)
    df[constants.is_next_weekend_col] = df.apply(prepare.is_next_day_holiday_or_weekend, axis=1)
    df[constants.season_col] = df.apply(prepare.get_season, axis=1)
    df[constants.workhour_col] = df.apply(prepare.is_workhour, axis=1)
    prepare.log_missing_values(df)
    required = feature_list or []
    if require_function_col:
        required = [constants.function_col] + required
    prepare.validate_input_data(df, required_columns=required)
    if normalize and feature_list:
        df = prepare.normalize_features(df, feature_list)
    return df

def clear_noise(df: pd.DataFrame, lower_percentile=0.0, upper_percentile=1.0) -> pd.DataFrame:
    return prepare.remove_outliers_percentile(df, constants.function_col, lower_percentile, upper_percentile)

def train_new_model_auto(df: pd.DataFrame, feature_list: list, n_trials: int):
    return train.hyperparameter_search(df, feature_list, constants.function_col, n_trials=n_trials)

def train_new_model_by_params(df: pd.DataFrame, feature_list: list, params: dict):
    return train.train_random_forest(df, feature_list, constants.function_col, params)

def test_model_classic(model, df: pd.DataFrame, feature_list: list, plot_save=False, plot_name="test_classic") -> None:
    X_train, X_test, y_train, y_test = train.train_test_split(
        df[feature_list], df[[constants.function_col]], test_size=0.2, random_state=4
    )
    train.evaluate_model(model, X_test, y_test, plot_save=plot_save, plot_path=f"../data/plots/{plot_name}.png")

def test_model_last_samples(model, df: pd.DataFrame, feature_list: list, samples_count: int = 1000, plot_save=False, plot_name="test_last") -> None:
    last_samples_df = df.tail(samples_count)
    test_x = last_samples_df[feature_list]
    test_y = last_samples_df[[constants.function_col]]
    train.evaluate_model(model, test_x, test_y, plot_save=plot_save, plot_path=f"../data/plots/{plot_name}.png")

def load_and_predict(path_to_df: str, path_to_model: str, features_list: list, path_to_save: str, normalize=False) -> None:
    if not Path(path_to_df).exists():
        Logger().log_error(f"Файл для прогнозу не знайдено: {path_to_df}")
        raise FileNotFoundError(path_to_df)
    model = util.load_model(path_to_model)
    df = load_and_prepare_df(
        path_to_df,
        normalize=normalize,
        feature_list=features_list,
        require_function_col=False
    )
    predicted = model.predict(df[features_list])
    df[constants.function_col] = predicted
    df.to_excel(path_to_save, index=False)
    Logger().log_info(f"Файл прогнозу збережено: {path_to_save}")

def log_configuration(df: pd.DataFrame, model_path: str, lower_percentile: float, upper_percentile: float):
    Logger().log_info(f"Кількість рядків у датасеті: {df.shape[0]}")
    Logger().log_info(f"Шлях до моделі: {model_path}")
    Logger().log_info(f"Процентиль для викидів: нижній = {lower_percentile}, верхній = {upper_percentile}")

def compare_and_save_if_better(model, new_params: dict, model_path: str, params_path: str, df: pd.DataFrame, feature_list: list):

    samples_count = 1000
    last_samples_df = df.tail(samples_count)
    X_new = last_samples_df[feature_list]
    y_new = last_samples_df[[constants.function_col]]

    y_pred_new = model.predict(X_new)
    new_mse = mean_squared_error(y_new, y_pred_new)

    if Path(params_path).exists():
        old_params = util.load_params(params_path)
        old_features = old_params.get("features_used")

        if old_features is None:
            Logger().log_warning("Файл параметрів не містить features_used. Пропускаємо порівняння моделей.")
            Logger().log_info("Зберігаємо нову модель і параметри.")
            util.save_model(model_path, model)
            new_params["features_used"] = feature_list
            util.save_params(params_path, new_params)
            return

        missing_features = [f for f in old_features if f not in last_samples_df.columns]
        if missing_features:
            Logger().log_warning(f"Неможливо порівняти з попередньою моделлю — відсутні ознаки: {missing_features}")
            Logger().log_info("Зберігаємо нову модель і параметри.")
            util.save_model(model_path, model)
            new_params["features_used"] = feature_list
            util.save_params(params_path, new_params)
            return

        old_model = util.load_model(model_path)
        y_old = last_samples_df[[constants.function_col]]
        y_pred_old = old_model.predict(last_samples_df[old_features])
        old_mse = mean_squared_error(y_old, y_pred_old)

        Logger().log_info(f"Порівняння моделей (1000 останніх записів) — метод той самий, що в оцінці:")
        Logger().log_info(f"Старий MSE: {old_mse:.5f}, Новий MSE: {new_mse:.5f}")

        if new_mse < old_mse:
            Logger().log_info("Нова модель краща — зберігаємо")
            new_params["features_used"] = feature_list
            util.save_model(model_path, model)
            util.save_params(params_path, new_params)
        else:
            Logger().log_info("Стара модель краща — залишаємо без змін")
    else:
        Logger().log_info("Параметри відсутні — зберігаємо поточну модель")
        new_params["features_used"] = feature_list
        util.save_model(model_path, model)
        util.save_params(params_path, new_params)

def load_and_train_new_auto(path_to_df: str, path_to_save_model: str, path_to_save_params: str,
                            n_trials: int, lower_percentile: float, upper_percentile: float,
                            log_config: bool = False, compare_models: bool = False,
                            feature_list: list = None, normalize: bool = False) -> Tuple:
    df = load_and_prepare_df(path_to_df, normalize=normalize, feature_list=feature_list)
    df = clear_noise(df, lower_percentile=lower_percentile, upper_percentile=upper_percentile)
    if log_config:
        log_configuration(df, path_to_save_model, lower_percentile, upper_percentile)

    if n_trials > 0:
        model, params = train_new_model_auto(df, feature_list, n_trials=n_trials)
    else:
        Logger().log_info("n_trials = 0, пропускаємо Optuna — виконуємо лише прогноз за існуючою моделлю")
        model = util.load_model(path_to_save_model)
        params = util.load_params(path_to_save_params)
        feature_list = params.get("features_used", feature_list)
        return model, df, feature_list

    if compare_models:
        compare_and_save_if_better(model, params, path_to_save_model, path_to_save_params, df, feature_list)
        Logger().log_info("Перезавантажуємо модель та параметри після порівняння...")
        params = util.load_params(path_to_save_params)
        model = util.load_model(path_to_save_model)
        feature_list = params.get("features_used", feature_list)
    else:
        params_to_save = dict(params)
        params_to_save["features_used"] = feature_list
        util.save_model(path_to_save_model, model)
        util.save_params(path_to_save_params, params_to_save)
    return model, df, feature_list

def load_and_train_new(path_to_df: str, path_to_params: str, path_to_save_model: str, lower_percentile: float, upper_percentile: float, log_config: bool = False, feature_list: list = None, normalize: bool = False):
    df = load_and_prepare_df(path_to_df, normalize=normalize, feature_list=feature_list)
    df = clear_noise(df, lower_percentile=lower_percentile, upper_percentile=upper_percentile)
    if log_config:
        log_configuration(df, path_to_save_model, lower_percentile, upper_percentile)
    params = util.load_params(path_to_params)
    model = train_new_model_by_params(df, feature_list, params)
    util.save_model(path_to_save_model, model)
    return model, df

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', required=False, choices=['train_auto', 'train_manual', 'predict', 'test', 'auto'])
    parser.add_argument('--train_path', default='../data/training_data.csv')
    parser.add_argument('--predict_path', default='../data/predict_data.csv')
    parser.add_argument('--model_path', default='../models/model.pkl')
    parser.add_argument('--params_path', default='../models/params.json')
    parser.add_argument('--output_path', default=None)
    parser.add_argument('--n_trials', type=int, default=100)
    parser.add_argument('--plot_save', action='store_true')
    parser.add_argument('--lower_percentile', type=float, default=0.001)
    parser.add_argument('--upper_percentile', type=float, default=0.999)
    parser.add_argument('--log_config', action='store_true')
    parser.add_argument('--compare_models', action='store_true')
    parser.add_argument('--exclude_features', nargs='+', help='Список ознак для виключення з моделі через пробіл')
    parser.add_argument('--show_features', action='store_true', help='Показати доступні ознаки та вийти')
    parser.add_argument('--normalize_features', action='store_true', help='Нормалізувати числові ознаки')
    args = parser.parse_args()

    if args.show_features:
        print("\nДоступні ознаки:")
        for f in get_feature_list():
            print(f" - {f}")
        sys.exit(0)

    if not args.mode:
        Logger().log_error("Не вказано --mode. Приклад: --mode predict")
        sys.exit(1)

    feature_list = get_feature_list(exclude=args.exclude_features or [])

    Logger().init_logging(log_dir="logs", level="DEBUG", size_mb=1, max_files=1)
    os.makedirs("../data/plots", exist_ok=True)
    timestamp = datetime.now().strftime("%d_%m_%Y_%H_%M")

    Logger().log_info("==============================")
    Logger().log_info(f"[ STARTING MODE: {args.mode.upper()} ]")
    Logger().log_info("==============================")

    if args.mode in ['train_auto', 'train_manual', 'test', 'auto'] and not args.train_path:
        Logger().log_error("Не вказано --train_path для режиму, що потребує навчальних даних")
        sys.exit(1)
    if args.mode in ['predict', 'auto'] and not args.predict_path:
        Logger().log_error("Не вказано --predict_path для прогнозу")
        sys.exit(1)

    if args.mode == 'train_auto':
        load_and_train_new_auto(args.train_path, args.model_path, args.params_path, args.n_trials, args.lower_percentile, args.upper_percentile, log_config=args.log_config, compare_models=args.compare_models, feature_list=feature_list, normalize=args.normalize_features)

    elif args.mode == 'train_manual':
        load_and_train_new(args.train_path, args.params_path, args.model_path, args.lower_percentile, args.upper_percentile, log_config=args.log_config, feature_list=feature_list, normalize=args.normalize_features)

    elif args.mode == 'predict':
        output_path = args.output_path or f"../data/predicted_{timestamp}.xlsx"
        load_and_predict(args.predict_path, args.model_path, feature_list, output_path, normalize=args.normalize_features)

    elif args.mode == 'test':
        df = load_and_prepare_df(args.train_path, normalize=args.normalize_features, feature_list=feature_list)
        model = util.load_model(args.model_path)
        test_model_classic(model, df, feature_list, plot_save=args.plot_save, plot_name=f"classic_{timestamp}")
        test_model_last_samples(model, df, feature_list, plot_save=args.plot_save, plot_name=f"last_{timestamp}")

    elif args.mode == 'auto':
        model, df, feature_list = load_and_train_new_auto(args.train_path, args.model_path, args.params_path, args.n_trials, args.lower_percentile, args.upper_percentile, log_config=args.log_config, compare_models=args.compare_models, feature_list=feature_list, normalize=args.normalize_features)
        output_path = args.output_path or f"../data/predicted_{timestamp}.xlsx"
        load_and_predict(args.predict_path, args.model_path, feature_list, output_path, normalize=args.normalize_features)
        test_model_classic(model, df, feature_list, plot_save=args.plot_save, plot_name=f"auto_classic_{timestamp}")
        test_model_last_samples(model, df, feature_list, plot_save=args.plot_save, plot_name=f"auto_last_{timestamp}")