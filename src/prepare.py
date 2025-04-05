import datetime
import holidays
import constants
from logger import Logger
import pandas as pd
from sklearn.preprocessing import StandardScaler

def day_from_row(row: pd.Series) -> datetime.date:
    day = int(row[constants.day_col])
    month = int(row[constants.month_col])
    year = int(row[constants.year_col])
    try:
        return datetime.date(year, month, day)
    except ValueError:
        Logger().log_error(f"Помилка дати: {day}-{month}-{year} | Рядок: {row.name}")
        return datetime.date.today()

def is_day_weekend(date: datetime.date) -> int:
    country_holidays = holidays.country_holidays(constants.country, years=date.year)
    return int(date.weekday() >= 5 or date in country_holidays)

def is_holiday_or_weekend(row: pd.Series) -> int:
    return is_day_weekend(day_from_row(row))

def is_last_day_holiday_or_weekend(row: pd.Series) -> int:
    return is_day_weekend(day_from_row(row) - datetime.timedelta(days=1))

def is_next_day_holiday_or_weekend(row: pd.Series) -> int:
    return is_day_weekend(day_from_row(row) + datetime.timedelta(days=1))

def get_season(row: pd.Series) -> int:
    date = day_from_row(row)
    if (date.month == 12 and date.day >= 1) or (1 <= date.month <= 2):
        return 1
    elif 3 <= date.month <= 5:
        return 2
    elif 6 <= date.month <= 8:
        return 3
    elif 9 <= date.month <= 11:
        return 4
    return 0

def is_workhour(row: pd.Series) -> int:
    hour = row[constants.hour_col]
    if pd.isnull(hour):
        return 0
    try:
        hour = int(hour)
    except Exception:
        return 0
    return int(9 <= hour < 18)

def remove_outliers_percentile(df: pd.DataFrame, column: str, lower_percentile: float = 0.001, upper_percentile: float = 0.999) -> pd.DataFrame:
    lower_bound = df[column].quantile(lower_percentile)
    upper_bound = df[column].quantile(upper_percentile)
    cleaned_df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    removed_count = df.shape[0] - cleaned_df.shape[0]
    Logger().log_info(f"Видалено {removed_count} рядків за межами перцентилів")
    return cleaned_df

def normalize_features(df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
    numeric_features = df[feature_cols].select_dtypes(include=['int64', 'float64']).columns.tolist()
    if not numeric_features:
        Logger().log_warning("Немає числових ознак для нормалізації")
        return df
    scaler = StandardScaler()
    df[numeric_features] = scaler.fit_transform(df[numeric_features])
    Logger().log_info(f"Ознаки нормалізовано: {numeric_features}")
    return df

def load_dataframe(path: str) -> pd.DataFrame:
    if path.endswith(".csv"):
        return pd.read_csv(path, delimiter=constants.csv_delim)
    elif path.endswith(".xlsx"):
        return pd.read_excel(path)
    else:
        raise ValueError(f"Формат не підтримується: {path}")

def validate_input_data(df: pd.DataFrame, required_columns: list):
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Відсутні обов'язкові колонки: {missing}")
    Logger().log_info("Усі обов'язкові колонки на місці.")

def log_missing_values(df: pd.DataFrame):
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if not missing.empty:
        Logger().log_warning(f"Пропущені значення у даних:\n{missing}")