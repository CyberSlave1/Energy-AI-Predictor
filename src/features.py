def get_feature_list(exclude: list = None) -> list:
    exclude = exclude or []
    all_features = [
        'year', 'month', 'day', 'hour',
        'is_workhour', 'weekday', 'is_weekend',
        'is_prev_weekend', 'is_next_weekend',
        'season', 'temp', 'blackout',
        'phases', 'deenergized'
    ]
    return [f for f in all_features if f not in exclude]