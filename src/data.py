"""Loading and preparing the Pokemon dataset."""

import pandas as pd

NUMERIC = ["hp", "attack", "defense", "sp_attack", "sp_defense",
           "speed", "height_m", "weight_kg", "capture_rate", "base_happiness"]
CATEGORICAL = ["type_1", "generation"]
TARGET = "base_experience"


def load_data(path="data/data.csv"):
    """Return X (numbers), y (target) and the list of feature names.

    Categorical columns are one-hot encoded (one 0/1 column per category).
    """
    df = pd.read_csv(path)
    X_df = pd.get_dummies(df[NUMERIC + CATEGORICAL], columns=CATEGORICAL, dtype=float)
    X = X_df.to_numpy(dtype=float)
    y = df[TARGET].to_numpy(dtype=float)
    return X, y, list(X_df.columns)