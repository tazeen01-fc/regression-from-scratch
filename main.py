import pandas as pd

from src.regression import (
    LinearRegressionGD, RidgeGD, LassoGD,
    StandardScaler, train_test_split, mse, rmse, mae, r2_score,
)

FEATURES = ["hp", "attack", "defense", "sp_attack", "sp_defense",
            "speed", "height_m", "weight_kg", "capture_rate", "base_happiness"]
TARGET = "base_experience"

# 1. Load data
df = pd.read_csv("data/data.csv")
X = df[FEATURES].to_numpy(dtype=float)
y = df[TARGET].to_numpy(dtype=float)

# 2. Split FIRST, then scale using the training data only
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, seed=42)
scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# 3. Train three models
models = {
    "Linear": LinearRegressionGD(lr=0.1, n_iters=2000),
    "Ridge (lam=0.1)": RidgeGD(lr=0.1, n_iters=2000, lam=0.1),
    "Lasso (lam=0.1)": LassoGD(lr=0.1, n_iters=2000, lam=0.1),
}

# 4. Evaluate
print(f"{'model':<16} | {'RMSE':>7} | {'MAE':>7} | {'R2':>6}")
print("-" * 46)
for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    print(f"{name:<16} | {rmse(y_test, pred):>7.2f} | {mae(y_test, pred):>7.2f} | {r2_score(y_test, pred):>6.3f}")

# 5. Baseline: always predict the training average
baseline = [y_train.mean()] * len(y_test)
print(f"{'Baseline (mean)':<16} | {rmse(y_test, baseline):>7.2f} | {mae(y_test, baseline):>7.2f} | {r2_score(y_test, baseline):>6.3f}")