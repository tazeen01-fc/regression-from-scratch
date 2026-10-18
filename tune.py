import numpy as np

from src.data import load_data
from src.regression import (
    LinearRegressionGD, RidgeGD, LassoGD, StandardScaler,
    train_test_split, rmse, mae, r2_score, cross_val_rmse,
)

# 1. Load features: numbers + one-hot encoded categories
X, y, feature_names = load_data()
print(f"Number of features: {X.shape[1]}")

# 2. Lock away the test set. We will not touch it until the very end.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, seed=42)


# 3. Choose lambda with 5-fold cross-validation on the TRAINING data only
def tune(model_class, lambdas):
    print(f"\n{model_class.__name__}")
    print(f"{'lam':>7} | {'CV RMSE':>8} | {'+/-':>5}")
    best_lam, best_score = None, np.inf
    for lam in lambdas:
        mean, std = cross_val_rmse(
            lambda: model_class(lr=0.1, n_iters=2000, lam=lam), X_train, y_train)
        print(f"{lam:>7} | {mean:>8.2f} | {std:>5.2f}")
        if mean < best_score:
            best_lam, best_score = lam, mean
    print(f"Best lam: {best_lam}")
    return best_lam


best_ridge = tune(RidgeGD, [0.001, 0.01, 0.03, 0.1, 0.3, 1.0])
best_lasso = tune(LassoGD, [0.01, 0.1, 0.3, 1.0, 2.0, 4.0, 8.0])

# 4. Final evaluation on the untouched test set
scaler = StandardScaler().fit(X_train)
X_tr, X_te = scaler.transform(X_train), scaler.transform(X_test)

final = {
    "Linear": LinearRegressionGD(lr=0.1, n_iters=2000),
    f"Ridge (lam={best_ridge})": RidgeGD(lr=0.1, n_iters=2000, lam=best_ridge),
    f"Lasso (lam={best_lasso})": LassoGD(lr=0.1, n_iters=2000, lam=best_lasso),
}

print("\nFinal test results")
print(f"{'model':<18} | {'RMSE':>7} | {'MAE':>7} | {'R2':>6}")
print("-" * 48)
for name, model in final.items():
    model.fit(X_tr, y_train)
    pred = model.predict(X_te)
    print(f"{name:<18} | {rmse(y_test, pred):>7.2f} | {mae(y_test, pred):>7.2f} | {r2_score(y_test, pred):>6.3f}")

# 5. Which features did Lasso keep?
lasso = final[f"Lasso (lam={best_lasso})"]
kept = [(n, w) for n, w in zip(feature_names, lasso.w) if w != 0]
kept.sort(key=lambda t: abs(t[1]), reverse=True)
print(f"\nLasso kept {len(kept)} of {len(feature_names)} features. Top 10:")
for n, w in kept[:10]:
    print(f"  {n:<28} {w:>8.2f}")