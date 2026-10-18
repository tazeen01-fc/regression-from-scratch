import os

import matplotlib.pyplot as plt
import numpy as np

from src.data import load_data
from src.regression import (
    LassoGD, RidgeGD, StandardScaler, train_test_split, cross_val_rmse,
)

os.makedirs("images", exist_ok=True)

X, y, feature_names = load_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, seed=42)
scaler = StandardScaler().fit(X_train)
X_tr, X_te = scaler.transform(X_train), scaler.transform(X_test)

# ---- Plot 1: predicted vs actual (Lasso, lam=1) ----
model = LassoGD(lr=0.1, n_iters=2000, lam=1.0).fit(X_tr, y_train)
pred = model.predict(X_te)

plt.figure(figsize=(5.5, 5))
plt.scatter(y_test, pred, alpha=0.6)
lims = [min(y_test.min(), pred.min()), max(y_test.max(), pred.max())]
plt.plot(lims, lims, "r--", label="perfect prediction")
plt.xlabel("Actual base_experience")
plt.ylabel("Predicted base_experience")
plt.title("Predicted vs actual (test set)")
plt.legend()
plt.tight_layout()
plt.savefig("images/predicted_vs_actual.png", dpi=150)


# ---- Plot 2: cross-validation curves ----
ridge_lams = [0.001, 0.01, 0.03, 0.1, 0.3, 1.0]
lasso_lams = [0.01, 0.1, 0.3, 1.0, 2.0, 4.0, 8.0]
ridge_cv = [cross_val_rmse(lambda: RidgeGD(lr=0.1, n_iters=2000, lam=l), X_train, y_train)[0]
            for l in ridge_lams]
lasso_cv = [cross_val_rmse(lambda: LassoGD(lr=0.1, n_iters=2000, lam=l), X_train, y_train)[0]
            for l in lasso_lams]

plt.figure(figsize=(6, 4.5))
plt.plot(ridge_lams, ridge_cv, "o-", label="Ridge")
plt.plot(lasso_lams, lasso_cv, "s-", label="Lasso")
plt.xscale("log")
plt.xlabel("lambda (log scale)")
plt.ylabel("cross-validated RMSE")
plt.title("Choosing lambda with 5-fold CV")
plt.legend()
plt.tight_layout()
plt.savefig("images/cv_curves.png", dpi=150)


# ---- Plot 3: Lasso regularization path ----
path_lams = [0.01, 0.03, 0.1, 0.3, 1.0, 2.0, 4.0, 8.0, 16.0]
weights = np.array([
    LassoGD(lr=0.1, n_iters=2000, lam=l).fit(X_tr, y_train).w for l in path_lams
])

top = np.argsort(np.abs(weights[0]))[::-1][:8]      # 8 biggest features
plt.figure(figsize=(8, 4.5))                         # changed: wider
for j in top:
    plt.plot(path_lams, weights[:, j], "o-", label=feature_names[j])
plt.xscale("log")
plt.xlabel("lambda (log scale)")
plt.ylabel("weight")
plt.title("Lasso path: weights shrink to zero as lambda grows")
plt.legend(fontsize=7, loc="center left", bbox_to_anchor=(1.02, 0.5))   # changed: legend outside
plt.tight_layout()
plt.savefig("images/lasso_path.png", dpi=150, bbox_inches="tight")      # changed: keeps legend in the saved image


print("Saved 3 plots in the images folder.")
plt.show()