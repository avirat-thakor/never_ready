import numpy as np
import pandas as pd

from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit



def add_civic_lags(df, max_lag):
    """
    Helper function for grid search over # of lags: Add civic_sales_lag1, ..., civic_sales_lag{max_lag} to df.
    Drops rows that don't have full lag history.
    """
    df = df.copy()
    for lag in range(1, max_lag + 1):
        df[f"civic_lag{lag}"] = df["civic_sales"].shift(lag)
    lag_history = ["civic_sales"] + [f"civic_lag{lag}" for lag in range(1, max_lag + 1)]
    df = df.dropna(subset=lag_history)
    return df

df = pd.read_csv("data/combined_table.csv")

base_features = [
    "corolla_sales",
    "sentra_sales",
    "cpi",
    "unemploy",
    "gas",
    "fedfunds",
    "tdsp",
]

candidate_lags = [0, 1, 2, 3, 4]
test_size = 12
results = []

#Grid search over # of lags
for max_lag in candidate_lags:

    if max_lag == 0:
        df_lagged = df.copy()
        feature_cols = base_features
    else:
        df_lagged = add_civic_lags(df, max_lag)
        lag_features = [f"civic_lag{lag}" for lag in range(1, max_lag + 1)]
        feature_cols = base_features + lag_features

    X = df_lagged[feature_cols].values
    y = df_lagged["civic_sales"].values


    train_size = len(df_lagged) - test_size
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    dates_test = df_lagged["date"].iloc[train_size:]

    # LASSO with cross-validation to choose lambda
    alphas = np.logspace(-3, 2, 50)
    tscv = TimeSeriesSplit(n_splits=5)
    lasso_pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("lasso", LassoCV(
                alphas=alphas,
                cv=tscv,
                random_state=0,
                max_iter=10000
            )),
        ]
    )

    lasso_pipeline.fit(X_train, y_train)
    lasso_cv = lasso_pipeline.named_steps["lasso"]

    coefs = lasso_cv.coef_
    non_zero_mask = coefs != 0

    y_pred_train = lasso_pipeline.predict(X_train)
    y_pred_test = lasso_pipeline.predict(X_test)

    mse_train = mean_squared_error(y_train, y_pred_train)
    mse_test = mean_squared_error(y_test, y_pred_test)
    r2_test = r2_score(y_test, y_pred_test)


    print(f"\nLASSO with max_lag = {max_lag} ")
    print(f"Optimal alpha (lambda): {lasso_cv.alpha_:.6f}")
    print(f"n_features: {len(feature_cols)} | n_kept: {int(non_zero_mask.sum())}")
    print(feature_cols)
    print(f"Train MSE: {mse_train:.2f} | Test MSE: {mse_test:.2f} | Test R^2: {r2_test:.3f}")

    results.append({
        "max_lag": max_lag,
        "mse_train": mse_train,
        "mse_test": mse_test,
        "r2_test": r2_test,
        "n_features": len(feature_cols),
        "n_kept": int(non_zero_mask.sum()),
    })
   
print("Train std:", y_train.std())
print("Test std :", y_test.std())