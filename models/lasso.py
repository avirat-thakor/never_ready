import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

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
                max_iter=100000
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




# Refit for optimal lag length (1) to create visualization
chosen_lag = 1
df_lagged_1 = add_civic_lags(df, chosen_lag)
df_lagged_1["date"] = pd.to_datetime(df_lagged_1["date"])

lag_features_1 = [f"civic_lag{lag}" for lag in range(1, chosen_lag + 1)]
feature_cols_1 = base_features + lag_features_1

X_1 = df_lagged_1[feature_cols_1].values
y_1 = df_lagged_1["civic_sales"].values

train_size_1 = len(df_lagged_1) - test_size
X_train_1, X_test_1 = X_1[:train_size_1], X_1[train_size_1:]
y_train_1, y_test_1 = y_1[:train_size_1], y_1[train_size_1:]

lasso_pipeline.fit(X_train_1, y_train_1)
y_pred_train_1 = lasso_pipeline.predict(X_train_1)
y_pred_test_1 = lasso_pipeline.predict(X_test_1)


plt.figure(figsize=(12, 6))

plt.plot(df_lagged_1["date"].iloc[:train_size_1], y_train_1, label="Train Actual", color="black", alpha=0.5)
plt.plot(df_lagged_1["date"].iloc[train_size_1:], y_test_1, label="Test Actual", color="blue", linewidth=2)

plt.plot(df_lagged_1["date"].iloc[:train_size_1], y_pred_train_1, label="Train Pred", color="red", linestyle="--", alpha=0.7)
plt.plot(df_lagged_1["date"].iloc[train_size_1:], y_pred_test_1, label="Test Pred", color="orange", linewidth=2)

plt.axvline(x=df_lagged_1["date"].iloc[train_size_1], color='green', linestyle=':', label='Train/Test Split')
plt.title(f"Civic Sales Forecast (Lasso with {chosen_lag} Lag)")
plt.legend()

# Space out X-axis markers
ax = plt.gca()
ax.xaxis.set_major_locator(mdates.YearLocator(2))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("visualization/lasso/lasso_lag1_plot.png", dpi=300, bbox_inches="tight")
plt.close()



# Refit the optimal lag 1 model using only the last 40 points of training data 
# (omitting 200 in earlier periods) as an effort to invetigate the cause of 
# the inflated train mse compared to test

X_1_short = df_lagged_1[feature_cols_1].values
y_1_short = df_lagged_1["civic_sales"].values

train_size_1_short = len(df_lagged_1) - test_size
X_train_1_short, X_test_1_short = X_1[200:train_size_1], X_1[train_size_1:]
y_train_1_short, y_test_1_short = y_1[200:train_size_1], y_1[train_size_1:]

lasso_pipeline.fit(X_train_1_short, y_train_1_short)
y_pred_train_1_short = lasso_pipeline.predict(X_train_1_short)
y_pred_test_1_short = lasso_pipeline.predict(X_test_1_short)


plt.figure(figsize=(12, 6))

plt.plot(df_lagged_1["date"].iloc[200:train_size_1], y_train_1_short, label="Train Actual", color="black", alpha=0.5)
plt.plot(df_lagged_1["date"].iloc[train_size_1:], y_test_1_short, label="Test Actual", color="blue", linewidth=2)

plt.plot(df_lagged_1["date"].iloc[200:train_size_1], y_pred_train_1_short, label="Train Pred", color="red", linestyle="--", alpha=0.7)
plt.plot(df_lagged_1["date"].iloc[train_size_1:], y_pred_test_1_short, label="Test Pred", color="orange", linewidth=2)

plt.axvline(x=df_lagged_1["date"].iloc[train_size_1], color='green', linestyle=':', label='Train/Test Split')
plt.title(f"Civic Sales Forecast (Lasso with {chosen_lag} Lag)")
plt.legend()

plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("visualization/lasso/lasso_lag1_plot_omitting_earlier_training_data.png", dpi=300, bbox_inches="tight")
plt.close()

mse_train_1_short = mean_squared_error(y_train_1_short, y_pred_train_1_short)
mse_test_1_short = mean_squared_error(y_test_1_short, y_pred_test_1_short)
r2_test_1_short = r2_score(y_test_1_short, y_pred_test_1_short)

print(f"\nLASSO with max_lag = {1} with only recent training data")
print(f"Optimal alpha (lambda): {lasso_cv.alpha_:.6f}")
print(f"n_features: {len(feature_cols_1)} | n_kept: {int(non_zero_mask.sum())}")
print(feature_cols_1)
print(f"Train MSE: {mse_train_1_short:.2f} | Test MSE: {mse_test_1_short:.2f} | Test R^2: {r2_test_1_short:.3f}")
