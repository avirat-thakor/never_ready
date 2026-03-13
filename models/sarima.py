import pandas as pd
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt
# from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Data
df = pd.read_csv("data/combined_table.csv")
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date")[["date","civic_sales","gas","unemploy","fedfunds", "cpi", 
                             "csi", "tdsp","corolla_sales", "sentra_sales"]]
df = df.set_index("date").asfreq("MS")

# Initial ADF Test
from statsmodels.tsa.stattools import adfuller
result1 = adfuller(df["civic_sales"])
print("ADF Stat: %f" % result1[0])
print("p-value: %f" % result1[1])

# Differencing for stationarity
df_diff = df["civic_sales"].diff().dropna()

# Confirm stationarity
result2 = adfuller(df_diff)
print("ADF Test (Differenced)")
print("ADF Stat: %f" % result2[0])
print("p-value: %f" % result2[1])

plot_acf(df_diff, lags=36)
plt.savefig("visualization/sarimax/acf_diff.png")

plot_pacf(df_diff, lags=36)
plt.savefig("visualization/sarimax/pacf_diff.png")

train = df.iloc[:-12]
test = df.iloc[-12:]

# ARIMA Model 
model_ARIMA = SARIMAX(
    train["civic_sales"],
    order=(0, 1, 1),
    seasonal_order=(0, 1, 1, 12),
)

fit = model_ARIMA.fit(disp=False)

y_train = train["civic_sales"]
y_test = test["civic_sales"]

y_train_pred = fit.predict(start=train.index[0], end=train.index[-1])
y_test_pred = fit.get_forecast(steps=len(test)).predicted_mean
conf_int = fit.get_forecast(steps=len(test)).conf_int()

train_mse = mean_squared_error(y_train, y_train_pred)
test_mse = mean_squared_error(y_test, y_test_pred)


print(fit.summary())

print(f"Train MSE:  {train_mse:.2f}")
print(f"Test MSE:   {test_mse:.2f}")

# SARIMAX Model
model_exog = SARIMAX(
    train["civic_sales"],
    exog = train[["gas","unemploy","fedfunds"]],
    order=(0, 1, 1),
    seasonal_order=(0, 1, 1, 12),
)

fit_exog = model_exog.fit(disp=False)

y_train_pred_exog = fit_exog.predict(
    start=train.index[0],
    end=train.index[-1],
     exog = train[["gas","unemploy","fedfunds"]],
)

y_test_pred_exog = fit_exog.get_forecast(
    steps=len(test),
     exog = test[["gas","unemploy","fedfunds"]],
).predicted_mean

train_mse_exog = mean_squared_error(y_train, y_train_pred_exog)
test_mse_exog = mean_squared_error(y_test, y_test_pred_exog)
train_rmse_exog = train_mse_exog ** 0.5
test_rmse_exog = test_mse_exog ** 0.5

print(fit_exog.summary())
print(f"SARIMAX Train MSE: {train_mse_exog:.2f}")
print(f"SARIMAX Test MSE: {test_mse_exog:.2f}")
print(f"SARIMAX Train RMSE: {train_rmse_exog:.2f}")
print(f"SARIMAX Test RMSE: {test_rmse_exog:.2f}")

'''
# SARIMAX + LASSO
lasso_var = ["gas", "unemploy", "fedfunds", "cpi", "csi", "tdsp", 
             "corolla_sales", "sentra_sales"]

lasso_pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("lasso", LassoCV(cv=5, random_state=42, max_iter=10000))
])

lasso_pipe.fit(train[lasso_var], train["civic_sales"])

lasso_model = lasso_pipe.named_steps["lasso"]
selected_var = pd.Series(lasso_model.coef_, index=lasso_var)
selected_var = selected_var[selected_var != 0].index.tolist()

if len(selected_var) == 0:
    selected_var = ["corolla_sales"]

model_lasso = SARIMAX(
    train["civic_sales"],
    exog=train[selected_var],
    order=(0, 1, 1),
    seasonal_order=(0, 1, 1, 12),
)

fit_lasso = model_lasso.fit(disp=False)

y_train_pred_lasso = fit_lasso.predict(
    start=train.index[0],
    end=train.index[-1],
    exog=train[selected_var],
)

y_test_pred_lasso = fit_lasso.get_forecast(
    steps=len(test),
    exog=test[selected_var],
).predicted_mean

train_mse_lasso = mean_squared_error(y_train, y_train_pred_lasso)
test_mse_lasso = mean_squared_error(y_test, y_test_pred_lasso)


print("LASSO selected:", selected_var)
print(f"SARIMAX + LASSO Train MSE: {train_mse_lasso:.2f}")
print(f"SARIMAX + LASSO Test MSE: {test_mse_lasso:.2f}")
'''

# SARIMAX + Shock Dummy
df["shock_dummy"] = 0
df.loc[(df.index >= "2008-09-01") & (df.index <= "2009-06-01"), "shock_dummy"] = 1
df.loc[(df.index >= "2020-03-01") & (df.index <= "2020-06-01"), "shock_dummy"] = 1
df.loc[(df.index >= "2012-01-01") & (df.index <= "2012-04-01"), "shock_dummy"] = 1

train_shock = df.iloc[:-12]
test_shock = df.iloc[-12:]

y_train_shock = train_shock["civic_sales"]
y_test_shock = test_shock["civic_sales"]

shock_var = ["gas", "unemploy", "fedfunds", "shock_dummy"]

model_shock = SARIMAX(
    train_shock["civic_sales"],
    exog=train_shock[shock_var],
    order=(0, 1, 1),
    seasonal_order=(0, 1, 1, 12),
)

fit_shock = model_shock.fit(disp=False)

y_train_pred_shock = fit_shock.predict(
    start=train_shock.index[0],
    end=train_shock.index[-1],
    exog=train_shock[shock_var],
)

y_test_pred_shock = fit_shock.get_forecast(
    steps=len(test_shock),
    exog=test_shock[shock_var],
).predicted_mean

train_mse_shock = mean_squared_error(y_train_shock, y_train_pred_shock)
test_mse_shock = mean_squared_error(y_test_shock, y_test_pred_shock)
train_rmse_shock = train_mse_shock ** 0.5
test_rmse_shock = test_mse_shock ** 0.5

print(fit_shock.summary())
print(f"SARIMAX + Dummy Train MSE: {train_mse_shock:.2f}")
print(f"SARIMAX + Dummy Test MSE: {test_mse_shock:.2f}")
print(f"SARIMAX + Dummy Train RMSE: {train_rmse_shock:.2f}")
print(f"SARIMAX + Dummy Test RMSE: {test_rmse_shock:.2f}")

# SARIMAX + Shock DUmmy Graph
plt.figure(figsize=(14, 6))
plt.plot(train_shock.index, y_train_shock, color="gray", label="Train Actual")
plt.plot(test_shock.index, y_test_shock, color="blue", label="Test Actual")
plt.plot(train_shock.index, y_train_pred_shock, color="red", linestyle="--", label="Train Pred")
plt.plot(test_shock.index, y_test_pred_shock, color="orange", linestyle="--", label="Test Pred")
plt.axvline(test_shock.index[0], color="green", linestyle=":", label="Train/Test Split")
plt.title("Civic Sales Forecast, SARIMAX + Shock Dummy")
plt.xlabel("Date")
plt.ylabel("Civic Sales")
plt.legend()
plt.tight_layout()
plt.savefig("visualization/sarimax/Civic_Sales_SARIMAX_Shocks_Forecast.png")
plt.show()

#SARIMAX Graph
plt.figure(figsize=(14, 6))
plt.plot(train.index, y_train, color="gray", label="Train Actual")
plt.plot(test.index, y_test, color="blue", label="Test Actual")
plt.plot(train.index, y_train_pred_exog, color="red", linestyle="--", label="Train Pred")
plt.plot(test.index, y_test_pred_exog, color="orange", linestyle="--", label="Test Pred")
plt.axvline(test.index[0], color="green", linestyle=":", label="Train/Test Split")
plt.title("Civic Sales Forecast, SARIMAX")
plt.xlabel("Date")
plt.ylabel("Civic Sales")
plt.legend()
plt.tight_layout()
plt.savefig("visualization/sarimax/Civic_Sales_SARIMAX_Forecast.png")
plt.show()
