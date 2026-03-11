import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt

# Data
df = pd.read_csv("data/combined_table.csv")
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date")[["date","civic_sales","gas","unemploy","fedfunds", "cpi", "csi", "tdsp",
                             "corolla_sales", "sentra_sales"]]
df = df.set_index("date").asfreq("MS")

# Initial ADF Test
from statsmodels.tsa.stattools import adfuller
result1 = adfuller(df["civic_sales"])
print("ADF Stat: %f" % result1[0])
print("p-value: %f" % result1[1])

# Differencing to get stationarity
df_diff = df["civic_sales"].diff().dropna()

# ADF Test to confirm stationarity
result2 = adfuller(df_diff)
print("ADF Test (Differenced)")
print("ADF Stat: %f" % result2[0])
print("p-value: %f" % result2[1])

# Show the ACF and PACF plots to determine AR or MA terms
plot_acf(df_diff, lags=36)
plot_pacf(df_diff, lags=36)
plt.show()

# Testing on the last 12 data points
train = df.iloc[:-12]
test = df.iloc[-12:]

# ARIMA Model 
model_ARIMA = SARIMAX(
    train["civic_sales"],
    order=(0, 1, 1),
    seasonal_order=(0, 1, 1, 12),
)

fit = model_ARIMA.fit(disp=False)

# Predictions for SARIMA
y_train = train["civic_sales"]
y_test = test["civic_sales"]

y_train_pred = fit.predict(start=train.index[0], end=train.index[-1])
y_test_pred = fit.get_forecast(steps=len(test)).predicted_mean
conf_int = fit.get_forecast(steps=len(test)).conf_int()

# Metrics
train_mse = mean_squared_error(y_train, y_train_pred)
test_mse = mean_squared_error(y_test, y_test_pred)


print(fit.summary())

print(f"Train MSE:  {train_mse:.2f}")
print(f"Test MSE:   {test_mse:.2f}")

# SARIMAX Model
model_exog = SARIMAX(
    train["civic_sales"],
    exog = train[["gas","unemploy","fedfunds"]],
    # exog=train[["gas","unemploy","fedfunds", "cpi", "csi", "tdsp",
    #                          "corolla_sales", "sentra_sales"]],
    order=(0, 1, 1),
    seasonal_order=(0, 1, 1, 12),
)

fit_exog = model_exog.fit(disp=False)

# Predictions for SARIMAX
y_train_pred_exog = fit_exog.predict(
    start=train.index[0],
    end=train.index[-1],
     exog = train[["gas","unemploy","fedfunds"]],
    # exog=train[["gas","unemploy","fedfunds", "cpi", "csi", "tdsp",
    #                          "corolla_sales", "sentra_sales"]],
)

y_test_pred_exog = fit_exog.get_forecast(
    steps=len(test),
     exog = test[["gas","unemploy","fedfunds"]],
    # exog=test[["gas","unemploy","fedfunds", "cpi", "csi", "tdsp",
    #                          "corolla_sales", "sentra_sales"]],
).predicted_mean

# MSE for SARIMAX
train_mse_exog = mean_squared_error(y_train, y_train_pred_exog)
test_mse_exog = mean_squared_error(y_test, y_test_pred_exog)

print(fit_exog.summary())

print(f"SARIMAX Train MSE: {train_mse_exog:.2f}")
print(f"SARIMAX Test MSE: {test_mse_exog:.2f}")

# Looking at the intial series, I suspected that the data was stochatic and would
# require differencing to make it stationary. The ADF test confirmed this, with 
# ADF p-value = 0.415156 which -> Fail to reject unit root and so we conclude 
# the data is non-stationary. An ADF test on the differenced seres showed that the 
# data is now stationary, with ADF p-value = 0.012345.

# The ACF and PACF plots of the differenced series suggest that an ARIMA(0,1,1) model 
# with seasonal order (0,1,1,12) is appropriate; i.e. there where spikes every 12 lags
# on both the PACF and PACF. The model was then fitted on the training data and evaluated 
# on the test data. The train MSE is 30,119,255 and the test MSE is 3,989,068,indicating 
# that the model performs reasonably well in prediction.

# Next I tried a SARIMAX model with the same order but with the exogenous variables included. 
# The train MSE for the SARIMAX model is 28,668,089 and the test MSE is 3,263,447, which shows 
# a slight improvement over the ARIMA model. This suggests that the exogenous variables do provide 
# some additional predictive power for forecasting civic sales.

# Notably, a SARIMAX model that inclulded all the features available had a higher train and test MSE
# of 38,567,618 and 8,174,137 respectively, which suggests that including all the regressors 
# may have led to overfitting.

# Summary
#           Train MSE    Test MSE
# ARIMA     30,119,255   3,989,068
# SARIMAX   28,668,089   3,263,447
# SARIMAX   38,567,618   8,174,137

future_steps = 6

future_forecast = fit.get_forecast(steps=future_steps)
future_mean = future_forecast.predicted_mean
future_ci = future_forecast.conf_int()

future_index = pd.date_range(
    start=df.index[-1] + pd.DateOffset(months=1),
    periods=future_steps,
    freq="MS"
)

future_mean.index = future_index
future_ci.index = future_index

print("Future Civic Sales:")
print(future_mean)