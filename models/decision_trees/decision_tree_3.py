import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.metrics import mean_squared_error

df = pd.read_csv("data/combined_table.csv")

df["date"] = pd.to_datetime(df["date"])

# seasonality setup
df["month"] = df["date"].dt.month

def get_season(month):
    if month in [12, 1, 2]:
        return "winter"
    elif month in [3, 4, 5]:
        return "spring"
    elif month in [6, 7, 8]:
        return "summer"
    else:
        return "fall"

df["season"] = df["month"].apply(get_season)

df["t"] = range(len(df))

df["civic_diff1"] = df["civic_sales"].diff()
df["civic_lag1"] = df["civic_sales"].shift(1)
df["civic_lag2"] = df["civic_sales"].shift(2)
df["civic_ma3"] = df["civic_sales"].rolling(3).mean()
df = df.dropna().reset_index(drop=True)

# create season dummies
df = pd.get_dummies(df, columns=["season"], drop_first=True)
df = df.dropna().reset_index(drop=True)

# target
y = df["civic_sales"]

# predictors
X = df.drop(columns=["civic_sales", "date"])

# 75/25 split
split = int(len(df) * 0.75)

X_train = X.iloc[:split]
X_test = X.iloc[split:]

y_train = y.iloc[:split]
y_test = y.iloc[split:]

date_test = df["date"].iloc[split:]

# fit model
model = DecisionTreeRegressor(
    max_depth=4,
    random_state=42
)

model.fit(X_train, y_train)

y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

train_mse = mean_squared_error(y_train, y_train_pred)
test_mse = mean_squared_error(y_test, y_test_pred)

print("Train MSE:", train_mse)
print("Test MSE:", test_mse)
print("Train RMSE:", np.sqrt(train_mse))
print("Test RMSE:", np.sqrt(test_mse))

#plot
plt.figure(figsize=(12, 6))
plt.plot(date_test, y_test, label="Actual Civic Sales")
plt.plot(date_test, y_test_pred, label="Predicted Civic Sales")
plt.xlabel("Date")
plt.ylabel("Civic Sales")
plt.title("Actual vs Predicted Civic Sales (75/25 Split)")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# decision tree
plt.figure(figsize=(18, 10))
plot_tree(
    model,
    feature_names=X.columns,
    filled=True,
    rounded=True
)
plt.title("Decision Tree for Civic Sales")
plt.show()