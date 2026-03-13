import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.metrics import mean_squared_error

df = pd.read_csv("data/combined_table.csv")

df["date"] = pd.to_datetime(df["date"])

# seasonality
df["month"] = df["date"].dt.month

def get_season(month):
    """
    Maps a numerical month to its corresponding season.
    Parameters:
    month (int): The month as an integer (1-12).
    Returns:
    str: The season corresponding to the month ("winter", "spring", "summer", "fall")
    """
    if month in [12, 1, 2]:
        return "winter"
    elif month in [3, 4, 5]:
        return "spring"
    elif month in [6, 7, 8]:
        return "summer"
    else:
        return "fall"

def prepare_data(df):
    """
   This includes:
    - Converting dates to datetime objects.
    - Extracting seasonal categories.
    - Creating lag features (previous month sales).
    - Calculating momentum (diff) and trend (12-month moving average) features.
    - Encoding categorical variables into dummies.
    """
df["season"] = df["month"].apply(get_season)

df["civic_diff1"] = df["civic_sales"].shift(1).diff()
df["civic_lag1"] = df["civic_sales"].shift(1)
df["civic_ma12"] = df["civic_sales"].shift(1).rolling(12).mean()

df = df.dropna().reset_index(drop=True)
df = pd.get_dummies(df, columns=["season"], drop_first=True)

y = df["civic_sales"]
X = df.drop(columns=["civic_sales", "date", "month"])

# train everything except last year of data, test on last year
X_train = X.iloc[:-12]
X_test = X.iloc[-12:]

y_train = y.iloc[:-12]
y_test = y.iloc[-12:]

date_train = df["date"].iloc[:-12]
date_test = df["date"].iloc[-12:]

# fit model
model = DecisionTreeRegressor(
    max_depth=5,
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

def plot_results():
    """
    Generates three plots including the full test/train plot, 
    the specific test window (last 12 months), and the visual structure of the decision tree.
    """
    plt.figure(figsize=(12, 6))


plt.plot(date_train, y_train,
         label="Train Actual",
         color="blue")

plt.plot(date_test, y_test,
         label="Test Actual",
         color="black")

plt.plot(date_train, y_train_pred,
         label="Train Prediction",
         linestyle="--",
         color="orange")

plt.plot(date_test, y_test_pred,
         label="Test Prediction",
         linestyle="--",
         color="red")

# vertical line for split
plt.axvline(x=date_test.iloc[0],
            color="purple",
            linestyle=":",
            label="Train/Test Split")

plt.xlabel("Date")
plt.ylabel("Civic Sales")
plt.title("Decision Tree Predictions with Train/Test Split")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Test period plot
plt.figure(figsize=(10, 5))

plt.plot(date_test, y_test,
         label="Actual",
         color="black",
         linewidth=2)

plt.plot(date_test, y_test_pred,
         label="Prediction",
         linestyle="--",
         color="red",
         linewidth=2)

plt.xlabel("Date")
plt.ylabel("Civic Sales")
plt.title("Decision Tree — Last 12 Months Only")

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
