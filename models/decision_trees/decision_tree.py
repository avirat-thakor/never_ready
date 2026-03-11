import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.metrics import mean_squared_error

# load data
df = pd.read_csv("data/combined_table.csv")

# convert date from string to datetime
df["date"] = pd.to_datetime(df["date"])

# target
y = df["civic_sales"]

# prediction variables
X = df.drop(columns=["civic_sales", "date"])

# 75/25 split
split = int(len(df) * 0.75)

X_train = X.iloc[:split]
X_test = X.iloc[split:]

y_train = y.iloc[:split]
y_test = y.iloc[split:]

date_train = df["date"].iloc[:split]
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

# plot actual vs predictions
plt.figure(figsize=(12, 6))
plt.plot(date_test, y_test, label="Actual Civic Sales")
plt.plot(date_test, y_test_pred, label="Predicted Civic Sales")
plt.xlabel("Date")
plt.ylabel("Civic Sales")
plt.title("Actual vs Predicted Civic Sales (Test Period)")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# plot decision tree
plt.figure(figsize=(18, 10))
plot_tree(
    model,
    feature_names=X.columns,
    filled=True,
    rounded=True
)
plt.show()