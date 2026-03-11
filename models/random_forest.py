import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    df = pd.read_csv("data/combined_table.csv").set_index("date")
    y = df["civic_sales"].values
    X = df.drop(columns = ["civic_sales"], axis="columns")
    
    # Random split
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    model = RandomForestRegressor(n_estimators=100, random_state=42)

    model.fit(X_train, y_train)
    y_hat_train = model.predict(X_train)
    y_hat_test = model.predict(X_test)

    train_mse = mean_squared_error(y_train, y_hat_train)
    test_mse = mean_squared_error(y_test, y_hat_test)
    
    train_rmse = np.sqrt(train_mse)
    test_rmse = np.sqrt(test_mse)
    
    print("Random split")
    print(f"{model.feature_names_in_=}")
    print(f"{model.feature_importances_=}")
    print(f"{train_mse=}")
    print(f"{test_mse=}")
    print(f"{train_rmse=}")
    print(f"{test_rmse=}")
    
    # Future predictions
    X_train = X[:247]
    y_train = y[:247]
    X_test = X[247:]
    y_test = y[247:]
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)

    model.fit(X_train, y_train)
    y_hat_train = model.predict(X_train)
    y_hat_test = model.predict(X_test)

    train_mse = mean_squared_error(y_train, y_hat_train)
    test_mse = mean_squared_error(y_test, y_hat_test)
    
    train_rmse = np.sqrt(train_mse)
    test_rmse = np.sqrt(test_mse)
    
    print()
    print("Future predictions")
    print(f"{model.feature_names_in_=}")
    print(f"{model.feature_importances_=}")
    print(f"{train_mse=}")
    print(f"{test_mse=}")
    print(f"{train_rmse=}")
    print(f"{test_rmse=}")