import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

def add_lags(lags, X, y):
    """
    Add a specified number of lags of y to X.

    Args:
        lags (int): number of lags of y to add to X.
        X (np.array): array of features without lags of y.
        y (np.array): array for the dependent variable.

    Returns:
        np.array: array of features with specified number of lags of y.
    """
    new_X = X.copy()
    if lags < 1:
        return new_X
    for lag in range(1, lags + 1):
        new_lag = "civic_sales_L" + str(lag)
        new_X[new_lag] = y.shift(lag)
    return new_X

def get_season(month, season):
    """
    Says whether a month is in a season.

    Args:
        month (int): the given month.
        season (list): the months in the season.

    Returns:
        int: 1 if in the season, 0 otherwise.
    """
    if month in season:
        return 1
    else:
        return 0

if __name__ == "__main__":
    # Importing data
    df = pd.read_csv("data/combined_table.csv")
    
    # Adding seasonality dummies
    df["date"] = pd.to_datetime(df["date"])
    df["month"] = df["date"].dt.month
    df["winter"] = df["month"].apply(func = get_season,
                                     season = [12, 1, 2])
    df["spring"] = df["month"].apply(func = get_season,
                                     season = [3, 4, 5])
    df["summer"] = df["month"].apply(func = get_season,
                                     season = [6, 7, 8])
    # Setting up default X, y
    y = df["civic_sales"]
    X = df.drop(columns = ["date", "month", "civic_sales"], axis="columns")
    
    # Random split
    best_train_mse = None
    best_test_mse = None
    best_num_ylags = None
    best_model = None
    for lags in range(0, 5):
        # Adding lags to dataset
        X_lagged = add_lags(lags, X, y)
        new_df = pd.concat([X_lagged, y], axis=1).dropna()
        new_X = new_df.drop(columns="civic_sales")
        new_y = new_df["civic_sales"]
        
        # Random split: 0.75 train, 0.25 test
        X_train, X_test, y_train, y_test = train_test_split(new_X,
                                                            new_y,
                                                            random_state=42)
        
        # Fitting model
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        y_hat_train = model.predict(X_train)
        y_hat_test = model.predict(X_test)
        
        # Obtaining MSEs
        train_mse = mean_squared_error(y_train, y_hat_train)
        test_mse = mean_squared_error(y_test, y_hat_test)
        
        # Checking if this number of dependent lags performed better
        if best_test_mse == None or best_test_mse > test_mse:
            best_model = model
            best_train_mse = train_mse
            best_test_mse = test_mse
            best_num_ylags = lags
    
    # Getting RMSEs
    best_train_rmse = np.sqrt(best_train_mse)
    best_test_rmse = np.sqrt(best_test_mse)
    
    # Printing results
    print("Random split")
    print(f"{best_model.feature_names_in_=}")
    print(f"{best_model.feature_importances_=}")
    print(f"{best_train_mse=}")
    print(f"{best_test_mse=}")
    print(f"{best_train_rmse=}")
    print(f"{best_test_rmse=}")
    print(f"{best_num_ylags=}")
    
    # Future predictions
    best_train_mse = None
    best_test_mse = None
    best_num_ylags = None
    best_model = None
    for lags in range(0, 5):
        # Adding lags to dataset
        X_lagged = add_lags(lags, X, y)
        new_df = pd.concat([X_lagged, y], axis=1).dropna()
        new_X = new_df.drop(columns="civic_sales")
        new_y = new_df["civic_sales"]
        
        # Testing the prediction of the most recent 12 months
        X_train = new_X[:-12]
        y_train = new_y[:-12]
        X_test = new_X[-12:]
        y_test = new_y[-12:]
        
        # Fitting model
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        y_hat_train = model.predict(X_train)
        y_hat_test = model.predict(X_test)
        
        # Obtaining MSEs
        train_mse = mean_squared_error(y_train, y_hat_train)
        test_mse = mean_squared_error(y_test, y_hat_test)
        
        # Checking if this number of dependent lags performed better
        if best_test_mse == None or best_test_mse > test_mse:
            best_model = model
            best_train_mse = train_mse
            best_test_mse = test_mse
            best_num_ylags = lags
    
    # Getting RMSEs
    best_train_rmse = np.sqrt(best_train_mse)
    best_test_rmse = np.sqrt(best_test_mse)
    
    # Printing results
    print()
    print("Future prediction")
    print(f"{best_model.feature_names_in_=}")
    print(f"{best_model.feature_importances_=}")
    print(f"{best_train_mse=}")
    print(f"{best_test_mse=}")
    print(f"{best_train_rmse=}")
    print(f"{best_test_rmse=}")
    print(f"{best_num_ylags=}")
    