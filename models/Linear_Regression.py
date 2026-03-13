import pandas as pd
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as mse


df = pd.read_csv('data/combined_table.csv')
df['date'] = pd.to_datetime(df['date'])

output_dir = "visualization/linear_regression"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

### Base Model
linear_model = smf.ols('civic_sales ~ corolla_sales + sentra_sales + cpi + fedfunds + gas + unemploy + csi + tdsp', data=df)
lin_res = linear_model.fit()
print(lin_res.summary())


### MSE and Training
y = df['civic_sales']
X = df.drop(columns=['civic_sales', 'date'])

X_train = X.iloc[:-12]
X_test = X.iloc[-12:]

y_train = y.iloc[:-12]
y_test = y.iloc[-12:]

model = LinearRegression()
model.fit(X_train, y_train)

y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

train_mse = mse(y_train, y_train_pred)
test_mse = mse(y_test, y_test_pred)

root_train = train_mse ** 0.5
root_test = test_mse ** 0.5

R_Squared = lin_res.rsquared

print(f'Train MSE: {train_mse}')
print(f'Test MSE: {test_mse}')
print(f'Root Train MSE: {root_train}')
print(f'Root Test MSE: {root_test}') 
print(f'R-squared: {R_Squared}')


"""
Note: The model by default produced some interesting results that will need attending to.
- The R-squared was roughly 0.674, which was surprisingly high given we didn't think it would
fit the data well.
- The coefficients for the corolla sales and sentra sales were both positive, which contradicts
our assumtion that they were substitutes for civic sales. 
- There's clearly issues with the data. The Durbin-Watson statistic was 0.8, and running 
the regression includes a note highlighting strong multicollinearity. 
- Will need further developing.
"""


### Plotting the actual vs predicted values for the test set, 
### using the 12 omitted data points as the test set.

plt.figure(figsize=(10, 6))
plt.plot(df['date'].iloc[-12:], y_test, label='Actual', color='blue')
plt.plot(df['date'].iloc[-12:], y_test_pred, label='Predicted', color='red')
plt.xlabel('Date')
plt.ylabel('Civic Sales')
plt.title('Linear Regression: Actual vs Predicted')
plt.legend()
plt.tight_layout()

plt.savefig(f"{output_dir}/linear_regression_plot.png", dpi=300, bbox_inches="tight")
plt.close()


### This part will now generate a plot of the actual vs predicted values across the 
### entire dataset

plt.figure(figsize=(10, 6))
plt.plot(df['date'], y, label='Actual', color='blue')
plt.plot(df['date'], model.predict(X), label='Predicted', color='red', linestyle='--')

plt.xlabel('Date')
plt.ylabel('Civic Sales')
plt.title('Linear Regression: Actual vs Predicted (Full Dataset)')
plt.legend()

ax = plt.gca()
ax.xaxis.set_major_locator(mdates.YearLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(f"{output_dir}/linear_regression_full_plot.png", dpi=300, bbox_inches="tight")
plt.close()

