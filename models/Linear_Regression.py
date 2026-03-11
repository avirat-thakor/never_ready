import pandas as pd
import statsmodels.formula.api as smf

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as mse

df = pd.read_csv('data/combined_table.csv')

### Base Model
linear_model = smf.ols('civic_sales ~ corolla_sales + sentra_sales + cpi + fedfunds + gas + unemploy + csi + tdsp', data=df)
lin_res = linear_model.fit()
print(lin_res.summary())


### MSE and Training
y = df['civic_sales']
X = df.drop(columns=['civic_sales', 'date'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

train_mse = mse(y_train, y_train_pred)
test_mse = mse(y_test, y_test_pred)

print(f'Train MSE: {train_mse}')
print(f'Test MSE: {test_mse}')



### May be worth adding RESET test on individual or combinations of variables

###
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