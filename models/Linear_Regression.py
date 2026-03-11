import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.diagnostic import linear_reset
from statsmodels.stats.anova import anova_lm


df = pd.read_csv('data/combined_table.csv')

### Base Model
linear_model = smf.ols('civic_sales ~ corolla_sales + sentra_sales + cpi + fedfunds + gas + unemploy + csi + tdsp', data=df)
lin_res = linear_model.fit()
print(lin_res.summary())

### May be worth adding RESET test on individual or combinations of variables

###