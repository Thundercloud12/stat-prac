
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor, OLSInfluence
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

df = pd.read_csv("train.csv")

# 1. Select data
cols = ["Age", "Pclass", "SibSp"]
data = df[["Fare"] + cols].dropna()
print(data.describe())

sns.heatmap(data.corr(), annot=True)
plt.show()

X = data[cols]
y = data["Fare"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 2. Linear regression
model = sm.OLS(y_train, sm.add_constant(X_train)).fit()
print(model.summary())

coef = pd.DataFrame({
    "Coefficient": model.params,
    "Std Error": model.bse,
    "t": model.tvalues,
    "p": model.pvalues
})
print(coef)

# 3. Predictions and evaluation
pred = model.predict(sm.add_constant(X_test))

mse = mean_squared_error(y_test, pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, pred)
r2 = r2_score(y_test, pred)

print("Train R2:", model.rsquared)
print("Adjusted R2:", model.rsquared_adj)
print("Test R2:", r2)
print("MSE:", mse, "RMSE:", rmse, "MAE:", mae)



# 4. VIF
vif = pd.DataFrame({
    "Feature": cols,
    "VIF": [variance_inflation_factor(
        sm.add_constant(X_train).values, i + 1
    ) for i in range(len(cols))]
})
print(vif)

res = model.resid

# 6. Cook's Distance
influence = OLSInfluence(model)
cooks = influence.cooks_distance[0]
limit = 4 / len(X_train)

plt.stem(range(len(cooks)), cooks)
plt.axhline(limit, color="red")
plt.show()

diagnostics = pd.DataFrame({
    "Actual": y_train,
    "Fitted": model.fittedvalues,
    "Residual": res,
    "Std Residual": influence.resid_studentized_internal,
    "Cooks D": cooks
})

print(diagnostics[diagnostics["Cooks D"] > limit].head(10))

# 7. Summary
summary = pd.DataFrame({
    "Metric": ["Train R2", "Adjusted R2", "Test R2",
               "MSE", "RMSE", "MAE", "Max VIF", "Influential"],
    "Value": [model.rsquared, model.rsquared_adj, r2,
              mse, rmse, mae, vif["VIF"].max(), (cooks > limit).sum()]
})
print(summary)
