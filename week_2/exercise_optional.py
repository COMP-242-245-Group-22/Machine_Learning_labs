from sklearn.metrics import mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing, make_friedman2
import numpy as np
"""
Object1: Get the best model by comparing the fit results
"""
# Generate synthetic data
x1 = np.arange(0, 10, 0.1)
x2 = np.arange(0, 10, 0.1)
x1, x2 = np.meshgrid(x1, x2)
y = np.sin(x1) * np.cos(x2) + np.random.normal(scale=0.1, size=x1.shape)
x1 = x1.flatten()
x2 = x2.flatten()
y = y.flatten()
X = np.vstack((x1, x2)).T

# Split into train/test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Store results
results = []

# Decision Tree Regressor (Exercise 1)
tree_regressor = DecisionTreeRegressor(max_depth=5, random_state=42)
tree_regressor.fit(X_train, y_train)
y_pred_tree = tree_regressor.predict(X_test)
mse_tree = mean_squared_error(y_test, y_pred_tree)
r2_tree = r2_score(y_test, y_pred_tree)
results.append(('Decision Tree', mse_tree, r2_tree))

# Polynomial Regressor (Exercise 1)
poly = PolynomialFeatures(degree=3)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)
polynomial_regressor = LinearRegression()
polynomial_regressor.fit(X_train_poly, y_train)
y_pred_poly = polynomial_regressor.predict(X_test_poly)
mse_poly = mean_squared_error(y_test, y_pred_poly)
r2_poly = r2_score(y_test, y_pred_poly)
results.append(('Polynomial Regression', mse_poly, r2_poly))

# Bagging Regressor(Exercise 2)
bagging_regressor = BaggingRegressor(estimator=DecisionTreeRegressor(max_depth=5), n_estimators=50,random_state=42)
bagging_regressor.fit(X_train, y_train)
y_pred_bagging = bagging_regressor.predict(X_test)
mse_bagging = mean_squared_error(y_test, y_pred_bagging)
r2_bagging = r2_score(y_test, y_pred_bagging)
results.append(('Bagging Regression', mse_bagging, r2_bagging))

# Random Forest Regressor(Exercise 3)
random_forest = RandomForestRegressor(n_estimators=50, random_state=42)
random_forest.fit(X_train, y_train)
y_pred_ran_for = random_forest.predict(X_test)
mse_ran_for = mean_squared_error(y_test, y_pred_ran_for)
r2_ran_for = r2_score(y_test, y_pred_ran_for)
results.append(('Random Forest Regression', mse_ran_for, r2_ran_for))

# Ada Boost Regressor (Exercise 4)
ada_regressor = AdaBoostRegressor(estimator=DecisionTreeRegressor(max_depth=5), random_state=42)
ada_regressor.fit(X_train, y_train)
y_pred_ada = ada_regressor.predict(X_test)
mse_ada = mean_squared_error(y_test, y_pred_ada)
r2_ada = r2_score(y_test, y_pred_ada)
results.append(('Ada Boost Regressor', mse_ada, r2_ada))

# Compare results and find the best model
best_model = min(results, key=lambda x: x[1])  # The model with the lowest MSE

# Print comparison results
print("Comparison of Regression Models:")
for model_name, mse, r2 in results:
    print(f"{model_name} - MSE: {mse:.4f}, R²: {r2:.4f}")

# Print the best model
print(f"\nBest Model: {best_model[0]} with MSE: {best_model[1]:.4f} and R²: {best_model[2]:.4f}")
"""
Object2: Predict the housing price in California using the best model 
"""
data = fetch_california_housing()
X = data.data
y = data.target
feature_names = data.feature_names

# Split the dataset into training and testing sets

