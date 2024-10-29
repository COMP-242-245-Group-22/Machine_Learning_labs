from sklearn.metrics import mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor, BaggingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.datasets import fetch_california_housing
import numpy as np

# Synthetic data generation
def generate_synthetic_data():
    x1 = np.arange(0, 10, 0.1)
    x2 = np.arange(0, 10, 0.1)
    x1, x2 = np.meshgrid(x1, x2)
    y = np.sin(x1) * np.cos(x2) + np.random.normal(scale=0.1, size=x1.shape)
    x1, x2, y = x1.flatten(), x2.flatten(), y.flatten()
    X = np.vstack((x1, x2)).T
    return train_test_split(X, y, test_size=0.3, random_state=42)

# Perform grid search for tuning parameters for each model
def perform_grid_search(model, param_grid, X_train, y_train):
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=5,
        scoring='neg_mean_squared_error',
        n_jobs=-1
    )
    grid_search.fit(X_train, y_train)
    return grid_search.best_params_, grid_search.best_estimator_

# Evaluate model performance using mse and r2
def evaluate_model(model, X_test, y_test, model_name="Model"):
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"{model_name} - MSE: {mse:.4f}, R²: {r2:.4f}")
    return mse, r2

# Polynomial Regression evaluation
def evaluate_polynomial_regression(X_train, X_test, y_train, y_test, degree=3):
    poly = PolynomialFeatures(degree=degree)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)
    model = LinearRegression().fit(X_train_poly, y_train)
    mse, r2 = evaluate_model(model, X_test_poly, y_test, "Polynomial Regression")
    return model, mse, r2

# Define parameter grids
param_grids = {
    "DecisionTree": {
        'max_depth': [None, 5, 10, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2', None]
    },
    "BaggingRegressor": {
        'n_estimators': [10, 50, 100],
        'max_samples': [0.5, 0.7, 1.0],
        'max_features': [0.5, 0.7, 1.0],
        'estimator': [DecisionTreeRegressor(max_depth=3), DecisionTreeRegressor(max_depth=5)]
    },
    "RandomForest": {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['auto', 'sqrt', 'log2']
    },
    "AdaBoost": {
        'n_estimators': [50, 100, 200, 300],
        'learning_rate': [0.001, 0.01, 0.1, 0.5, 1.0],
        'loss': ['linear', 'square', 'exponential'],
        'estimator': [DecisionTreeRegressor(max_depth=3), DecisionTreeRegressor(max_depth=5)]
    }
}

# Initialize models
model_constructors = {
    "DecisionTree": DecisionTreeRegressor(random_state=42),
    "BaggingRegressor": BaggingRegressor(random_state=42),
    "RandomForest": RandomForestRegressor(random_state=42),
    "AdaBoost": AdaBoostRegressor()
}

# Perform model comparisons
def compare_models(X_train, X_test, y_train, y_test):
    best_model = None
    lowest_mse = float('inf')
    best_model_name = ""

    for model_name, model in model_constructors.items():
        print(f"\nPerforming grid search for {model_name}...")
        param_grid = param_grids[model_name]
        best_params, trained_model = perform_grid_search(model, param_grid, X_train, y_train)
        print(f"Best parameters for {model_name}: {best_params}")
        
        mse, r2 = evaluate_model(trained_model, X_test, y_test, model_name)
        if mse < lowest_mse:
            best_model, lowest_mse, best_model_name = trained_model, mse, model_name

    # Evaluate Polynomial Regression separately
    print("\nEvaluating Polynomial Regression...")
    poly_model, mse_poly, r2_poly = evaluate_polynomial_regression(X_train, X_test, y_train, y_test)
    if mse_poly < lowest_mse:
        best_model, lowest_mse, best_model_name = poly_model, mse_poly, "Polynomial Regression"

    print(f"\nBest model based on lowest MSE: {best_model_name} with MSE: {lowest_mse:.4f}")
    return best_model, best_model_name

# Predict California housing prices with the best model
def predict_housing_prices(best_model, best_model_name):
    data = fetch_california_housing()
    X, y = data.data, data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    if best_model_name == "Polynomial Regression":
        poly = PolynomialFeatures(degree=3)
        X_train_poly = poly.fit_transform(X_train)
        X_test_poly = poly.transform(X_test)
        best_model.fit(X_train_poly, y_train)
        y_pred = best_model.predict(X_test_poly)
    else:
        best_model.fit(X_train, y_train)
        y_pred = best_model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"\nBest Model ({best_model_name}) on California Housing Dataset - MSE: {mse:.4f}, R²: {r2:.4f}")

# Run the comparison and use the best model on California Housing dataset
X_train, X_test, y_train, y_test = generate_synthetic_data()
best_model, best_model_name = compare_models(X_train, X_test, y_train, y_test)
predict_housing_prices(best_model, best_model_name)
