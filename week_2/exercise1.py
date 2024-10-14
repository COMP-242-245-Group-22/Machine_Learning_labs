import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.datasets import make_friedman2
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel
from sklearn.gaussian_process.kernels import Hyperparameter
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression


# Generate synthetic data
x1 = np.arange(0, 10, 0.1)#100*100 matrix
x2 = np.arange(0, 10, 0.1)#100*100 matrix
x1, x2 = np.meshgrid(x1, x2)
y = np.sin(x1) * np.cos(x2) + np.random.normal(scale=0.1, size=x1.shape)# add some random Gaussian Noise

# Flatten the arrays
x1 = x1.flatten()#1D array with 10000 elements
x2 = x2.flatten()#1D array with 10000 elements
y = y.flatten()#1D array with 10000 elements
X = np.vstack((x1, x2)).T#10000*2 matrix

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)#feature matrix, target vector

"""
Object 1: Implement a decision tree regression model using scikit-learn to predict the output of a 2D sinusoidal signal based on two input features.
"""

# Initialize and train the decision tree regressor
tree_regressor = DecisionTreeRegressor(max_depth=5, splitter='best', random_state=42)
tree_regressor.fit(X_train, y_train)

# Predict on the test set
y_pred = tree_regressor.predict(X_test)

# Plot the true vs predicted values
plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_pred, alpha=0.6, color='b')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r', linewidth=2)
plt.title('Actual vs Predicted (Decision Tree Regressor)')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')

try:
    plt.savefig('/Users/huangyuting/Machine Learning for Robotics/week_2/Lab_2/actual_vs_predicted_decision_tree.png')
    print(f"Figure saved successfully")
except Exception as e:
    print(f"Error saving figure: {e}")

plt.show()

# Visualizing the 2D signal and model predictions
# Reshape the test predictions back into a grid for visualization
x1_test = X_test[:, 0]  # the first column of data
x2_test = X_test[:, 1]  # the second column of data

plt.figure(figsize=(10, 6))

# Plot the ground truth (original y values)
plt.subplot(1, 2, 1)
plt.tricontourf(x1_test, x2_test, y_test, cmap='viridis')
plt.title('True 2D Sinusoidal Signal')
plt.colorbar()

# Plot the predicted values by the decision tree
plt.subplot(1, 2, 2)
plt.tricontourf(x1_test, x2_test, y_pred, cmap='viridis')
plt.title('Predicted by Decision Tree')
plt.colorbar()

plt.tight_layout()

try:
    plt.savefig('/Users/huangyuting/Machine Learning for Robotics/week_2/Lab_2/2d_sinunoidal_signal_decision_tree.png')
    print(f"Figure saved successfully")
except Exception as e:
    print(f"Error saving figure: {e}")

plt.show()

    # return y_pred

"""
Object 2: Check on the scikit manual what are the hyperparameter that you can tune for the decision tree regressor.
"""

# Access each hyperparameter
X, y = make_friedman2(n_samples=50, noise=0, random_state=0)
kernel = ConstantKernel(constant_value=1.0,
   constant_value_bounds=(0.0, 10.0))
for hyperparameter in kernel.hyperparameters:
   print(hyperparameter)

# Initialize polynomial regression model
poly = PolynomialFeatures(degree=3, include_bias=False)# A polynomial regressor with 'degrees=3' is recommended
poly_features = poly.fit_transform(X)
polynomial_regressor = LinearRegression(fit_intercept=True, copy_X=True, n_jobs=None, positive=False)

# Fit the model to the training data
polynomial_regressor.fit(X_train, y_train)

# Predict on the test set
y_pred_poly = polynomial_regressor.predict(X_test)

# Evaluate the model
mse_poly = mean_squared_error(y_test, y_pred_poly)
r2_poly = r2_score(y_test, y_pred_poly)

"""
Object3: Perform a brief analisys of the effect of max_depth and splitter on the test error.
Object4: Compare the decision tree's performance on the test set with a polynomial regression.
"""

print("\nComparison of Decision Tree vs Polynomial Regression:")

# Initialize an empty list to store results
results = []

# Perform analysis of the effect of max_depth and splitter on test error
for depth in [3, 5, 7, 10]:  # Different values of max_depth
    for split in ['best', 'random']:  # Different values of splitter
        # Initialize and train the decision tree regressor
        tree_regressor = DecisionTreeRegressor(max_depth=depth, splitter=split, random_state=42)
        tree_regressor.fit(X_train, y_train)
        
        # Predict on the test set
        y_pred = tree_regressor.predict(X_test)
        
        # Calculate the mean squared error and R^2 score
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Append the results to the list
        results.append((depth, split, mse, r2))

# Loop through the decision tree results
for result in results:
    depth, split, mse, r2 = result
    print(f"Decision Tree (max_depth={depth}, splitter={split}) - MSE: {mse:.4f}, R^2: {r2:.4f}")

# Print polynomial regression results separately
print(f"Polynomial Regression - MSE: {mse_poly:.4f}, R^2: {r2_poly:.4f}")

