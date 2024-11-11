import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV

"""
Step 1: Dataset generation and data storage
"""
# Generate synthetic data
x1 = np.arange(0, 10, 0.1)#100*100 matrix
x2 = np.arange(0, 10, 0.1)#100*100 matrix
x1, x2 = np.meshgrid(x1, x2)
y = np.sin(x1) * np.cos(x2) + np.random.normal(scale=0.1, size=x1.shape)# add some random Gaussian Noise

# Flatten the arrays
x1 = x1.flatten()#1D array with 10000 elements
x2 = x2.flatten()#1D array with 10000 elements
y = y.flatten()#1D array with 10000 elements
X = np.vstack((x1, x2)).T#2*10000 matrix

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)#feature matrix, target vector

# store data into list
depths = []
MSE_best = []
R2_best = []
MSE_random = []
R2_random = []

"""
Step 2: Check on the scikit manual what are the hyperparameter that we can tune for the decision tree regressor.
"""
print(f"Hyperparameters for decision tree regressor:{DecisionTreeRegressor().get_params()}")

"""
Step 3: Perform analysis of the effect of max_depth and splitter on test error
"""
for depth in [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]: 
    depths.append(depth)
    for split in ['best', 'random']: 
        # Initialize and train the decision tree regressor
        tree_regressor_tuning = DecisionTreeRegressor(max_depth=depth, splitter=split, random_state=42)
        tree_regressor_tuning.fit(X_train, y_train)

        # Predict on the test set
        y_pred_tuning = tree_regressor_tuning.predict(X_test)
        
        # Calculate the mean squared error and R^2 score
        mse = mean_squared_error(y_test, y_pred_tuning)
        r2 = r2_score(y_test, y_pred_tuning)
        
        # Append the results to the list
        if split == 'best':
            MSE_best.append(mse)
            R2_best.append(r2)
        elif split == 'random':
            MSE_random.append(mse)
            R2_random.append(r2)
"""
Step 4: Compare the decision tree's performance on the test set with a polynomial regression, and plot the results.
"""
# Initialise polynomial regression
poly = PolynomialFeatures(degree=3, include_bias=False)# A polynomial regressor with 'degree=3' is recommended
poly_features = poly.fit_transform(X)
polynomial_regressor = LinearRegression(fit_intercept=True, copy_X=True, n_jobs=None, positive=False)
# Fit the model to the training data
polynomial_regressor.fit(X_train, y_train)
# Predict on the test set
y_pred_poly = polynomial_regressor.predict(X_test)
# Evaluate the model
mse_poly = mean_squared_error(y_test, y_pred_poly)
r2_poly = r2_score(y_test, y_pred_poly)

# Initialize figure and axis
fig, ax1 = plt.subplots(figsize=(9, 5))

# Plot MSE for random and best splitters
ax1.set_xlabel('Max Depth / Degree')
ax1.set_ylabel('MSE')# Left axis for MSE
ax1.set_ylim([0, 0.3])
ax1.legend(loc='upper left')
ax1.plot(depths, MSE_best, 'bo-', label='MSE (splitter=best)')
ax1.plot(depths, MSE_random, 'd--', color='brown', label='MSE (splitter=random)')
ax1.plot([3], mse_poly, 'gs', label='MSE (polynomial)', markersize=8)
# Annotate MSE points for both 'best' and 'random'
for i, depth in enumerate(depths):
    # Annotating 'best' splitter MSE
    ax1.annotate(f'{MSE_best[i]:.2f}', (depth, MSE_best[i]), textcoords="offset points", xytext=(0, -10), ha='center', fontsize=8, color='blue')
    # Annotating 'random' splitter MSE
    ax1.annotate(f'{MSE_random[i]:.2f}', (depth, MSE_random[i]), textcoords="offset points", xytext=(0, 5), ha='center', fontsize=8, color='brown')
ax1.legend(loc='center left', bbox_to_anchor=(0.55, 0.6))

# Plot R² for random and best splitters
ax2 = ax1.twinx()
ax2.set_ylabel('R²')# Second y-axis for R²
ax2.set_ylim([0, 1])
ax2.legend(loc='upper right')
ax2.plot(depths, R2_best, 'rs-', label='R² (splitter=best)')
ax2.plot(depths, R2_random, 'k*--', label='R² (splitter=random)')
ax2.plot([3], r2_poly, 'ys', label='R² (polynomial)', markersize=8)
# Annotate R² points for both 'best' and 'random'
for i, depth in enumerate(depths):
    # Annotating 'best' splitter R²
    ax2.annotate(f'{R2_best[i]:.2f}', (depth, R2_best[i]), textcoords="offset points", xytext=(0, 5), ha='center', fontsize=8, color='red')
    # Annotating 'random' splitter R²
    ax2.annotate(f'{R2_random[i]:.2f}', (depth, R2_random[i]), textcoords="offset points", xytext=(0, -10), ha='center', fontsize=8, color='black')
ax2.legend(loc='upper left', bbox_to_anchor=(0.55, 0.45))

# Mark MSE and R² for polynomial regression
# MSE annotation for polynomial regression on ax1
ax1.annotate(f'{mse_poly:.2f}', (3, mse_poly), textcoords="offset points", xytext=(0, 10), ha='center', fontsize=8, color='green')
# R² annotation for polynomial regression on ax2
ax2.annotate(f'{r2_poly:.2f}', (3, r2_poly), textcoords="offset points", xytext=(0, -10), ha='center', fontsize=8, color='yellow')

plt.title('Comparison between Decision Tree and Polynomial Regression')

try:
    plt.savefig('/Users/huangyuting/Machine Learning for Robotics/week_2/Lab_2/decision_tree_and_polynomial_regression.png')
    print(f"Figure saved successfully")
except Exception as e:
    print(f"Error saving figure: {e}")
plt.show()

"""
Step 5: Apply the best max_depth and splitter on decision tree regression using scikit-learn to predict the output of a 2D sinusoidal signal based on two input features.
"""
# Initialize and train the decision tree regressor
# Observation needed to determine the best parameters used for decision tree regression that strikes a balance between underfitting and overfitting issues
tree_regressor_best = DecisionTreeRegressor(max_depth=10, splitter='best', random_state=42)
tree_regressor_best.fit(X_train, y_train)

# Predict on the test set
y_pred_best = tree_regressor_best.predict(X_test)

# Print evaluation metrics
MSE_decision_tree = mean_squared_error(y_test, y_pred_best)
R2_decision_tree = r2_score(y_test, y_pred_best)
print(f"MSE for best Decision Tree Regressor : {MSE_decision_tree:.4f}")
print(f"R^2 Score for best Decision Tree Regressor: {R2_decision_tree:.4f}")
# Plot the true vs predicted values
plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_pred_best, alpha=0.6, color='b')
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
plt.tricontourf(x1_test, x2_test, y_pred_best, cmap='viridis')
plt.title('Predicted by Decision Tree')
plt.colorbar()

plt.tight_layout()

try:
    plt.savefig('/Users/huangyuting/Machine Learning for Robotics/week_2/Lab_2/2d_sinunoidal_signal_decision_tree.png')
    print(f"Figure saved successfully")
except Exception as e:
    print(f"Error saving figure: {e}")

plt.show()
