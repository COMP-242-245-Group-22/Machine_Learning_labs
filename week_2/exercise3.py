import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV

"""
Step 1: Dataset generation and data storage
"""
# Generate or reuse synthetic data
x1 = np.arange(0, 10, 0.1)
x2 = np.arange(0, 10, 0.1)
x1, x2 = np.meshgrid(x1, x2)
y = np.sin(x1) * np.cos(x2) + np.random.normal(scale=0.1, size=x1.shape)

# Flatten the arrays
x1 = x1.flatten()
x2 = x2.flatten()
y = y.flatten()
X = np.vstack((x1, x2)).T

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Data storage
# Lists to store parameters
depths = list(range(3, 16))
n_estimators = list(range(10, 51, 10))
# Dictionary to store MSE and R² scores for each depth and estimator count
mse_matrix = np.zeros((len(depths), len(n_estimators)))
r2_matrix = np.zeros((len(depths), len(n_estimators)))

"""
Step 2: Check on the scikit manual what are the hyperparameter that we can tune for random forest regressor.
"""
print(f"\nhyperparameters for random forest:{RandomForestRegressor().get_params()}\n")

"""
Step 3: Parameter tuning for random forest(max_depth and n_estimators).
"""

for i, depth in enumerate(depths): 
    for j, n_estimator in enumerate(n_estimators):
        # Initialize and train the random forest regressor
        random_forest_tuning = RandomForestRegressor(max_depth=depth, n_estimators=n_estimator, random_state=42)
        random_forest_tuning.fit(X_train, y_train)
        
        # Predict on the test set
        y_pred_tuning = random_forest_tuning.predict(X_test)
        
        # Calculate MSE and R² scores
        mse = mean_squared_error(y_test, y_pred_tuning)
        r2 = r2_score(y_test, y_pred_tuning)
        
        # Store the results in matrices
        mse_matrix[i, j] = mse
        r2_matrix[i, j] = r2

"""
Step 4: Visualise data on random forest with different max depths and n_estimators(trees)
""" 
fig, ax = plt.subplots(1, 2, figsize=(14, 6))
# Initialize figure and axis
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Plot MSE for different max depths and n_estimators
for j, n_estimator in enumerate(n_estimators):
    mse_vals = mse_matrix[:, j]
    ax1.plot(depths, mse_vals, marker='o', linestyle='-', label=f'n_estimators={n_estimator}')
    
    # Annotate MSE points for each depth and n_estimator
    for i, depth in enumerate(depths):
        ax1.annotate(f'{mse_vals[i]:.2f}', (depth, mse_vals[i]), textcoords="offset points", xytext=(0, 5), ha='center', fontsize=8)

# MSE axis settings
ax1.set_xlabel('Max Depth')
ax1.set_ylabel('Mean Squared Error (MSE)')
ax1.set_title('MSE for Random Forest with Different Max Depths and n_estimators')
ax1.legend(title="n_estimators")

# Plot R² for different max depths and n_estimators
for j, n_estimator in enumerate(n_estimators):
    r2_vals = r2_matrix[:, j]
    ax2.plot(depths, r2_vals, marker='x', linestyle='--', label=f'n_estimators={n_estimator}')
    
    # Annotate R² points for each depth and n_estimator
    for i, depth in enumerate(depths):
        ax2.annotate(f'{r2_vals[i]:.2f}', (depth, r2_vals[i]), textcoords="offset points", xytext=(0, -10), ha='center', fontsize=8)

# R² axis settings
ax2.set_xlabel('Max Depth')
ax2.set_ylabel('R² Score')
ax2.set_title('R² Score for Random Forest with Different Max Depths and n_estimators')
ax2.legend(title="n_estimators")

# Display the plots
plt.suptitle("Random Forest Performance with different Max Depth and Number of Estimators")
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

# # Heatmap for MSE
# im1 = ax[0].imshow(mse_matrix, cmap="viridis", aspect="auto", origin="lower",
#                    extent=[min(n_estimators), max(n_estimators), min(depths), max(depths)])
# ax[0].set_xlabel("Number of Estimators")
# ax[0].set_ylabel("Max Depth")
# ax[0].set_title("Mean Squared Error (MSE)")
# fig.colorbar(im1, ax=ax[0], orientation="vertical")

# # Heatmap for R²
# im2 = ax[1].imshow(r2_matrix, cmap="viridis", aspect="auto", origin="lower",
#                    extent=[min(n_estimators), max(n_estimators), min(depths), max(depths)])
# ax[1].set_xlabel("Number of Estimators")
# ax[1].set_ylabel("Max Depth")
# ax[1].set_title("R² Score")
# fig.colorbar(im2, ax=ax[1], orientation="vertical")

# plt.tight_layout()

# try:
#     plt.savefig('/Users/huangyuting/Machine Learning for Robotics/week_2/Lab_2/Random Forest Performance with Varying Max Depth and Estimators.png')
#     print(f"Figure saved successfully")
# except Exception as e:
#     print(f"Error saving figure: {e}")
# plt.show()

random_forest_best = RandomForestRegressor(max_depth= 10 , n_estimators= 10 , random_state=42)
# Fit the model to the training data
random_forest_best.fit(X_train, y_train)

# Predict on the test set
y_pred_ran_for = random_forest_best.predict(X_test)

# Evaluate the model
mse_ran_for = mean_squared_error(y_test, y_pred_ran_for)
r2_ran_for = r2_score(y_test, y_pred_ran_for)

# Print evaluation metrics
print(f"Random Forest Regressor - Mean Squared Error (MSE): {mse_ran_for:.4f}")
print(f"Random Forest Regressor - R^2 Score: {r2_ran_for:.4f}")

# Plot true vs predicted values for Bagging Regressor
plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_pred_ran_for, alpha=0.6, color='b')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r', linewidth=2)
plt.title('Actual vs Predicted (Random Forest Regressor)')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')

try:
    plt.savefig('/Users/huangyuting/Machine Learning for Robotics/week_2/Lab_2/actual_vs_predicted_random_forest.png')
    print(f"Figure saved successfully")
except Exception as e:
    print(f"Error saving figure: {e}")

plt.show()

# Visualizing the 2D signal and model predictions
x1_test = X_test[:, 0]
x2_test = X_test[:, 1]

plt.figure(figsize=(10, 6))

# Plot the ground truth (original y values)
plt.subplot(1, 2, 1)
plt.tricontourf(x1_test, x2_test, y_test, cmap='viridis')
plt.title('True 2D Sinusoidal Signal')
plt.colorbar()

# Plot the predicted values by the Bagging Regressor
plt.subplot(1, 2, 2)
plt.tricontourf(x1_test, x2_test, y_pred_ran_for, cmap='viridis')
plt.title('Predicted by Random Forest Regressor')
plt.colorbar()

plt.tight_layout()
try:
    plt.savefig('/Users/huangyuting/Machine Learning for Robotics/week_2/Lab_2/2d_sinunoidal_signal_random_forest.png')
    print(f"Figure saved successfully")
except Exception as e:
    print(f"Error saving figure: {e}")

plt.show()
