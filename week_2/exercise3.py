import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Generate or reuse synthetic data
# Generate synthetic data
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

# Initialize and train the Random Forest regressor
random_forest = RandomForestRegressor(n_estimators=50, random_state=42)

# Fit the model to the training data
random_forest.fit(X_train, y_train)

# Predict on the test set
y_pred_ran_for = random_forest.predict(X_test)

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
    plt.savefig('/Users/huangyuting/Machine Learning for Robotics/week_2/Lab_2/2d_sinunoidal_signal_random_forest.png')
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
    plt.savefig('/Users/huangyuting/Machine Learning for Robotics/week_2/Lab_2/actual_vs_predicted_random_forest.png')
    print(f"Figure saved successfully")
except Exception as e:
    print(f"Error saving figure: {e}")

plt.show()
