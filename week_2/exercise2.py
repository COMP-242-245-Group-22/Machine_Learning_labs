import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.metrics import mean_squared_error, r2_score

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

# store data into list
depths = []
MSE_best = []
R2_best = []
MSE_random = []
R2_random = []

for depth in [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]:  # Different values of max_depth
    depths.append(depth)
    for split in ['best', 'random']: 
        # Initialize bagging regressor for tuning and fit the data 
        base_estimator = DecisionTreeRegressor(max_depth=depth, splitter=split, random_state=42)
        bagging_regressor_tuning = BaggingRegressor(estimator=base_estimator, random_state=42)
        bagging_regressor_tuning.fit(X_train, y_train)
        
        # Predict on the test set
        y_pred_tuning = bagging_regressor_tuning.predict(X_test)
        
        # Calculate the mean squared error and R^2 score for 'best' splitter
        mse = mean_squared_error(y_test, y_pred_tuning)
        r2 = r2_score(y_test, y_pred_tuning)

        # Append the results to the list
        if split == 'best':
            MSE_best.append(mse)
            R2_best.append(r2)
        elif split == 'random':
            MSE_random.append(mse)
            R2_random.append(r2)

# Plot the results
fig, ax1 = plt.subplots()

# Plot MSE for 'best' and 'random'
ax1.plot(depths, MSE_best, 'bo-', label='MSE (splitter=best)')
ax1.plot(depths, MSE_random, 'd--', color='brown', label='MSE (splitter=random)')
ax1.set_xlabel('Max Depth')
ax1.set_ylabel('MSE')
ax1.set_ylim([0, 1])

# Annotate MSE points for both 'best' and 'random'
for i, depth in enumerate(depths):
    # Annotating 'best' splitter MSE
    ax1.annotate(f'{MSE_best[i]:.2f}', (depth, MSE_best[i]), textcoords="offset points", xytext=(0, -10), ha='center', fontsize=8, color='blue')
    # Annotating 'random' splitter MSE
    ax1.annotate(f'{MSE_random[i]:.2f}', (depth, MSE_random[i]), textcoords="offset points", xytext=(0, 5), ha='center', fontsize=8, color='brown')

ax1.legend(loc='center left', bbox_to_anchor=(0.55, 0.6))

# Create a second y-axis for R²
ax2 = ax1.twinx()
ax2.plot(depths, R2_best, 'rs-', label='R² (splitter=best)')
ax2.plot(depths, R2_random, 'k*--', label='R² (splitter=random)')
ax2.set_ylabel('R²')
ax2.set_ylim([0, 1])

# Annotate R² points for both 'best' and 'random'
for i, depth in enumerate(depths):
    # Annotating 'best' splitter R²
    ax2.annotate(f'{R2_best[i]:.2f}', (depth, R2_best[i]), textcoords="offset points", xytext=(0, 5), ha='center', fontsize=8, color='red')
    # Annotating 'random' splitter R²
    ax2.annotate(f'{R2_random[i]:.2f}', (depth, R2_random[i]), textcoords="offset points", xytext=(0, -10), ha='center', fontsize=8, color='black')

ax2.legend(loc='center left', bbox_to_anchor=(0.55, 0.45))

# Add the title and show the plot
plt.title('MSE and R² for Bagging Regressor with Different Max Depths')

try:
    plt.savefig('/Users/huangyuting/Machine Learning for Robotics/week_2/Lab_2/MSE and R² for Bagging Regressor with Different Max Depths.png')
    print(f"Figure saved successfully")
except Exception as e:
    print(f"Error saving figure: {e}")

plt.show()
"""
Step 3: Choose the best max_depth and splitter for bagging regression, then visualise the data.
"""
# Initialize a Bagging Regressor with Decision Tree as the base estimator
bagging_regressor = BaggingRegressor(estimator=DecisionTreeRegressor(max_depth=10), n_estimators=50,random_state=42)

bagging_regressor.fit(X_train, y_train)

# Predict on the test set
y_pred_best = bagging_regressor.predict(X_test)

# Evaluate the model
mse_bagging = mean_squared_error(y_test, y_pred_best)
r2_bagging = r2_score(y_test, y_pred_best)

# Print evaluation metrics
print(f"MSE for best Bagging Regressor : {mse_bagging:.4f}")
print(f"R^2 Score for best Bagging Regressor: {r2_bagging:.4f}")

# Plot true vs predicted values for Bagging Regressor
plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_pred_best, alpha=0.6, color='b')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r', linewidth=2)
plt.title('Actual vs Predicted (Bagging Regressor)')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')

try:
    plt.savefig('/Users/huangyuting/Machine Learning for Robotics/week_2/Lab_2/actual_vs_predicted_bagging_regressor.png')
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
plt.tricontourf(x1_test, x2_test, y_pred_best, cmap='viridis')
plt.title('Predicted by Bagging Regressor')
plt.colorbar()

plt.tight_layout()

try:
    plt.savefig('/Users/huangyuting/Machine Learning for Robotics/week_2/Lab_2/2d_sinunoidal_signal_bagging_regressor.png')
    print(f"Figure saved successfully")
except Exception as e:
    print(f"Error saving figure: {e}")
plt.show()
