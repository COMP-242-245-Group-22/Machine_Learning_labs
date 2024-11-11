
import numpy as np
import os
from matplotlib import pyplot as plt
from simulation_and_control import pb, MotorCommands, PinWrapper, feedback_lin_ctrl, SinusoidalReference, CartesianDiffKin
from skopt import gp_minimize
from skopt.space import Real
from skopt.learning import GaussianProcessRegressor
from skopt.learning.gaussian_process.kernels import RBF
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # For 3D plotting


def fit_gp_model_1d(X_values, y_values):
    # Define the kernel
    rbf_kernel = RBF(
    length_scale=1.0,            # Initial length scale
    length_scale_bounds=(1e-2, 1e2)  # Bounds for length scale
    )
    
    # Create and fit the GP regressor
    gp = GaussianProcessRegressor(
        kernel=rbf_kernel,
        normalize_y=True,
        n_restarts_optimizer=10,
        random_state=42
    )
    gp.fit(X_values, y_values)
    
    return gp

def fit_gp_model(kp0_values, kd0_values, tracking_errors, kernels):
    # Convert inputs to appropriate numpy arrays and reshape for fitting
    kp0_values = np.array(kp0_values).reshape(-1, 1)
    kd0_values = np.array(kd0_values).reshape(-1, 1)
    tracking_errors = np.array(tracking_errors)

    # Define two different RBF kernels for "small" and "large" length scales
    kernels = {"small": RBF(length_scale=1e-2), "large": RBF(length_scale=1e2)}

    gp_models = {}
    for scale, kernel in kernels.items():
        # Fit GP model for kp0 vs tracking error
        gp_kp0 = GaussianProcessRegressor(kernel=kernel, normalize_y=True)
        gp_kp0.fit(kp0_values, tracking_errors)
        
        # Fit GP model for kd0 vs tracking error
        gp_kd0 = GaussianProcessRegressor(kernel=kernel, normalize_y=True)
        gp_kd0.fit(kd0_values, tracking_errors)

        # Store the models for this scale
        gp_models[scale] = (gp_kp0, gp_kd0)

    return gp_models



def create_prediction_range_kp0():
    kp0_range = np.linspace(0.1, 1000, 200).reshape(-1, 1)
    return kp0_range

def create_prediction_range_kd0():
    kd0_range = np.linspace(0.0, 100, 200).reshape(-1, 1)
    return kd0_range

def plot_gp_results_1d(gp_kp0, gp_kd0):
    # Create prediction ranges
    kp0_pred = create_prediction_range_kp0()
    kd0_pred = create_prediction_range_kd0()
    
    # Predict for kp0
    y_mean_kp0, y_std_kp0 = gp_kp0.predict(kp0_pred, return_std=True)
    
    # Predict for kd0
    y_mean_kd0, y_std_kd0 = gp_kd0.predict(kd0_pred, return_std=True)
    

    # Plotting
    plt.figure(figsize=(14, 6))

    # Plot for kp0
    plt.subplot(1, 2, 1)
    plt.plot(kp0_pred, y_mean_kp0, 'k-', lw=1.5, zorder=9, label='Mean prediction')
    plt.fill_between(kp0_pred.ravel(), y_mean_kp0 - 1.96 * y_std_kp0, y_mean_kp0 + 1.96 * y_std_kp0,
                    alpha=0.5, fc='orange', ec='None', label='95% confidence interval')
    plt.title("Gaussian process regression on noise-free dataset for kp0")
    plt.xlabel('X')
    plt.ylabel('f(X)')
    plt.legend(loc='upper left')
    
    # Plot for kd0
    plt.subplot(1, 2, 2)
    plt.plot(kd0_pred, y_mean_kd0, 'k-', lw=1.5, zorder=9, label='Mean prediction')
    plt.fill_between(kd0_pred.ravel(), y_mean_kd0 - 1.96 * y_std_kd0, y_mean_kd0 + 1.96 * y_std_kd0,
                    alpha=0.5, fc='orange', ec='None', label='95% confidence interval')
    plt.title("Gaussian process regression on noise-free dataset for kd0")
    plt.xlabel('X')
    plt.ylabel('f(X)')
    plt.legend(loc='upper left')
    
    plt.tight_layout()
    # file_path = f'/Users/huangyuting/Machine Learning for Robotics/week_3/Lab_3/figures/objective1/plot_gp_results_1d.pdf'
    # try:
    #     plt.savefig(file_path)
    #     print(f'gp 1d results Size saved successfully')
    # except Exception as e:
    #     print(f'Error saving figure for plot_gp_results_1d: {e}')
    
    plt.show()

def plot_gp_results(kp0_values, kd0_values, tracking_errors, gp_kp0, gp_kd0):
    kp0_values = np.array(kp0_values).reshape(-1, 1)
    kd0_values = np.array(kd0_values).reshape(-1, 1)
    tracking_errors = np.array(tracking_errors)

   # Create prediction ranges
    kp0_pred = create_prediction_range_kp0()
    kd0_pred = create_prediction_range_kd0()

    # Predict for kp0
    y_mean_kp0, y_std_kp0 = gp_kp0.predict(kp0_pred, return_std=True)
    
    # Predict for kd0
    y_mean_kd0, y_std_kd0 = gp_kd0.predict(kd0_pred, return_std=True)

    # Plot for kp0
    plt.figure(figsize=(14, 6))
    
    plt.subplot(1, 2, 1)
    plt.fill_between(
       kp0_pred.ravel(),
       y_mean_kp0 - 1.96 * y_std_kp0,
       y_mean_kp0 + 1.96 * y_std_kp0,
       alpha=0.2,
       label='95% Confidence Interval'
    )
    plt.plot(kp0_pred, y_mean_kp0, 'k-', label='GP Mean Prediction')
    plt.scatter(kp0_values, tracking_errors, c='r', label='Data Points')
    plt.title('GP Regression for kp0')
    plt.xlabel('kp0')
    plt.ylabel('Tracking Error')
    plt.legend()
    
    # # Plot for kd0
    plt.subplot(1, 2, 2)

    plt.fill_between(
       kd0_pred.ravel(),
       y_mean_kd0 - 1.96 * y_std_kd0,
       y_mean_kd0 + 1.96 * y_std_kd0,
       alpha=0.2,
       label='95% Confidence Interval'
    )
    plt.plot(kd0_pred, y_mean_kd0, 'k-', label='GP Mean Prediction')
    plt.scatter(kd0_values, tracking_errors, c='r', label='Data Points')
    plt.title('GP Regression for kd0')
    plt.xlabel('kd0')
    plt.ylabel('Tracking Error')
    plt.legend()
    
    plt.tight_layout()
    # file_path = f'/Users/huangyuting/Machine Learning for Robotics/week_3/Lab_3/figures/objective1/plot_gp_results_1d.pdf'
    # try:
    #     plt.savefig(file_path)
    #     print(f'gp 1d results Size saved successfully')
    # except Exception as e:
    #     print(f'Error saving figure for plot_gp_results_1d: {e}')
    plt.show()
