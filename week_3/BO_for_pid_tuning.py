from skopt import gp_minimize
from skopt.space import Real
import numpy as np
import os
from matplotlib import pyplot as plt
from simulation_and_control import pb, MotorCommands, PinWrapper, feedback_lin_ctrl, SinusoidalReference, CartesianDiffKin
from skopt import gp_minimize
from skopt.space import Real
from skopt.learning import GaussianProcessRegressor
from skopt.learning.gaussian_process.kernels import RBF
from gp_functions import fit_gp_model_1d, plot_gp_results_1d,fit_gp_model,plot_gp_results


# Configuration for the simulation
conf_file_name = "pandaconfig.json"  # Configuration file for the robot
cur_dir = os.path.dirname(os.path.abspath(__file__))
sim = pb.SimInterface(conf_file_name, conf_file_path_ext = cur_dir, use_gui=False)  # Initialize simulation interface

# Get active joint names from the simulation
ext_names = sim.getNameActiveJoints()
ext_names = np.expand_dims(np.array(ext_names), axis=0)  # Adjust the shape for compatibility

source_names = ["pybullet"]  # Define the source for dynamic modeling

# Create a dynamic model of the robot
dyn_model = PinWrapper(conf_file_name, "pybullet", ext_names, source_names, False,0,cur_dir)
num_joints = dyn_model.getNumberofActuatedJoints()

init_joint_angles = sim.GetInitMotorAngles()

print(f"Initial joint angles: {init_joint_angles}")

# Sinusoidal reference
# Specify different amplitude values for each joint
amplitudes = [np.pi/4, np.pi/6, np.pi/4, np.pi/4, np.pi/4, np.pi/4, np.pi/4]  # Example amplitudes for joints
# Specify different frequency values for each joint
frequencies = [0.4, 0.5, 0.4, 0.4, 0.4, 0.4, 0.4]  # Example frequencies for joints

# Convert lists to NumPy arrays for easier manipulation in computations
amplitude = np.array(amplitudes)
frequency = np.array(frequencies)
ref = SinusoidalReference(amplitude, frequency, sim.GetInitMotorAngles())  # Initialize the reference


# Global lists to store data
kp0_values = []
kd0_values = []
tracking_errors = []

def simulate_with_given_pid_values(sim_, kp, kd, episode_duration=10):
    
    # here we reset the simulator each time we start a new test
    sim_.ResetPose()
    
    # IMPORTANT: to ensure that no side effect happens, we need to copy the initial joint angles
    q_des = init_joint_angles.copy()
    qd_des = np.array([0]*dyn_model.getNumberofActuatedJoints())

    time_step = sim_.GetTimeStep()
    current_time = 0
    # Command and control loop
    cmd = MotorCommands()  # Initialize command structure for motors

    # Initialize data storage
    q_mes_all, qd_mes_all, q_d_all, qd_d_all,  = [], [], [], []
    
    steps = int(episode_duration/time_step)
    # testing loop
    for i in range(steps):
        # measure current state
        q_mes = sim_.GetMotorAngles(0)
        qd_mes = sim_.GetMotorVelocities(0)
    
        # Compute sinusoidal reference trajectory
        q_des, qd_des = ref.get_values(current_time)  # Desired position and velocity
        # Control command
        tau_cmd = feedback_lin_ctrl(dyn_model, q_mes, qd_mes, q_des, qd_des, kp, kd)  # Zero torque command
        cmd.SetControlCmd(tau_cmd, ["torque"]*7)
        sim_.Step(cmd, "torque")  # Simulation step with torque command

        # Exit logic with 'q' key
        keys = sim_.GetPyBulletClient().getKeyboardEvents()
        qKey = ord('q')
        if qKey in keys and keys[qKey] and sim_.GetPyBulletClient().KEY_WAS_TRIGGERED:
            break
        
        #simulation_time = sim.GetTimeSinceReset()

        # Store data for plotting
        q_mes_all.append(q_mes)
        qd_mes_all.append(qd_mes)
        q_d_all.append(q_des)
        qd_d_all.append(qd_des)
        
        #time.sleep(0.01)  # Slow down the loop for better visualization
        # get real time
        current_time += time_step
       
    # Calculate tracking error
    q_mes_all = np.array(q_mes_all)
    q_des_all = np.array(q_d_all)

    tracking_error = np.sum((q_mes_all - q_des_all)**2)  # Sum of squared error as the objective
    # print tracking error
    print("Tracking error: ", tracking_error)
    # print PD gains
    print("kp: ", kp)
    print("kd: ", kd)
    
    return tracking_error


# Objective function for optimization
def objective(params):
    kp = np.array(params[:7])  # First 7 elements correspond to kp
    kd = np.array(params[7:])  # Last 7 elements correspond to kd
    episode_duration = 10
    
    # TODO Call the simulation with given kp and kd values
    tracking_error = simulate_with_given_pid_values(sim, kp, kd, episode_duration)

    # TODO Collect data for the first kp and kd  
    kp0_values.append(kp[0])
    kd0_values.append(kd[0])
    tracking_errors.append(tracking_error)
    return tracking_error


def main():
    # Define the search space for Kp and Kd
   # Define the search space as before
    space = [
        Real(0.1, 1000, name=f'kp{i}') for i in range(7)
    ] + [
        Real(0.0, 100, name=f'kd{i}') for i in range(7)
    ]

    rbf_kernel = RBF(
    length_scale=1.0,            # Initial length scale
    length_scale_bounds=(1e-2, 1e2)  # Bounds for length scale
    )

    gp = GaussianProcessRegressor(
    kernel=rbf_kernel,
    normalize_y=True,
    n_restarts_optimizer=10  # Optional for better hyperparameter optimization
    )

    # Test for each Aquisition Function
    acq_funcs = ['EI', 'PI', 'LCB']
    results = {}
    for acq_func in acq_funcs:
        n_calls=50
        print(f"\nRunning optimization with acquisition function: {acq_func}")
        
        # Perform Bayesian optimization with the chosen acquisition function
        result = gp_minimize(
            objective, 
            space, 
            n_calls=n_calls,  # Adjust the number of iterations if necessary
            base_estimator=gp,  # Use Custom Gaussian Process Regressor
            acq_func=acq_func,  # Choose acquisition function
            random_state=42
        )
        results[acq_func] = {
        'tracking_error': result.fun,
        'parameters': result.x,
        'iterations': result.func_vals  # Keeps track of objective values over iterations
    }
    # Plot convergence behavior
    plt.figure(figsize=(10, 6))
    for acq_func, data in results.items():
        plt.plot(data['iterations'], label=f'{acq_func} (final error: {data["tracking_error"]:.4f})')

    plt.xlabel('Iteration')
    plt.ylabel('Tracking Error')
    plt.title('Convergence of Acquisition Functions')
    plt.legend()
    # file_path = f'/Users/huangyuting/Machine Learning for Robotics/week_3/Lab_3/figures/objective2/convergence_of_acquisition_function.pdf'
    # try:
    #     plt.savefig(file_path)
    #     print(f'Convergence of Acquisition Functions')
    # except Exception as e:
    #     print(f'Error saving figure for Convergence of Acquisition Functions: {e}')
    plt.show()

    # Print final results for each acquisition function
    for acq_func, data in results.items():
        print(f"\nAcquisition Function: {acq_func}")
        print(f"Final Tracking Error: {data['tracking_error']}")
        print(f"Optimal Parameters (kp, kd): {data['parameters']}")
    # Analyzing results
    for acq_func, data in results.items():
        print(f"\nAcquisition Function: {acq_func}")
        print(f"Final Tracking Error: {data['tracking_error']}")
        print(f"Optimal Parameters (kp, kd): {data['parameters']}")
        print(f"Convergence Behavior: {data['iterations']}")
        
    # Prepare data
    kp0_values_array = np.array(kp0_values).reshape(-1, 1)
    kd0_values_array = np.array(kd0_values).reshape(-1, 1)
    tracking_errors_array = np.array(tracking_errors)

    # Fit GP models
    gp_kp0 = fit_gp_model_1d(kp0_values_array, tracking_errors_array)
    gp_kd0 = fit_gp_model_1d(kd0_values_array, tracking_errors_array)

    # Plot the results
    plot_gp_results_1d(gp_kp0, gp_kd0)

    # Define kernels with small and large length scales
    kernels = {
        "small": RBF(length_scale=1e-2, length_scale_bounds=(1e-3, 1e-1)),
        "large": RBF(length_scale=1e2, length_scale_bounds=(1e1, 1e4))
    }
    # Fit GP models with different length scales
    gp_models = fit_gp_model(kp0_values, kd0_values, tracking_errors, kernels)

    # Plot the GP results for both models (small and large kernels)
    for scale, (gp_kp0, gp_kd0) in gp_models.items():
        print(f"\nPlotting results for {scale} length scale kernel:")
        try:
            if scale == "small":
                plot_gp_results(kp0_values, kd0_values, tracking_errors, gp_kp0, gp_kd0)
                # plt.savefig(f'/Users/huangyuting/Machine Learning for Robotics/week_3/Lab_3/figures/objective3/re')
                print(f'gp small results saved successfully')
            elif scale == "large":
                plot_gp_results(kp0_values, kd0_values, tracking_errors, gp_kp0, gp_kd0)
                # plt.savefig(f'/Users/huangyuting/Machine Learning for Robotics/week_3/Lab_3/figures/objective3/plot_gp_large_results.pdf')
                print(f'gp large results saved successfully')
        except Exception as e:
            print(f'Error saving figure for plot_gp_{scale}_results: {e}')
    
    # Apply the best aquisition function with Ziegler Nichols
    best_acquisition_function = 'EI'
    result = gp_minimize(
    objective,      
    space,         
    n_calls=50,       
    acq_func=best_acquisition_function,  
    random_state=42
)
    # Extract the optimal values
    best_kp = result.x[:7]  # Optimal kp vector
    best_kd = result.x[7:]  # Optimal kd vector
    print(f"Best kp: {best_kp}, Best kd: {best_kd}")
    print(f"Minimum tracking error achieved: {result.fun}")

if __name__ == "__main__":
    main()
