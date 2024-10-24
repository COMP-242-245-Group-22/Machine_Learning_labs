import os

import matplotlib.pyplot as plt
import numpy as np
from simulation_and_control import (
    MotorCommands,
    PinWrapper,
    SinusoidalReference,
    feedback_lin_ctrl,
    pb,
)
from sklearn.metrics import r2_score


def main():
    # Configuration for the simulation
    conf_file_name = "pandaconfig.json"  # Configuration file for the robot
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    sim = pb.SimInterface(
        conf_file_name, conf_file_path_ext=cur_dir, use_gui=False
    )  # Initialize simulation interface

    # Get active joint names from the simulation
    ext_names = sim.getNameActiveJoints()
    ext_names = np.expand_dims(
        np.array(ext_names), axis=0
    )  # Adjust the shape for compatibility

    source_names = ["pybullet"]  # Define the source for dynamic modeling

    # Create a dynamic model of the robot
    dyn_model = PinWrapper(
        conf_file_name, "pybullet", ext_names, source_names, False, 0, cur_dir
    )
    num_joints = dyn_model.getNumberofActuatedJoints()
    noise_level = dyn_model.GetConfigurationVariable("robot_noise")[0][
        "joint_cov"
    ]

    # Print initial joint angles
    print(f"Initial joint angles: {sim.GetInitMotorAngles()}")

    # Sinusoidal reference
    # Specify different amplitude values for each joint
    amplitudes = [
        np.pi / 4,
        np.pi / 6,
        np.pi / 4,
        np.pi / 4,
        np.pi / 4,
        np.pi / 4,
        np.pi / 4,
    ]  # Example amplitudes for joints
    # Specify different frequency values for each joint
    frequencies = [
        0.4,
        0.5,
        0.4,
        0.4,
        0.4,
        0.4,
        0.4,
    ]  # Example frequencies for joints

    # Convert lists to NumPy arrays for easier manipulation in computations
    amplitude = np.array(amplitudes)
    frequency = np.array(frequencies)
    ref = SinusoidalReference(
        amplitude, frequency, sim.GetInitMotorAngles()
    )  # Initialize the reference

    # Simulation parameters
    time_step = sim.GetTimeStep()
    current_time = 0
    max_time = 10  # seconds

    # Command and control loop
    cmd = MotorCommands()  # Initialize command structure for motors
    # PD controller gains
    kp = 1000
    kd = 100

    # Initialize data storage
    tau_mes_all = []
    regressor_all = []

    # Data collection loop
    while current_time < max_time:
        # Measure current state
        q_mes = sim.GetMotorAngles(0)
        qd_mes = sim.GetMotorVelocities(0)
        qdd_mes = sim.ComputeMotorAccelerationTMinusOne(0)

        # Compute sinusoidal reference trajectory
        q_d, qd_d = ref.get_values(
            current_time
        )  # Desired position and velocity

        # Control command
        cmd.tau_cmd = feedback_lin_ctrl(
            dyn_model, q_mes, qd_mes, q_d, qd_d, kp, kd
        )
        sim.Step(cmd, "torque")

        # Get measured torque
        tau_mes = sim.GetMotorTorques(0)

        if dyn_model.visualizer:
            for index in range(
                len(sim.bot)
            ):  # Conditionally display the robot model
                q = sim.GetMotorAngles(index)
                dyn_model.DisplayModel(
                    q
                )  # Update the display of the robot model

        # Exit logic with 'q' key
        keys = sim.GetPyBulletClient().getKeyboardEvents()
        qKey = ord("q")
        if (
            qKey in keys
            and keys[qKey]
            and sim.GetPyBulletClient().KEY_WAS_TRIGGERED
        ):
            break

        # Compute regressor and store it
        regressor_all.append(
            dyn_model.ComputeDynamicRegressor(q_mes, qd_mes, qdd_mes)
        )
        tau_mes_all.append(tau_mes)

        current_time += time_step
        # Optional: print current time
        # print(f"Current time in seconds: {current_time:.2f}")

    # After data collection, stack all the regressor and all the torques and compute the parameters 'a' using pseudoinverse
    cutoff = 500
    regressor_all = np.vstack(regressor_all[1 + cutoff :])  # 70000x70
    tau_mes_all = np.hstack(tau_mes_all[1 + cutoff :])  # 70000x1
    n, p = regressor_all.shape  # 70000, 70
    timestamps = np.arange(0, n // num_joints) + cutoff  # 10000

    # compute dynamic parameters
    a = np.linalg.pinv(regressor_all) @ tau_mes_all  # 70x1
    tau_pred = regressor_all @ a  # 70000x1

    # compute the metrics for the linear model
    tss = np.sum((tau_mes_all - np.mean(tau_mes_all)) ** 2)
    rss = np.sum((tau_mes_all - tau_pred) ** 2)
    mse = rss / (n - 1)
    f_stat = ((tss - rss) / p) / (rss / (n - p - 1))
    r2 = r2_score(tau_mes_all, tau_pred)
    adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
    print(f"MSE: {mse}")
    print(f"Adjusted R-squared: {adjusted_r2}")
    print(f"F-statistics: {f_stat}")

    # confidence intervals
    var = np.diag(mse * np.linalg.pinv(regressor_all.T @ regressor_all))  # 70x1
    se = np.sqrt(np.abs(var))  # 70x1
    plt.errorbar(np.arange(p), a, yerr=se * 1.96, fmt="o", capsize=5)
    plt.xlabel("Index")
    plt.ylabel("Value")
    plt.title(
        f"Dynamic Parameters with 95% Confidence Intervals (noise = {noise_level})"
    )
    [
        plt.savefig(
            f"./Machine_Learning_labs/week_1_2/{folder}/confidence_intervals_{noise_level}{ext}",
            dpi=300,
        )
        for folder in ["figures", "67080d899e6574a2ca0083fa/figures"]
        for ext in [".png", ".pdf"]
    ]
    plt.close()

    # prediction intervals
    pred_se = []
    for i in range(len(timestamps)):
        x0 = tau_mes_all[i * num_joints : (i + 1) * num_joints].reshape(
            -1, 1
        )  # 7x1
        X = regressor_all[i * num_joints : (i + 1) * num_joints]  # 7x70
        inv = np.linalg.inv(X @ X.T)  # 7x7
        pred_se.append(np.sqrt(mse * (1 + x0.T @ inv @ x0)))  # 1x1
    pred_se = np.vstack(pred_se)  # 10000x1
    pred_intervals = (
        tau_pred.reshape(-1, num_joints) - 1.96 * pred_se,
        tau_pred.reshape(-1, num_joints) + 1.96 * pred_se,
    )

    # comparing predicted and measured torques for each joint
    for i in range(num_joints):
        plt.figure(figsize=(10, 2.5))
        plt.plot(timestamps, tau_mes_all[i::num_joints])
        plt.plot(timestamps, tau_pred[i::num_joints])
        plt.fill_between(
            timestamps,
            pred_intervals[0][:, i],
            pred_intervals[1][:, i],
            color="gray",
            alpha=0.5,
        )
        plt.xlim(left=0)
        plt.legend(
            ["Measured", "Predicted", "95% Pred Int"],
            loc="right",
            framealpha=0.5,
        )
        plt.xlabel("Time (ms)")
        plt.ylabel("Torque (Nm)")
        plt.title(
            rf"Joint {i} Measured and Predicted Torques ($t \geq t_{{{cutoff}}}$, noise = {noise_level})"
        )
        plt.tight_layout()
        [
            plt.savefig(
                f"./Machine_Learning_labs/week_1_2/{folder}/joint_{i}_torque_comparison_{noise_level}{ext}",
                dpi=300,
            )
            for folder in ["figures", "67080d899e6574a2ca0083fa/figures"]
            for ext in [".png", ".pdf"]
        ]
        plt.close()

    # plot the torque prediction error for each joint (optional)
    pred_err = tau_mes_all - tau_pred
    pred_err = np.reshape(pred_err, (n // num_joints, num_joints))

    for i in range(num_joints):
        plt.figure(figsize=(10, 2.5))
        plt.plot(timestamps, pred_err[:, i])
        plt.xlim(left=0)
        plt.xlabel("Time (ms)")
        plt.ylabel("Torque error (Nm)")
        plt.title(
            rf"Joint {i} Torque Prediction Error ($t \geq t_{{{cutoff}}}$, noise = {noise_level})"
        )
        plt.tight_layout()
        [
            plt.savefig(
                f"./Machine_Learning_labs/week_1_2/{folder}/joint_{i}_torque_prediction_error_{noise_level}{ext}",
                dpi=300,
            )
            for folder in ["figures", "67080d899e6574a2ca0083fa/figures"]
            for ext in [".png", ".pdf"]
        ]
        plt.close()


if __name__ == "__main__":
    main()
