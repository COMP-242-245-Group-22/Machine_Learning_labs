import os
import pickle
import time

import joblib  # For saving and loading models
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from simulation_and_control import (
    MotorCommands,
    PinWrapper,
    feedback_lin_ctrl,
    pb,
)

TASK = 3
CUR_DIR = os.path.dirname(os.path.realpath(__file__))  # current directory
DIR = os.path.join(CUR_DIR, "figures", f"task{TASK}")  # figure directory
EXT = "pdf"  # figure extension
os.makedirs(DIR, exist_ok=True)  # create figure directory if not exist
print(f"Performing Task {TASK}...")

model_types = ["neural_network", "random_forest", "smoothed_random_forest"]

DEPTH = 2  # Random Forest depth
CUTOFF = 100

SEED = 42  # random seed for reproducibility
np.random.seed(SEED)
torch.manual_seed(SEED)


# MLP Model Definition
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(
                4, 128
            ),  # Input layer to hidden layer (4 inputs: time + goal positions)
            nn.ReLU(),
            nn.Linear(128, 1),  # Hidden layer to output layer
        )

    def forward(self, x):
        return self.model(x)


def exponential_moving_average(data, alpha=0.01):
    ema_data = np.zeros_like(data)
    ema_data[0] = data[0]
    for t in range(CUTOFF + 1, len(data)):
        ema_data[t] = alpha * data[t] + (1 - alpha) * ema_data[t - 1]
    return ema_data


def main():
    # Load the saved data
    script_dir = os.path.dirname(os.path.abspath(__file__))
    filename = os.path.join(
        script_dir, "data.pkl"
    )  # Replace with your actual filename
    if not os.path.isfile(filename):
        print(f"Error: File {filename} not found in {script_dir}")
        return
    else:
        with open(filename, "rb") as f:
            data = pickle.load(f)

        # Extract data
        time_array = np.array(data["time"])  # Shape: (N,)
        # Optional: Normalize time data for better performance
        # time_array = (time_array - time_array.min()) / (time_array.max() - time_array.min())

    # Generate a new goal position
    goal_position_bounds = {
        "x": (0.6, 0.8),
        "y": (-0.1, 0.1),
        "z": (0.12, 0.12),
    }
    # Create a set of goal positions
    number_of_goal_positions_to_test = 1
    goal_positions = []
    for _ in range(number_of_goal_positions_to_test):
        goal_positions.append(
            [
                np.random.uniform(*goal_position_bounds["x"]),
                np.random.uniform(*goal_position_bounds["y"]),
                np.random.uniform(*goal_position_bounds["z"]),
            ]
        )

    conf_file_name = "pandaconfig.json"  # Configuration file for the robot
    root_dir = os.path.dirname(os.path.abspath(__file__))

    # Configuration for the simulation
    sim = pb.SimInterface(
        conf_file_name, root_dir, use_gui=False
    )  # Initialize simulation interface

    # Get active joint names from the simulation
    ext_names = sim.getNameActiveJoints()
    ext_names = np.expand_dims(
        np.array(ext_names), axis=0
    )  # Adjust the shape for compatibility

    source_names = ["pybullet"]  # Define the source for dynamic modeling

    # Create a dynamic model of the robot
    dyn_model = PinWrapper(
        conf_file_name,
        "pybullet",
        ext_names,
        source_names,
        False,
        0,
        root_dir,
    )
    num_joints = dyn_model.getNumberofActuatedJoints()

    controlled_frame_name = "panda_link8"
    init_joint_angles = sim.GetInitMotorAngles()
    init_cartesian_pos, init_R = dyn_model.ComputeFK(
        init_joint_angles, controlled_frame_name
    )
    print(f"Initial joint angles: {init_joint_angles}")

    # Command and control loop
    cmd = MotorCommands()  # Initialize command structure for motors

    # PD controller gains
    kp = 1000  # Proportional gain
    kd = 100  # Derivative gain

    # Get joint velocity limits
    joint_vel_limits = sim.GetBotJointsVelLimit()

    time_step = sim.GetTimeStep()
    # Generate test time array
    test_time_array = np.arange(time_array.min(), time_array.max(), time_step)

    for goal_position in goal_positions:
        print("Testing new goal position------------------------------------")
        print(f"Goal position: {goal_position}")
        FIG_DIR = os.path.join(DIR, f"depth_{DEPTH}")
        [
            os.makedirs(os.path.join(FIG_DIR, f"task{i}"), exist_ok=True)
            for i in [3.1, 3.2, 3.3]
        ]

        q_pred_all, qd_pred_all = {}, {}
        q_mes_all, qd_mes_all = {}, {}
        q_des_all, qd_des_all = {}, {}
        tau_all = {}
        for model_type in [
            "neural_network",
            "random_forest",
            "smoothed_random_forest",
        ]:
            q_mes_all[model_type], qd_mes_all[model_type] = [], []
            q_des_all[model_type], qd_des_all[model_type] = [], []
            tau_all[model_type] = []

            # Load all the models in a list
            models = []
            if model_type == "neural_network":
                for joint_idx in range(num_joints):
                    # Instantiate the model
                    model = MLP()
                    # Load the saved model
                    model_filename = os.path.join(
                        script_dir, f"neuralq{joint_idx+1}.pt"
                    )
                    model.load_state_dict(torch.load(model_filename))
                    model.eval()
                    models.append(model)
            elif model_type in ["random_forest", "smoothed_random_forest"]:
                for joint_idx in range(num_joints):
                    # Load the saved Random Forest model
                    model_filename = os.path.join(
                        script_dir,
                        "rf",
                        f"{DEPTH}",
                        f"rf_joint{joint_idx+1}.joblib",
                    )
                    model = joblib.load(model_filename)
                    models.append(model)
            else:
                print(
                    "Invalid model type specified. Please set neural_network_or_random_forest to 'neural_network' or 'random_forest'"
                )
                return

            # Initialize the simulation
            sim.ResetPose()
            current_time = 0  # Initialize current time

            # Create test input features
            test_goal_positions = np.tile(
                goal_position, (len(test_time_array), 1)
            )  # Shape: (num_points, 3)
            test_input = np.hstack(
                (test_time_array.reshape(-1, 1), test_goal_positions)
            )  # Shape: (num_points, 4)

            # Predict joint positions for the new goal position
            predicted_joint_positions_over_time = np.zeros(
                (len(test_time_array), num_joints)
            )  # Shape: (num_points, 7)

            for joint_idx in range(num_joints):
                if model_type == "neural_network":
                    # Prepare the test input
                    test_input_tensor = torch.from_numpy(
                        test_input
                    ).float()  # Shape: (num_points, 4)

                    # Predict joint positions using the neural network
                    with torch.no_grad():
                        predictions = (
                            models[joint_idx](test_input_tensor)
                            .numpy()
                            .flatten()
                        )  # Shape: (num_points,)
                elif model_type == "random_forest":
                    # Predict joint positions using the Random Forest
                    predictions = models[joint_idx].predict(
                        test_input
                    )  # Shape: (num_points,)
                elif model_type == "smoothed_random_forest":
                    # Predict joint positions using the Random Forest and apply smoothing
                    predictions = models[joint_idx].predict(
                        test_input
                    )  # Shape: (num_points,)
                    predictions = exponential_moving_average(predictions)

                # Store the predicted joint positions
                predicted_joint_positions_over_time[:, joint_idx] = predictions

            q_pred_all[model_type] = predicted_joint_positions_over_time

            # Compute qd_des_over_time by numerically differentiating the predicted joint positions
            qd_des_over_time = (
                np.gradient(
                    predicted_joint_positions_over_time, axis=0, edge_order=2
                )
                / time_step
            )
            # Clip the joint velocities to the joint limits
            qd_des_over_time_clipped = np.clip(
                qd_des_over_time,
                -np.array(joint_vel_limits),
                np.array(joint_vel_limits),
            )
            qd_pred_all[model_type] = qd_des_over_time_clipped

            # Data collection loop
            while current_time < test_time_array.max():
                # Measure current state
                q_mes = sim.GetMotorAngles(0)
                qd_mes = sim.GetMotorVelocities(0)
                qdd_est = sim.ComputeMotorAccelerationTMinusOne(0)

                # Get the index corresponding to the current time
                current_index = int(current_time / time_step)
                if current_index >= len(test_time_array):
                    current_index = len(test_time_array) - 1

                # Get q_des and qd_des_clip from predicted data
                q_des = predicted_joint_positions_over_time[
                    current_index, :
                ]  # Shape: (7,)
                qd_des_clip = qd_des_over_time_clipped[
                    current_index, :
                ]  # Shape: (7,)

                # Control command
                tau_cmd = feedback_lin_ctrl(
                    dyn_model, q_mes, qd_mes, q_des, qd_des_clip, kp, kd
                )
                cmd.SetControlCmd(
                    tau_cmd, ["torque"] * num_joints
                )  # Set the torque command
                sim.Step(cmd, "torque")  # Simulation step with torque command

                # Keyboard event handling
                keys = sim.GetPyBulletClient().getKeyboardEvents()
                qKey = ord("q")

                # Exit logic with 'q' key
                if (
                    qKey in keys
                    and keys[qKey] & sim.GetPyBulletClient().KEY_WAS_TRIGGERED
                ):
                    print("Exiting simulation.")
                    break

                q_mes_all[model_type].append(q_mes)
                qd_mes_all[model_type].append(qd_mes)
                q_des_all[model_type].append(q_des)
                qd_des_all[model_type].append(qd_des_clip)
                tau_all[model_type].append(tau_cmd)

                # Time management
                time.sleep(time_step)  # Control loop timing
                current_time += time_step

            # After the trajectory, compute the final cartesian position
            final_predicted_joint_positions = (
                predicted_joint_positions_over_time[-1, :]
            )  # Shape: (7,)
            final_cartesian_pos, final_R = dyn_model.ComputeFK(
                final_predicted_joint_positions, controlled_frame_name
            )
            print(f"Final computed cartesian position: {final_cartesian_pos}")
            # Compute position error
            position_error = np.linalg.norm(final_cartesian_pos - goal_position)
            print(
                f"Position error between computed position and goal: {position_error}"
            )

            q_pred_all[model_type] = q_pred_all[model_type][CUTOFF + 1 :]
            qd_pred_all[model_type] = qd_pred_all[model_type][CUTOFF + 1 :]
            q_mes_all[model_type] = np.vstack(q_mes_all[model_type][CUTOFF:])
            qd_mes_all[model_type] = np.vstack(qd_mes_all[model_type][CUTOFF:])
            q_des_all[model_type] = np.vstack(q_des_all[model_type][CUTOFF:])
            qd_des_all[model_type] = np.vstack(qd_des_all[model_type][CUTOFF:])
            tau_all[model_type] = np.vstack(tau_all[model_type][CUTOFF:])
        test_time_array = test_time_array[CUTOFF + 1 :]
        tracking_error = {
            model_type: q_mes_all[model_type] - q_des_all[model_type]
            for model_type in [
                "neural_network",
                "random_forest",
                "smoothed_random_forest",
            ]
        }

        # Task 3.1 - predicted joint positions and velocities
        for joint_idx in range(num_joints):
            fig, axs = plt.subplots(2, 1, sharex=True)
            axs[0].plot(
                test_time_array,
                q_pred_all["random_forest"][:, joint_idx],
                label=f"Random Forest (Depth={DEPTH})",
            )
            axs[0].plot(
                test_time_array,
                q_pred_all["neural_network"][:, joint_idx],
                label="Neural Network",
            )
            axs[0].legend()
            axs[0].set_ylabel("Position (rad)")
            axs[0].set_title(f"Joint {joint_idx+1} Position and Velocity")
            axs[1].plot(
                test_time_array,
                qd_pred_all["random_forest"][:, joint_idx],
                label=f"Random Forest (Depth={DEPTH})",
            )
            axs[1].plot(
                test_time_array,
                qd_pred_all["neural_network"][:, joint_idx],
                label="Neural Network",
            )
            axs[1].set_xlabel("Time (s)")
            axs[1].set_ylabel("Velocity (rad/s)")
            fig.subplots_adjust(hspace=0)
            plt.tight_layout()
            plt.savefig(
                f"{FIG_DIR}/task3.1/joint_{joint_idx+1}_pos_vel_pred.{EXT}"
            )

        # Task 3.2 - tracking error and control inputs
        for joint_idx in range(num_joints):
            fig, axs = plt.subplots(2, 1, sharex=True)
            axs[0].plot(
                test_time_array,
                tracking_error["random_forest"][:, joint_idx],
                label=f"Random Forest (Depth={DEPTH})",
            )
            axs[0].plot(
                test_time_array,
                tracking_error["neural_network"][:, joint_idx],
                label="Neural Network",
            )
            axs[0].legend()
            axs[0].set_ylabel("Tracking Error (rad)")
            axs[0].set_title(
                f"Joint {joint_idx+1} Tracking Error and Control Input"
            )
            axs[1].plot(
                test_time_array,
                tau_all["random_forest"][:, joint_idx],
                label=f"Random Forest (Depth={DEPTH})",
            )
            axs[1].plot(
                test_time_array,
                tau_all["neural_network"][:, joint_idx],
                label="Neural Network",
            )
            axs[1].set_xlabel("Time (s)")
            axs[1].set_ylabel("Torque (Nm)")
            fig.subplots_adjust(hspace=0)
            plt.tight_layout()
            plt.savefig(
                f"{FIG_DIR}/task3.2/joint_{joint_idx+1}_error_tau.{EXT}"
            )
            plt.close()

        # Task 3.3 - smoothed vs original data
        for joint_idx in range(num_joints):
            fig, axs = plt.subplots(2, 1, sharex=True)
            axs[0].plot(
                test_time_array,
                q_mes_all["smoothed_random_forest"][:, joint_idx],
                label=f"Measured",
            )
            axs[0].plot(
                test_time_array,
                q_des_all["smoothed_random_forest"][:, joint_idx],
                label="Desired",
            )
            axs[0].legend()
            axs[0].set_ylabel("Position (rad)")
            axs[0].set_title(
                f"Joint {joint_idx+1} Position and Velocity (Smoothed)"
            )
            axs[1].plot(
                test_time_array,
                qd_mes_all["smoothed_random_forest"][:, joint_idx],
                label="Measured",
            )
            axs[1].plot(
                test_time_array,
                qd_des_all["smoothed_random_forest"][:, joint_idx],
                label="Desired",
            )
            axs[1].set_xlabel("Time (s)")
            axs[1].set_ylabel("Velocity (rad/s)")
            fig.subplots_adjust(hspace=0)
            plt.tight_layout()
            plt.savefig(
                f"{FIG_DIR}/task3.3/joint_{joint_idx+1}_pos_vel_smoothed.{EXT}"
            )

            fig, axs = plt.subplots(2, 1, sharex=True)
            axs[0].plot(
                test_time_array,
                tracking_error["random_forest"][:, joint_idx],
                label=f"Random Forest (Depth={DEPTH})",
            )
            axs[0].plot(
                test_time_array,
                tracking_error["smoothed_random_forest"][:, joint_idx],
                label="Smoothed Random Forest",
            )
            axs[0].legend()
            axs[0].set_ylabel("Tracking Error (rad)")
            axs[0].set_title(
                f"Joint {joint_idx+1} Tracking Error and Control Input"
            )
            axs[1].plot(
                test_time_array,
                tau_all["random_forest"][:, joint_idx],
                label=f"Random Forest (Depth={DEPTH})",
            )
            axs[1].plot(
                test_time_array,
                tau_all["smoothed_random_forest"][:, joint_idx],
                label="Smoothed Random Forest",
            )
            axs[1].set_xlabel("Time (s)")
            axs[1].set_ylabel("Torque (Nm)")
            fig.subplots_adjust(hspace=0)
            plt.tight_layout()
            plt.savefig(
                f"{FIG_DIR}/task3.3/joint_{joint_idx+1}_error_tau_smoothed.{EXT}"
            )
            plt.close()


if __name__ == "__main__":
    main()
