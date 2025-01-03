import os
import pickle

import joblib  # For saving and loading models
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

TASK = 2.2
CUR_DIR = os.path.dirname(os.path.realpath(__file__))  # current directory
DIR = os.path.join(CUR_DIR, "figures", f"task{TASK}")  # figure directory
EXT = "pdf"  # figure extension
os.makedirs(DIR, exist_ok=True)  # create figure directory if not exist
RF_MODEL_DIR = "rf"
DEPTH = 4
DEPTH_STR = str(DEPTH) if DEPTH else 'none'
print(f"Performing Task {TASK}...")


# Set the visualization flag
visualize = True  # Set to True to enable visualization, False to disable
training_flag = False  # Set to True to train the models, False to skip training
test_cartesian_accuracy_flag = True  # Set to True to test the model with a new goal position, False to skip testing

if training_flag:
    # Load the saved data
    script_dir = os.path.dirname(os.path.abspath(__file__))
    filename = os.path.join(
        script_dir, "data.pkl"
    )  # Replace with your actual filename

    # Check if the file exists
    if not os.path.isfile(filename):
        print(f"Error: File {filename} not found in {script_dir}")
    else:
        with open(filename, "rb") as f:
            data = pickle.load(f)

        # Extract data
        time_array = np.array(data["time"])  # Shape: (N,)
        q_mes_all = np.array(data["q_mes_all"])  # Shape: (N, 7)
        goal_positions = np.array(data["goal_positions"])  # Shape: (N, 3)

        # Visualizing training data
        # for iii in range(7):
        #     plt.figure()
        #     plt.plot(time_array, q_mes_all[:, iii], label=f"Joint {iii+1}")
        #     plt.xlabel("Time (s)")
        #     plt.ylabel("Joint Position (rad)")
        #     plt.title(
        #         f"All Recorded Trajectories for Joint {iii+1}"
        #     )
        #     plt.legend()
        #     plt.savefig(f'joint_{iii+1}.pdf')
        #     plt.show()

        # Optional: Normalize time data for better performance
        # time_array = (time_array - time_array.min()) / (time_array.max() - time_array.min())

        # Combine time and goal data to form the input features
        X = np.hstack(
            (time_array.reshape(-1, 1), goal_positions)
        )  # Shape: (N, 4)

        # Split ratio
        split_ratio = 0.8

        # Initialize lists to hold training and test data for all joints
        x_train_list = []
        x_test_list = []
        y_train_list = []
        y_test_list = []

        for joint_idx in range(7):
            # Extract joint data
            y = q_mes_all[:, joint_idx]  # Shape: (N,)

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, train_size=split_ratio, shuffle=True, random_state=42
            )

            # Store split data
            x_train_list.append(X_train)
            x_test_list.append(X_test)
            y_train_list.append(y_train)
            y_test_list.append(y_test)

            # Initialize the Random Forest regressor
            rf_model = RandomForestRegressor(
                n_estimators=100,  # Number of trees
                max_depth=DEPTH,  # Maximum depth of the tree
                random_state=42,
                n_jobs=-1,  # Use all available cores
            )

            # Train the model
            rf_model.fit(X_train, y_train)

            # Evaluate on training set
            y_train_pred = rf_model.predict(X_train)
            train_mse = np.mean((y_train - y_train_pred) ** 2)

            # Evaluate on test set
            y_test_pred = rf_model.predict(X_test)
            test_mse = np.mean((y_test - y_test_pred) ** 2)

            print(f"\nJoint {joint_idx+1}")
            print(f"Train MSE: {train_mse:.6f}")
            print(f"Test MSE: {test_mse:.6f}")
            depths = [e.get_depth() for e in rf_model.estimators_]
            print(f"Max Depth of Random Forest: {max(depths)}")

            # Save the trained model
            os.makedirs(os.path.join(script_dir, RF_MODEL_DIR, DEPTH_STR), exist_ok=True)
            model_filename = os.path.join(
                script_dir, RF_MODEL_DIR, DEPTH_STR, f"rf_joint{joint_idx+1}.joblib"
            )
            joblib.dump(rf_model, model_filename)
            print(f"Model for Joint {joint_idx+1} saved as {model_filename}")

            # Visualization (if enabled)
            if visualize:
                print(f"Visualizing results for Joint {joint_idx+1}...")

                # Plot true vs predicted positions on the test set
                sorted_indices = np.argsort(X_test[:, 0])
                X_test_sorted = X_test[sorted_indices]
                y_test_sorted = y_test[sorted_indices]
                y_test_pred_sorted = y_test_pred[sorted_indices]

                plt.figure(figsize=(10, 5))
                plt.plot(
                    X_test_sorted[:, 0],
                    y_test_sorted,
                    label="True Joint Positions",
                )
                plt.plot(
                    X_test_sorted[:, 0],
                    y_test_pred_sorted,
                    label="Predicted Joint Positions",
                    linestyle="--",
                )
                plt.xlabel("Time (s)")
                plt.ylabel("Joint Position (rad)")
                plt.title(
                    f"Joint {joint_idx+1} Position Prediction on Test Set"
                )
                plt.legend()
                plt.grid(True)
                os.makedirs(f"{DIR}/{DEPTH_STR}", exist_ok=True)
                plt.savefig(f"{DIR}/{DEPTH_STR}/predicted_pos_goal_{joint_idx+1}.{EXT}")
                #plt.show()

        print("Training and visualization completed.")

if test_cartesian_accuracy_flag:

    if not training_flag:
        # Load the saved data
        script_dir = os.path.dirname(os.path.abspath(__file__))
        filename = os.path.join(
            script_dir, "data.pkl"
        )  # Replace with your actual filename
        if not os.path.isfile(filename):
            print(f"Error: File {filename} not found in {script_dir}")
        else:
            with open(filename, "rb") as f:
                data = pickle.load(f)

            # Extract data
            time_array = np.array(data["time"])  # Shape: (N,)

    # Testing with new goal positions
    print("\nTesting the model with new goal positions...")

    # Load all the models into a list
    models = []
    for joint_idx in range(7):
        # Load the saved model
        script_dir = os.path.dirname(os.path.abspath(__file__))

        # The name of the saved model
        model_filename = os.path.join(
            script_dir, RF_MODEL_DIR, DEPTH_STR, f"rf_joint{joint_idx+1}.joblib"
        )

        try:
            rf_model = joblib.load(model_filename)

        except FileNotFoundError:
            print(f"Cannot find file {model_filename}")
            print(
                "task_22_goal_pos needs to be run at least once with training_flag=True"
            )
            quit()

        models.append(rf_model)

    # Generate new goal positions
    goal_position_bounds = {
        "x": (0.6, 0.8),
        "y": (-0.1, 0.1),
        "z": (0.12, 0.12),
    }
    # Create a set of goal positions
    number_of_goal_positions_to_test = 10
    goal_positions = []
    # for i in range(number_of_goal_positions_to_test):
    #     goal_positions.append(
    #         [
    #             np.random.uniform(*goal_position_bounds["x"]),
    #             np.random.uniform(*goal_position_bounds["y"]),
    #             np.random.uniform(*goal_position_bounds["z"]),
    #         ]
    #     )

    goal_positions = [[0.691397925624174, 0.002102621718463246, 0.12], [0.7098937836805903, -0.08694438235835333, 0.12], [0.7672410862743346, 0.035875201286842656, 0.12], [0.6627737729819734, 0.046329614423186144, 0.12], [0.6539269657617497, 0.00494133901407659, 0.12], [0.6132856638707109, -0.0991282973197782, 0.12], [0.7156702852353941, -0.003447701778067927, 0.12], [0.6054355103671692, 0.005484734697656202, 0.12], [0.7837080889282269, -0.030574751431943464, 0.12], [0.6534992783237846, -0.04376744160873707, 0.12]]

    # Generate test time array
    test_time_array = np.linspace(
        time_array.min(), time_array.max(), 100
    )  # For example, 100 time steps

    # Initialize the dynamic model
    from simulation_and_control import (
        CartesianDiffKin,
        MotorCommands,
        PinWrapper,
        feedback_lin_ctrl,
        pb,
    )

    conf_file_name = "pandaconfig.json"  # Configuration file for the robot
    root_dir = os.path.dirname(os.path.abspath(__file__))
    # Adjust root directory if necessary
    name_current_directory = "tests"
    root_dir = root_dir.replace(name_current_directory, "")
    # Initialize simulation interface
    sim = pb.SimInterface(conf_file_name, conf_file_path_ext=root_dir, use_gui=False)

    # Get active joint names from the simulation
    ext_names = sim.getNameActiveJoints()
    ext_names = np.expand_dims(
        np.array(ext_names), axis=0
    )  # Adjust the shape for compatibility

    source_names = ["pybullet"]  # Define the source for dynamic modeling

    # Create a dynamic model of the robot
    dyn_model = PinWrapper(
        conf_file_name, "pybullet", ext_names, source_names, False, 0, root_dir
    )
    num_joints = dyn_model.getNumberofActuatedJoints()

    controlled_frame_name = "panda_link8"
    init_joint_angles = sim.GetInitMotorAngles()
    init_cartesian_pos, init_R = dyn_model.ComputeFK(
        init_joint_angles, controlled_frame_name
    )
    init_cartesian_pos = init_cartesian_pos.copy()
    print(f"Initial joint angles: {init_joint_angles}")
    print(f"Initial cartesial position: {init_cartesian_pos}")

    pos_errs = []

    for ii, goal_position in enumerate(goal_positions):
        print(f"\nTesting {ii+1} goal position------------------------------------")

        # Create test input features
        test_goal_positions = np.tile(
            goal_position, (len(test_time_array), 1)
        )  # Shape: (100, 3)
        test_input = np.hstack(
            (test_time_array.reshape(-1, 1), test_goal_positions)
        )  # Shape: (100, 4)

        # Predict joint positions for the new goal position
        predicted_joint_positions_over_time = np.zeros(
            (len(test_time_array), 7)
        )  # Shape: (num_points, 7)

        for joint_idx in range(7):
            # Predict joint positions
            y_pred = models[joint_idx].predict(
                test_input
            )  # Shape: (num_points,)
            # Store the predicted joint positions
            predicted_joint_positions_over_time[:, joint_idx] = y_pred

        # Get the final predicted joint positions (at the last time step)
        final_predicted_joint_positions = predicted_joint_positions_over_time[
            -1, :
        ]  # Shape: (7,)

        # Compute forward kinematics
        final_cartesian_pos, final_R = dyn_model.ComputeFK(
            final_predicted_joint_positions, controlled_frame_name
        )

        print(f"Goal position: {goal_position}")
        print(f"Computed cartesian position: {final_cartesian_pos}")
        print(
            f"Predicted joint positions at final time step: {final_predicted_joint_positions}"
        )

        # Compute position error
        position_error = np.linalg.norm(final_cartesian_pos - goal_position)
        print(
            f"Position error between computed position and goal: {position_error}"
        )
        mse = mean_squared_error(goal_position, final_cartesian_pos)
        print(
            f"MSE between computed position and goal: {mse}"
        )

        pos_errs.append(position_error)

        # Optional: Visualize the cartesian trajectory over time
        if visualize:
            cartesian_positions_over_time = []
            for i in range(len(test_time_array)):
                joint_positions = predicted_joint_positions_over_time[i, :]
                cartesian_pos, _ = dyn_model.ComputeFK(
                    joint_positions, controlled_frame_name
                )
                cartesian_positions_over_time.append(cartesian_pos.copy())

            cartesian_positions_over_time = np.array(
                cartesian_positions_over_time
            )  # Shape: (num_points, 3)

            # Plot x, y, z positions over time
            plt.figure(figsize=(10, 5))
            plt.plot(
                test_time_array,
                cartesian_positions_over_time[:, 0],
                label="X Position",
            )
            plt.plot(
                test_time_array,
                cartesian_positions_over_time[:, 1],
                label="Y Position",
            )
            plt.plot(
                test_time_array,
                cartesian_positions_over_time[:, 2],
                label="Z Position",
            )
            print(init_cartesian_pos)
            plt.scatter([0], [init_cartesian_pos[0]], s=50, c='blue', label="Init X position")
            plt.scatter([0], [init_cartesian_pos[1]], s=50, c='orange', label="Init Y position")
            plt.scatter([0], [init_cartesian_pos[2]], s=50, c='green', label="Init Z position")
            plt.scatter([test_time_array[-1]] * 3, goal_position, s=50, c='red', label="Goal position")
            plt.xlabel("Time (s)")
            plt.ylabel("Cartesian Position (m)")
            plt.title("Predicted Cartesian Positions Over Time")
            plt.legend()
            plt.grid(True)
            os.makedirs(f"{DIR}/{DEPTH_STR}/testing", exist_ok=True)
            plt.savefig(f"{DIR}/{DEPTH_STR}/testing/predicted_pos_goal_{ii+1}.{EXT}")
            #plt.show()

            # Plot the trajectory in 3D space
            from mpl_toolkits.mplot3d import Axes3D

            fig = plt.figure()
            ax = fig.add_subplot(111, projection="3d")
            ax.plot(
                cartesian_positions_over_time[:, 0],
                cartesian_positions_over_time[:, 1],
                cartesian_positions_over_time[:, 2],
                label="Predicted Trajectory",
            )
            ax.scatter(
                init_cartesian_pos[0], 
                init_cartesian_pos[1],
                init_cartesian_pos[2], 
                color='green', 
                label="Init position")
            ax.scatter(
                goal_position[0],
                goal_position[1],
                goal_position[2],
                color="red",
                label="Goal Position",
            )
            ax.set_xlabel("X Position (m)")
            ax.set_ylabel("Y Position (m)")
            ax.set_zlabel("Z Position (m)")
            ax.set_title("Predicted Cartesian Trajectory")
            plt.legend()
            plt.savefig(f"{DIR}/{DEPTH_STR}/testing/predicted_trajectories_{ii+1}.{EXT}")
            #plt.show()

    formatted_vector = [float(f"{x:.3f}") for x in pos_errs]
    print('Pos Errors: ')
    print(*formatted_vector, sep=' & ')
