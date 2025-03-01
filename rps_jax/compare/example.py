import numpy as np
from rps.robotarium import Robotarium
from tqdm import tqdm

# Example usage
def drive_in_circle(num_env, num_t):
    poses_batch = []
    for env in range(num_env):
        poses_batch.append([])
        # Create Robotarium object
        r = Robotarium(number_of_robots=1, sim_in_real_time=False, show_figure=False, initial_conditions=np.array([[0.0, 0.0, 0.0]]).astype(np.float32).T)

        # Define the angular velocity
        radius = 1
        v = 1
        omega = v / radius

        poses = r.get_poses()
        # poses_batch[env].append(poses)
        for t in tqdm(range(num_t)):
            # Calculate the unicycle velocity commands
            w = omega

            # Set the robot's velocity
            dxu = np.array([[v], [w]])

            # Set the velocities of the robots
            r.set_velocities(np.array([0]), dxu)

            # Iterate the simulation
            r.step()
            poses = r.get_poses()
            poses_batch[env].append(poses.tolist())
            # poses.vstack((poses, r.get_poses()))
    return poses_batch

# # Example usage
# num_env = 1
# num_t = 1_000_000
# drive_in_circle(num_env, num_t)