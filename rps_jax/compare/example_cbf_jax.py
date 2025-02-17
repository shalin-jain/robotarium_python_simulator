from rps_jax.robotarium import Robotarium

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from jax import jit, vmap, lax
from functools import partial
from rps_jax.utilities.barrier_certificates2 import create_robust_barriers
from rps_jax.utilities.controllers import create_clf_unicycle_position_controller
import numpy as np

class WrappedRobotarium(object):
    def __init__(self, env, num_envs):
        self.env = env
        self.num_envs = num_envs
        self.barrier_fn = create_robust_barriers()
        self.controller = create_clf_unicycle_position_controller()
    
    def wrapped_step(self):
        return self.env.step()
    
    def move(self, pose):
        goals = jnp.array([[-10, 0.], [10, -0.]]).T
        dxu = self.controller(pose, goals)
        # dxu = jnp.array([[1, 0], [1, 0]]).T
        dxu_safe = self.barrier_fn(dxu, pose, [])
        print(dxu_safe.shape)
        return dxu_safe

    def batched_step(self, poses, unused):
        actions = vmap(self.move, in_axes=(0))(poses)
        print(actions.shape)
        new_poses = jax.vmap(self.env.batch_step, in_axes=(0, 0))(poses, actions)
        # print(new_poses.shape)
        return new_poses, new_poses

@partial(jax.jit, static_argnames=('num_envs', 'num_t'))
def drive_straight_jax(num_envs, num_t):
    env = Robotarium(number_of_robots=2)
    wrapped_env = WrappedRobotarium(env, num_envs)
    initial_poses = jnp.stack([jnp.array([[10., -5, 0.0], [-10., -5, 0]]).T for _ in range(num_envs)], axis=0)
    # print(initial_poses.shape)
    final_poses, batch = jax.lax.scan(wrapped_env.batched_step, initial_poses, None, num_t)
    return batch

if __name__ == "__main__":
    num_envs = 10
    num_t = 300
    batch = jax.block_until_ready(drive_straight_jax(num_envs, num_t))
    # print(batch.shape)

    # Select one environment to plot
    env_index = 0

    # Extract x and y positions for both robots over timesteps
    x_positions_robot_1 = np.array(batch[:, env_index, 0, 0])
    y_positions_robot_1 = np.array(batch[:, env_index, 1, 0])
    x_positions_robot_2 = np.array(batch[:, env_index, 0, 1])
    y_positions_robot_2 = np.array(batch[:, env_index, 1, 1])
    # print(x_positions_robot_1)

    def check_safety_violation(x1, y1, x2, y2, safety_radius):
        # Compute squared distance to avoid sqrt computation
        squared_distance = (x1 - x2) ** 2 + (y1 - y2) ** 2
        return squared_distance < safety_radius ** 2  # Element-wise comparison

    # Define the safety radius
    safety_radius = 0.12  # Adjust as needed

    # Compute safety violations over time
    violations = check_safety_violation(
        x_positions_robot_1, y_positions_robot_1,
        x_positions_robot_2, y_positions_robot_2,
        safety_radius
    )

    # Find timesteps where safety is violated
    violated_timesteps = jnp.where(violations)[0]

    if violated_timesteps.size > 0:
        print(f"Safety violation occurred at timesteps: {violated_timesteps}")
    else:
        print("No safety violations detected.")


    # Plot x and y positions for both robots
    # plt.figure(figsize=(5, 5))
    # plt.plot(x_positions_robot_1, y_positions_robot_1, label='Robot 1', color='blue')
    # plt.plot(x_positions_robot_2, y_positions_robot_2, label='Robot 2', color='red')
    # plt.xlabel('X Position')
    # plt.ylabel('Y Position')
    # plt.title(f'Trajectories of both robots in environment {env_index} over time')
    # plt.legend()
    # plt.grid(True)
    # plt.show()

    import numpy as np
    import matplotlib.pyplot as plt
    from PIL import Image
    import imageio

    # Convert JAX arrays to NumPy for compatibility
    x_positions_robot_1 = np.asarray(x_positions_robot_1)
    y_positions_robot_1 = np.asarray(y_positions_robot_1)
    x_positions_robot_2 = np.asarray(x_positions_robot_2)
    y_positions_robot_2 = np.asarray(y_positions_robot_2)

    # Setup plot
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set_xlim(np.min(x_positions_robot_1) - 5, np.max(x_positions_robot_1) + 5)
    ax.set_ylim(np.min(y_positions_robot_1) - 5, np.max(y_positions_robot_1) + 5)
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_title(f'(Env {env_index})')

    # Plot full trajectory with transparency (trails)
    ax.plot(x_positions_robot_1, y_positions_robot_1, color='blue', alpha=0.3)
    ax.plot(x_positions_robot_2, y_positions_robot_2, color='red', alpha=0.3)

    # Initialize moving robot markers
    robot1, = ax.plot([], [], 'bo', markersize=6, label="robot 0")
    robot2, = ax.plot([], [], 'ro', markersize=6, label="robot 1")

    ax.legend()

    # List to store frames
    frames = []

    # Generate and save each frame
    for frame in range(len(x_positions_robot_1)):
        x1, y1 = x_positions_robot_1[frame], y_positions_robot_1[frame]
        x2, y2 = x_positions_robot_2[frame], y_positions_robot_2[frame]

        # Update robot positions
        robot1.set_data([x1], [y1])
        robot2.set_data([x2], [y2])

        # Save the current frame as an image
        fig.canvas.draw()
        frame_image = np.array(fig.canvas.renderer.buffer_rgba())  # Get image from canvas
        frames.append(Image.fromarray(frame_image))  # Convert to PIL Image

    # Save frames as a GIF
    gif_path = "robot_trajectory.gif"
    frames[0].save(gif_path, save_all=True, append_images=frames[1:], duration=20, loop=0)

    print(f"GIF saved at {gif_path}")