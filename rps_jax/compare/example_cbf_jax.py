from rps_jax.robotarium import Robotarium

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from jax import jit, vmap, lax
from functools import partial
from rps_jax.utilities.barrier_certificates2 import create_robust_barriers
from rps_jax.utilities.controllers import create_clf_unicycle_position_controller

class WrappedRobotarium(object):
    def __init__(self, env, num_envs):
        self.env = env
        self.num_envs = num_envs
        self.barrier_fn = create_robust_barriers()
        self.controller = create_clf_unicycle_position_controller()
    
    def wrapped_step(self):
        return self.env.step()
    
    def move(self, pose):
        goals = jnp.array([[-10, 0], [10, 0]]).T
        dxu = self.controller(pose, goals)
        print(dxu.shape)
        print(pose.shape)
        dxu_safe = dxu
        dxu_safe = self.barrier_fn(dxu, pose, [])
        print(dxu_safe.shape)
        return dxu_safe

    def batched_step(self, poses, unused):
        actions = vmap(self.move, in_axes=(0))(poses)
        # print(actions.shape)
        new_poses = jax.vmap(self.env.batch_step, in_axes=(0, 0))(poses, actions)
        # print(new_poses.shape)
        return new_poses, new_poses

@partial(jax.jit, static_argnames=('num_envs', 'num_t'))
def drive_straight_jax(num_envs, num_t):
    env = Robotarium(number_of_robots=2)
    wrapped_env = WrappedRobotarium(env, num_envs)
    initial_poses = jnp.stack([jnp.array([[10., 0., 0], [-10., 0., 0.]]).T for _ in range(num_envs)], axis=0)
    # print(initial_poses.shape)
    final_poses, batch = jax.lax.scan(wrapped_env.batched_step, initial_poses, None, num_t)
    return batch

if __name__ == "__main__":
    num_envs = 10
    num_t = 2
    batch = jax.block_until_ready(drive_straight_jax(num_envs, num_t))
    # print(batch.shape)

    # Select one environment to plot
    env_index = 0

    # Extract x and y positions for both robots over timesteps
    x_positions_robot_1 = batch[:, env_index, 0, 0]
    y_positions_robot_1 = batch[:, env_index, 1, 0]
    x_positions_robot_2 = batch[:, env_index, 0, 1]
    y_positions_robot_2 = batch[:, env_index, 1, 1]
    print(x_positions_robot_1)

    # Plot x and y positions for both robots
    plt.figure(figsize=(5, 5))
    plt.plot(x_positions_robot_1, y_positions_robot_1, label='Robot 1', color='blue')
    plt.plot(x_positions_robot_2, y_positions_robot_2, label='Robot 2', color='red')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.title(f'Trajectories of both robots in environment {env_index} over time')
    plt.legend()
    plt.grid(True)
    plt.show()