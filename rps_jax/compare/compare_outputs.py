import matplotlib.pyplot as plt
from example import drive_in_circle
from example_jax import drive_in_circle_jax
import numpy as np

num_envs = 1
timesteps = 100000

poses_jax = drive_in_circle_jax(num_envs, timesteps)
poses = np.array(drive_in_circle(num_envs, timesteps))

poses_jax = poses_jax.squeeze()
poses = poses.squeeze()

print(np.mean(np.linalg.norm(np.array(poses_jax)-poses, axis=-1)))
