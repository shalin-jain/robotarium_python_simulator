from rps.robotarium import Robotarium
from rps.utilities.barrier_certificates2 import create_robust_barriers
from rps.utilities.controllers import create_clf_unicycle_pose_controller as clf_uni_pose, create_clf_unicycle_position_controller as clf_uni_position

from rps_jax.robotarium import Robotarium as RobotariumJax
from rps_jax.utilities.barrier_certificates2 import create_robust_barriers as create_robust_barriers_jax
from rps_jax.utilities.controllers import create_clf_unicycle_pose_controller as clf_uni_pose_jax, create_clf_unicycle_position_controller as clf_uni_position_jax

from functools import partial
from concurrent.futures import ProcessPoolExecutor, as_completed

import jax
jax.config.update("jax_enable_x64", True)
import numpy as np
import pickle
import jax.numpy as jnp
import random
import time
import timeit
import os

np.random.seed(0)
random.seed(0)

# EXP PARAMS
POSITION_THRESHOLD = 0.01
ANGLE_THRESHOLD = 0.05
SAFETY_RADIUS = 0.2
CONTROLLERS = ['clf_uni_position', 'clf_uni_pose']
BARRIERS = [None, 'robust']
SIMULATORS = ['python', 'jax']
NUM_ENVS = [1, 5, 10]
NUM_AGENTS = 4
NUM_TIMESTEPS = 10_000
NUM_TRIALS = 30
WAYPOINTS_PER_AGENT = 50

class ControllerJax:
    def __init__(
        self,
        controller = None,
        barrier_fn = None,
        **kwargs
    ):
        """
        Initialize wrapper class for handling calling controllers and barrier functions.

        Args:
            controller: (str) name of controller, supported controllers defined in constants.py
            barrier_fn: (str) name of barrier fn, supported barrier functions defined in constants.py 
        """
        if controller is None:
            # if controller is not set, return trivial pass through of actions
            controller = lambda x, g: g
        elif controller == 'clf_uni_position':
            controller = clf_uni_position_jax()
        elif controller == 'clf_uni_pose':
            controller = clf_uni_pose_jax()

        if barrier_fn is None:
            barrier_fn = lambda dxu, x, unused: dxu
        else:
            barrier_fn = create_robust_barriers_jax(safety_radius=0.2)

        self.controller = controller
        self.barrier_fn = barrier_fn
    
    def get_action(self, x, g):
        """
        Applies controller and barrier function to get action
        
        Args:
            x: (jnp.ndarray) 3xN states (x, y, theta)
            g: (jnp.ndarray) 2xN (x, y) positions or 3xN poses (x, y, theta)
        
        Returns:
            (jnp.ndarray) 2xN unicycle controls (linear velocity, angular velocity)
        """
        dxu = self.controller(x, g)
        dxu_safe = self.barrier_fn(dxu, x, [])

        return dxu_safe

class Controller:
    def __init__(
        self,
        controller = None,
        barrier_fn = None,
        **kwargs
    ):
        """
        Initialize wrapper class for handling calling controllers and barrier functions.

        Args:
            controller: (str) name of controller, supported controllers defined in constants.py
            barrier_fn: (str) name of barrier fn, supported barrier functions defined in constants.py 
        """
        if controller is None:
            # if controller is not set, return trivial pass through of actions
            controller = lambda x, g: g
        elif controller == 'clf_uni_position':
            controller = clf_uni_position()
        elif controller == 'clf_uni_pose':
            controller = clf_uni_pose()

        if barrier_fn is None:
            barrier_fn = lambda dxu, x, unused: dxu
        else:
            barrier_fn = create_robust_barriers(safety_radius=0.2)

        self.controller = controller
        self.barrier_fn = barrier_fn
    
    def get_action(self, x, g):
        """
        Applies controller and barrier function to get action
        
        Args:
            x: (jnp.ndarray) 3xN states (x, y, theta)
            g: (jnp.ndarray) 2xN (x, y) positions or 3xN poses (x, y, theta)
        
        Returns:
            (jnp.ndarray) 2xN unicycle controls (linear velocity, angular velocity)
        """
        dxu = self.controller(x, g)
        dxu_safe = self.barrier_fn(dxu, x, jnp.zeros(0))

        return dxu_safe

class WrappedRobotariumJax(object):
    def __init__(self, num_agents, num_envs, waypoints, controller_fn=None, barrier_fn=None):
        self.env = RobotariumJax(number_of_robots=num_agents)
        self.num_agents = num_agents
        self.num_envs = num_envs
        self.controller = ControllerJax(controller_fn, barrier_fn)
        self.pose_sensitive = controller_fn == 'clf_uni_pose'
        self.waypoints = waypoints

    def update_goals(self, poses, goals):
        """
        If current pose is within THRESHOLD of goals, mark it as visited.

        Args:
            goals: indices of goals per robot in waypoints (n_agents, 1)
        
        Returns:
            goals: (ndarray) updated goal
        """
        goal_poses = jnp.array([self.waypoints[i, goals[i], :] for i in range(self.num_agents)]).T
        position_in_range = jnp.linalg.norm(poses[:2, :] - goal_poses[:2, :], axis=0) < POSITION_THRESHOLD
        angle_diff = jnp.abs(jnp.mod((poses[2, :] - goal_poses[2, :]) + jnp.pi, 2*jnp.pi) - jnp.pi)
        angle_in_range = jnp.where(self.pose_sensitive, angle_diff < ANGLE_THRESHOLD, True)

        # increment goal index per agent if within range
        goals = jnp.where(jnp.logical_and(position_in_range, angle_in_range), goals + 1, goals)
        goals = jnp.where(goals >= self.waypoints.shape[1], jnp.zeros_like(goals), goals)  # wrap around if goal exceeds number of waypoints

        return goals

    def batched_step_pose(self, step_state, unused):
        poses, goals = step_state

        # update waypoints
        goals = self.update_goals(poses, goals)
        goal_poses = jnp.array([self.waypoints[i, goals[i], :] for i in range(self.num_agents)]).T

        actions = self.controller.get_action(poses, goal_poses)
        new_poses = self.env.batch_step(poses, actions)

        return (new_poses, goals), new_poses
    
    def batched_step(self, step_state, unused):
        poses, goals = step_state

        # update waypoints
        goals = self.update_goals(poses, goals)
        goal_poses = jnp.array([self.waypoints[i, goals[i], :] for i in range(self.num_agents)]).T

        actions = self.controller.get_action(poses, goal_poses[:2, :])
        new_poses = self.env.batch_step(poses, actions)

        return (new_poses, goals), new_poses


class WrappedRobotarium(object):
    def __init__(self, num_agents, num_envs, waypoints, controller_fn=None, barrier_fn=None):
        self.env = Robotarium(number_of_robots=num_agents, sim_in_real_time=False, show_figure=False, initial_conditions=waypoints[:, 0, :].T)
        self.num_agents = num_agents
        self.num_envs = num_envs
        self.controller = Controller(controller_fn, barrier_fn)
        self.pose_sensitive = controller_fn == 'clf_uni_pose'
        self.waypoints = waypoints

    def update_goals(self, poses, goals):
        """
        If current pose is within THRESHOLD of goals, mark it as visited.

        Args:
            goals: indices of goals per robot in waypoints (n_agents, 1)
        
        Returns:
            goals: (ndarray) updated goal
        """
        goal_poses = np.array([self.waypoints[i, goals[i], :] for i in range(self.num_agents)]).T
        position_in_range = np.linalg.norm(poses[:2, :] - goal_poses[:2, :], axis=0) < POSITION_THRESHOLD
        angle_diff = np.abs(np.mod((poses[2, :] - goal_poses[2, :]) + np.pi, 2*np.pi) - np.pi)
        angle_in_range = np.where(self.pose_sensitive, angle_diff < ANGLE_THRESHOLD, True)

        # increment goal index per agent if within range
        goals = np.where(np.logical_and(position_in_range, angle_in_range), goals + 1, goals)
        goals = np.where(goals >= self.waypoints.shape[1], 0, goals)  # wrap around if goal exceeds number of waypoints

        return goals

    def batched_step_pose(self, step_state, unused):
        poses, goals = step_state

        # update waypoints
        goals = self.update_goals(poses, goals)
        goal_poses = np.array([self.waypoints[i, goals[i], :] for i in range(self.num_agents)]).T

        dxu = np.array(self.controller.get_action(poses, goal_poses))
        self.env.set_velocities(None, dxu)
        self.env.step()
        new_poses = self.env.get_poses()

        return (new_poses, goals), new_poses

    def batched_step(self, step_state, unused):
        poses, goals = step_state

        # update waypoints
        goals = self.update_goals(poses, goals)
        goal_poses = np.array([self.waypoints[i, goals[i], :] for i in range(self.num_agents)]).T

        dxu = np.array(self.controller.get_action(poses, goal_poses[:2, :]))
        self.env.set_velocities(None, dxu)
        self.env.step()
        new_poses = self.env.get_poses()

        return (new_poses, goals), new_poses

@partial(jax.jit, static_argnames=('num_agents', 'num_envs', 'num_t', 'controller_fn', 'barrier_fn'))
def move_random_jax(
    num_agents,
    num_envs,
    num_t,
    waypoints,
    controller_fn,
    barrier_fn,
):
    wrapped_env = WrappedRobotariumJax(num_agents, num_envs, waypoints, controller_fn, barrier_fn)
    initial_poses = waypoints[jnp.arange(num_agents), 0, :].T
    initial_goals = jnp.ones((num_agents,), dtype=jnp.int32)
    initial_poses = jnp.array([initial_poses for _ in range(num_envs)])
    initial_goals = jnp.array([initial_goals for _ in range(num_envs)])
    step_fn = wrapped_env.batched_step_pose if controller_fn == 'clf_uni_pose' else wrapped_env.batched_step
    # Use jax.vmap to vectorize the step function over the number of environments
    step_fn = jax.vmap(step_fn, in_axes=(0, None), out_axes=(0, 0))
    # Use jax.lax.scan to iterate over the number of timesteps
    step_state, batch = jax.lax.scan(step_fn, (initial_poses, initial_goals), None, num_t)

    return step_state, batch

def move_random(
    num_agents,
    num_envs,
    num_t,
    waypoints,
    controller_fn,
    barrier_fn,
):
    wrapped_env = WrappedRobotarium(num_agents, num_envs, waypoints, controller_fn, barrier_fn)
    initial_goals = np.ones((num_agents,), dtype=np.int32)
    
    poses = wrapped_env.env.get_poses()
    goals = initial_goals
    batch = np.zeros((num_t, 3, num_agents))
    step_fn = wrapped_env.batched_step_pose if controller_fn == 'clf_uni_pose' else wrapped_env.batched_step
    for i in range(num_t):
        step_state, _ = step_fn((poses, goals), None)
        poses, goals = step_state
        batch[i, ...] = poses
    batch = np.array(batch)

    # del wrapped_env
 
    return step_state, batch

def parallel_move_random(
    num_agents,
    num_envs,
    num_t,
    waypoints,
    controller_fn,
    barrier_fn,
):
    results = []
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = [
            executor.submit(move_random, num_agents, num_envs, num_t, waypoints, controller_fn, barrier_fn)
            for i in range(num_envs)
        ]
        for future in as_completed(futures):
            step_state, batch = future.result()
            results.append((step_state, batch))
    return results

def generate_waypoints(num_agents, num_waypoints, min_distance=0.5, max_tries=1000):
    waypoints = np.zeros((num_agents, num_waypoints, 3))
    
    for t in range(num_waypoints):
        tries = 0
        points = []

        while len(points) < num_agents:
            candidate = np.array([
                np.random.uniform(-1, 1),
                np.random.uniform(-1, 1),
                np.random.uniform(-np.pi, np.pi)
            ])

            # Check distance from other agents' waypoints at the same timestep
            if all(np.linalg.norm(candidate[:2] - p[:2]) >= min_distance for p in points):
                agent_idx = len(points)

                # If not the first waypoint, check distance from previous one for this agent
                if t > 0:
                    prev = waypoints[agent_idx, t - 1]
                    if np.linalg.norm(candidate[:2] - prev[:2]) < min_distance:
                        tries += 1
                        if tries > max_tries:
                            raise RuntimeError("Could not generate spaced waypoints")
                        continue

                points.append(candidate)
            tries += 1
            if tries > max_tries:
                raise RuntimeError("Could not generate spaced waypoints")

        for i, p in enumerate(points):
            waypoints[i, t] = p

    return np.array(waypoints)

def benchmark(func, args):
    elapsed_time = timeit.timeit(lambda: func(**args), number=1)
    return elapsed_time

def benchmark_jax(func, args):
    elapsed_time = timeit.timeit(lambda: jax.block_until_ready(func(**args)), number=1)
    return elapsed_time

def run_experiment():
    results = []
    
    all_conditions = [(s, c, b, n) for s in SIMULATORS for c in CONTROLLERS for b in BARRIERS for n in NUM_ENVS]

    for trial in range(NUM_TRIALS+1):
        print(f"Running trial {trial}/{NUM_TRIALS}...")
        random.shuffle(all_conditions)

        for sim_type, controller, barrier, num_envs in all_conditions:
            print(f"  Condition: Sim={sim_type}, Num Envs={num_envs}")

            # Regenerate same waypoints for both sim types for fairness
            waypoints = generate_waypoints(NUM_AGENTS, WAYPOINTS_PER_AGENT)
            
            if sim_type == 'jax':
                args = {
                    "num_agents": NUM_AGENTS,
                    "num_envs": num_envs,
                    "num_t": NUM_TIMESTEPS // num_envs,
                    "waypoints": jnp.array(waypoints),
                    "controller_fn": controller,
                    "barrier_fn": barrier
                }
                wall_time = benchmark_jax(move_random_jax, args)
            # Python sim
            else:
                args = {
                    "num_agents": NUM_AGENTS,
                    "num_envs": num_envs,
                    "num_t": NUM_TIMESTEPS // num_envs,
                    "waypoints": np.array(waypoints),
                    "controller_fn": controller,
                    "barrier_fn": barrier
                }
                wall_time = benchmark(parallel_move_random, args)

            # wall_time = end - start
            step_time = wall_time / NUM_TIMESTEPS
            print(f"    Wall time: {wall_time:.2f}s, Step time: {step_time:.6f}s, Num Envs: {num_envs}")
            
            # throw away first trial to avoide first run bias
            if trial == 0:
                continue

            results.append({
                'trial': trial,
                'simulator': sim_type,
                'wall_time': wall_time,
                'step_time': step_time,
                'num_envs': num_envs,
            })

        with open('jax_sim_parallel_results.pkl', 'wb') as f:
            pickle.dump(results, f)

    print("All trials complete. Results saved.")

def print_results_table_human_readable(results):
    """
    Prints the results in a human-readable table format.
    """
    # Print header
    print("+" + "-"*12 + "+" + "-"*15 + "+" + "-"*12 + "+" + "-"*20 + "+" + "-"*20 + "+" + "-"*20 + "+")
    print(f"| {'Num Envs':<12} | {'Simulator':<12} | {'Wall Time (ms)':<20} | {'Step Time (ms)':<20} |")
    print("+" + "-"*12 + "+" + "-"*15 + "+" + "-"*20 + "+" + "-"*20 + "+")

    # Print rows
    for num_envs in NUM_ENVS:
        for simulator in SIMULATORS:
            # Filter results for this condition
            filtered_results = [r for r in results if r['num_envs'] == num_envs and r['simulator'] == simulator]

            if filtered_results:
                # Aggregate metrics (average across trials)
                wall_time_avg = sum(r['wall_time'] for r in filtered_results) / len(filtered_results) * 1000  # Convert to ms
                step_time_avg = sum(r['step_time'] for r in filtered_results) / len(filtered_results) * 1000  # Convert to ms

                # Print row
                print(f"| {num_envs:<12} | {simulator:<12} | {wall_time_avg:<20.2f} | {step_time_avg:<20.6f} |")
    
    # Print footer
    print("+" + "-"*12 + "+" + "-"*15 + "+" + "-"*20 + "+" + "-"*20 + "+")

def print_results_table(results):
    """
    Prints the results in a tabular format as specified.
    """
    print(r"\begin{tabular}{|c|c|c|c|}")
    print(r"    \hline")
    print(r"    Environments & Simulator & Wall Time (ms $\downarrow$) & Step Time (ms $\downarrow$) \\")
    print(r"    \hline")

    for num_envs in NUM_ENVS:
        for simulator in SIMULATORS:
            # Filter results for this condition
            filtered_results = [r for r in results if r['num_envs'] == num_envs and r['simulator'] == simulator]

            # Aggregate metrics (average across trials)
            wall_time_avg = sum(r['wall_time'] for r in filtered_results) / len(filtered_results) * 1000  # Convert to ms
            step_time_avg = sum(r['step_time'] for r in filtered_results) / len(filtered_results) * 1000  # Convert to ms

            # Print rows for Python and JAX simulators
            if simulator == "python":
                print(f"    {num_envs} & Python & {wall_time_avg:.2f} & {step_time_avg:.6f} \\")
            elif simulator == "jax":
                print(f"        & Jax & {wall_time_avg:.2f} & {step_time_avg:.6f} \\")
    
    print(r"    \hline")
    print(r"\end{tabular}")
    

if __name__ == "__main__":
    # run_experiment()
    # Load results from file or use directly if already available
    with open('jax_sim_parallel_results.pkl', 'rb') as f:
        results = pickle.load(f)

    # Print the summary table
    print_results_table_human_readable(results)

    # print the LaTeX table
    print_results_table(results)