import jax
import jax.numpy as jnp
from jaxopt import BoxCDQP
# jax.config.update("jax_enable_x64", True)

def create_robust_barriers(max_num_obstacles=100, max_num_robots=30, d=5, wheel_vel_limit=12.5, base_length=0.105, wheel_radius=0.016,
    projection_distance=0.05, gamma=150, safety_radius=0.12):
    """
    Creates barrier certificate function for collision avoidance.
    Based on "Robust Barrier Functions for a Fully Autonomous, Remotely Accessible Swarm-Robotics Testbed 
    https://ieeexplore.ieee.org/document/9029886"
    

    Args:
        max_num_obstacles: (int) maximum number of obstacles considered
        max_num_robots: (int) maximum number of robots considered
        d (float): constant for evaluating disturbance convex hull at extrema
        wheel_vel_limit (float): maximum wheel velocities
        base_length (float): distance between wheels
        wheel_radius (float): radius of wheels
        projection_distance (float): distance from wheel base to centroid of safety bubble
        gamma (float): barrier function sensitivity
        safety_radius (float): minimum distance between robots
    
    Returns:
        (function) robust_barriers
    """

    def robust_barriers(dxu, x, obstacles = None):
        """
        Solves quadratic program for enforcing barrier certificates specified by the parameters in create_robust_barriers.

        Args:
            dxu: (jnp.ndarray) nominal control inputs
            x: (jnp.ndarray) robot poses
            obstacles: (Optional[jnp.ndarray]) obstacles to avoid

        Returns:
            (jnp.ndarray) modified control inputs s.t. min ||dxu-dxu_new|| and robots remain in the safe set
        """
        # D: Maps wheel velocities to robot's linear and angular velocities (g(x) in system dynamics)
        D = jnp.array([[wheel_radius/2, wheel_radius/2], [-wheel_radius/base_length, wheel_radius/base_length]])
        
        # L: Maps wheel velocities to the rate of change of the robot's projected position (used in barrier function derivative)
        L = jnp.array([[1, 0],[0,projection_distance]]) @ D
        
        # disturb: Defines the set of disturbances (Ψ) as a convex hull
        disturb = jnp.array([[-d, -d, d, d],[-d, d, d, -d]])
        
        # initialize QP Solver
        qp_solver = BoxCDQP(tol=1e-6, maxiter=500)

        num_robots = dxu.shape[1]
        num_obstacles = obstacles.shape[1] if obstacles else 0
        num_constraints = (num_robots**2-num_robots)//2 + num_robots*num_obstacles if num_robots >= 2 else 0
        num_robot_constraints = (num_robots**2-num_robots)//2

        # x,y components of the orientation of the robots
        Os = jnp.vstack([jnp.cos(x[2, :]), jnp.sin(x[2, :])])
        # print(Os)

        # position of the safety bubble in front of the robot (projected position)
        ps = x[0:2, :] + projection_distance * Os

        # x,y components of the orientations of the robots and the projection distance (used to calculate h_dot)
        Ms = jnp.zeros((2, 2 * num_robots))
        Ms = Ms.at[0, 0:2*num_robots:2].set(Os[0, 0:num_robots])
        Ms = Ms.at[0, 1:2*num_robots:2].set(Os[1, 0:num_robots])
        Ms = Ms.at[1, 1:2*num_robots:2].set(projection_distance * Os[0, 0:num_robots])
        Ms = Ms.at[1, 0:2*num_robots:2].set(-projection_distance * Os[1, 0:num_robots])
        
        # maps wheel velocities to the rate of change of robots' projected positions
        MDs = (Ms.T @ D).T
        temp = MDs[1, 0:2*num_robots:2]
        MDs = MDs.at[1, 0:2*num_robots:2].set(MDs[0, 1:2*num_robots:2])
        MDs = MDs.at[0, 1:2*num_robots:2].set(temp)
        
        def robot_pair_constraints(i, j):
            """
            Helper function to generate constratint between robots i and j
            """

            diff = ps[:, i] - ps[:, j]
            diff = diff.reshape(-1,1)
            h = jnp.sum(jnp.square(diff)) - safety_radius**2

            # constraint for i
            MDs_i = jnp.concatenate((MDs[:, 2*i].reshape(-1, 1), MDs[:, 2*i+1].reshape(-1,1)), axis=-1)
            h_dot_i = 2 * diff.T @ MDs_i
            h_dot_i = h_dot_i.squeeze()
            
            # constraint for j
            MDs_j = jnp.concatenate((MDs[:, 2*j].reshape(-1, 1), MDs[:, 2*j+1].reshape(-1,1)), axis=-1)
            h_dot_j = -2 * diff.T @ MDs_j
            h_dot_j = h_dot_j.squeeze()

            # create row in A for robot i constraint
            A_ = jnp.zeros([2*max_num_robots])

            # add robot i term
            A_ = A_.at[2*i].set(h_dot_i[0])
            A_ = A_.at[2*i+1].set(h_dot_i[1])

            # add robot j term
            A_ = A_.at[2*j].set(h_dot_j[0])
            A_ = A_.at[2*j+1].set(h_dot_j[1])

            # add corresponding inequality value
            b_ = -gamma*(jnp.power(h,3)) - jnp.min(h_dot_i.reshape(1,-1) @ disturb, 1) - jnp.min(h_dot_j.reshape(1,-1) @ disturb, 1)

            return A_, b_

        # generate inter-robot constraints
        robot_pairs = jnp.triu_indices(num_robots, k=1)
        robot_pair_results = jax.vmap(robot_pair_constraints, out_axes=0)(robot_pairs[0], robot_pairs[1])
        A, b = robot_pair_results
        
        # wheel velocity constraints
        A_velocity = jnp.vstack([-jnp.eye(2*num_robots), jnp.eye(2*num_robots)])
        A_velocity = jnp.concatenate([A_velocity, jnp.zeros((A_velocity.shape[0], 2*max_num_robots-2*num_robots))], axis=-1)
        b_velocity = jnp.full((4*num_robots,), -wheel_vel_limit)

        # final inequality constraint Ax >= b
        A = jnp.vstack([A, A_velocity])
        b = jnp.concatenate([b, b_velocity.reshape(-1,1)])

        # initialize objective 1/2*x^TQx + cx
        L_all = jnp.kron(jnp.eye(num_robots), L)
        dxdd = jnp.linalg.inv(D) @ dxu
        v_hat = jnp.reshape(dxdd ,(2*num_robots,1), order='F')
        Q = 2 * L_all.T @ L_all
        c = 2 * (v_hat.T @ L_all.T @ L_all)

        # solve QP
        qp_solution = qp_solver.run(v_hat.squeeze(), params_obj=(Q, -c.squeeze()), params_ineq=(b.squeeze(), jnp.full(b.squeeze().shape, jnp.inf)))
        vnew = qp_solution.params
        dxu_new = D @ vnew.reshape((2, num_robots), order='F')

        return dxu_new

    return robust_barriers
