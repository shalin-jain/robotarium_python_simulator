import jax
import jax.numpy as jnp
from jaxopt import OSQP

def create_robust_barriers(max_num_obstacles=100, max_num_robots=30, d=5, wheel_vel_limit=12.5, base_length=0.105, wheel_radius=0.016,
    projection_distance=0.05, gamma=150, safety_radius=0.12):

    def robust_barriers(dxu, x, obstacles = None):
        # D: Maps wheel velocities to robot's linear and angular velocities (g(x) in system dynamics)
        D = jnp.array([[wheel_radius/2, wheel_radius/2], [-wheel_radius/base_length, wheel_radius/base_length]])
        
        # L: Maps wheel velocities to the rate of change of the robot's projected position (used in barrier function derivative)
        L = jnp.array([[1, 1],[0,projection_distance]]) @ D
        
        # disturb: Defines the set of disturbances (Ψ) as a convex hull
        disturb = jnp.array([[-d, -d, d, d],[-d, d, d, -d]])
        num_disturbs = disturb.shape[1]

        # Max number of constraints based on number of robots and obstacles
        max_num_constraints = (max_num_robots**2-max_num_robots)//2 + max_num_robots*max_num_obstacles + 4*max_num_robots
        
        qp_solver = OSQP(tol=1e-3, maxiter=50)

        num_robots = dxu.shape[1]
        num_obstacles = obstacles.shape[1] if obstacles else 0

        num_constraints = (num_robots**2-num_robots)//2 + num_robots*num_obstacles if num_robots >= 2 else 0
        num_robot_constraints = (num_robots**2-num_robots)//2

        # Os: Set the x,y components of the orientation of the robots
        Os = jnp.vstack([jnp.cos(x[2, :]), jnp.sin(x[2, :])])

        # ps: Get the position of the safety bubble in front of the robot (projected position)
        ps = x[0:2, :] + projection_distance * Os

        # Ms: populate the Ms matrix based on the x,y components of the orientations of the robots and the projection distance (used to calculate h_dot)
        Ms = jnp.vstack([
            jnp.hstack([Os[0], Os[1]]),
            jnp.hstack([projection_distance * Os[1], -projection_distance * Os[0]])
        ])
        
        # MDs: Maps wheel velocities to the rate of change of robots' projected positions
        MDs = (Ms.T @ D).T
        MDs = MDs.at[0, 1::2].set(MDs[1, ::2])
        MDs = MDs.at[1, ::2].set(MDs[0, 1::2])

        A = jnp.zeros([max_num_constraints, 2*max_num_robots])
        
        def robot_pair_constraints(i, j):
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

            # add robot i constraint
            A_ = jnp.zeros([2*max_num_robots])
            A_ = A_.at[2*i].set(h_dot_i[0])
            A_ = A_.at[2*i+1].set(h_dot_i[1])

            # add robot j constraint
            A_ = A_.at[2*j].set(h_dot_j[0])
            A_ = A_.at[2*j+1].set(h_dot_j[1])

            # add inequality value
            b_ = -gamma*(jnp.power(h,3)) - jnp.min(h_dot_i.reshape(1,-1)@disturb, 1) - jnp.min(h_dot_j.reshape(1,-1)@disturb, 1)

            return A_, b_

        robot_pairs = jnp.triu_indices(num_robots, k=1)
        robot_pair_results = jax.vmap(robot_pair_constraints, out_axes=0)(robot_pairs[0], robot_pairs[1])
        A, b = robot_pair_results
        print(A.shape)
        print(b.shape)

        # def obstacle_constraints(i, obstacle):
        #     diff = ps[:, i] - obstacle
        #     h = jnp.sum(jnp.square(diff)) - safety_radius**2
        #     h_dot_i = 2 * diff.T @ MDs[:, 2*i:2*i+2]
        #     return h, h_dot_i


        # if obstacles:
        #     robot_obstacle_pairs = jnp.meshgrid(jnp.arange(num_robots), jnp.arange(num_obstacles))
        #     obstacle_results = jax.vmap(obstacle_constraints)(robot_obstacle_pairs[0].ravel(), obstacles.T)
        #     hs_obstacles, h_dot_is_obstacles = obstacle_results
        #     A_obstacles = jax.vmap(lambda h_i, i: jax.ops.index_update(
        #         jnp.zeros((2*num_robots,)), [2*i, 2*i+1], h_i
        #     ))(h_dot_is_obstacles, robot_obstacle_pairs[0].ravel())
        #     b_obstacles = -gamma * jnp.power(hs_obstacles, 3) - jnp.min(h_dot_is_obstacles @ disturb, axis=1)
        

        A_velocity = jnp.vstack([jnp.eye(2*num_robots), -jnp.eye(2*num_robots)])
        A_velocity = jnp.concatenate([A_velocity, jnp.zeros((A_velocity.shape[0], 2*max_num_robots-2*num_robots))], axis=-1)
        b_velocity = jnp.full((4*num_robots,), wheel_vel_limit)
 
        # if obstacles:
        #     A = jnp.vstack([A_robots, A_obstacles, A_velocity])
        #     b = jnp.concatenate([b_robots, b_obstacles, b_velocity])
        # else:
        A = jnp.vstack([A, A_velocity])
        b = jnp.concatenate([b, b_velocity.reshape(-1,1)])

        L_all = jnp.kron(jnp.eye(num_robots), L)
        dxu = jnp.linalg.inv(D) @ dxu
        v_hat = jnp.reshape(dxu ,(2*num_robots,1), order='F')

        P = 2 * L_all.T @ L_all
        q = -2 * v_hat.T @ L_all.T @ L_all

        # print(P.shape)
        # print(q.shape)

        # Solve QP program
        qp_solution = qp_solver.run(params_obj=(P, q.squeeze()), params_ineq=(A[:, :2*num_robots], b.squeeze()))
        vnew = qp_solution.params.primal

        dxu_new = D @ vnew.reshape((2, num_robots), order='F')
        return dxu_new

    return robust_barriers
