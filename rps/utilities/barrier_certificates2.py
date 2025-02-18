from cvxopt import matrix, solvers
import quadprog as solver2
import numpy as np
import time

solvers.options['show_progress'] = False
solvers.options['reltol'] = 1e-2 
solvers.options['feastol'] = 1e-2
solvers.options['maxiters'] = 50 

def create_robust_barriers(max_num_obstacles = 100, max_num_robots = 30, d = 5, wheel_vel_limit = 12.5, base_length = 0.105, wheel_radius = 0.016,
    projection_distance =0.05, gamma = 150, safety_radius = 0.12):

    # D: Maps wheel velocities to robot's linear and angular velocities (g(x) in system dynamics)
    D = np.matrix([[wheel_radius/2, wheel_radius/2], [-wheel_radius/base_length, wheel_radius/base_length]])
    
    # L: Maps wheel velocities to the rate of change of the robot's projected position (used in barrier function derivative)
    L = np.matrix([[1, 1],[0,projection_distance]])* D
    
    # disturb: Defines the set of disturbances (Ψ) as a convex hull
    disturb = np.matrix([[-d, -d, d, d],[-d, d, d, -d]])
    num_disturbs = np.size(disturb[1,:])

    # Max number of constraints based on number of robots and obstacles
    max_num_constraints = (max_num_robots**2-max_num_robots)//2 + max_num_robots*max_num_obstacles + 4*max_num_robots
    
    # A: Matrix to store the QP constraints
    A = np.matrix(np.zeros([max_num_constraints, 2*max_num_robots]))
    # b: Vector to store the QP constraints
    b = np.matrix(np.zeros([max_num_constraints, 1]))
    # Os: Matrix to store the orientations of robots
    Os = np.matrix(np.zeros([2,max_num_robots]))
    # ps: Matrix to store the projected positions of the robots
    ps = np.matrix(np.zeros([2,max_num_robots]))
    # Ms: Matrix used to calculate the derivatives of the barrier function
    Ms = np.matrix(np.zeros([2,2*max_num_robots]))

    def robust_barriers(dxu, x, obstacles):
        num_robots = np.size(dxu[0,:])

        if obstacles.size != 0:
            num_obstacles = np.size(obstacles[0,:])
        else:
            num_obstacles = 0

        if(num_robots < 2):
            temp = 0
        else:
            temp = (num_robots**2-num_robots)//2 # number of unique robot pairs

        if num_robots == 0:
            return []

        # num_constraints: Number of unique robot pairs + robot obstacle pairs (defines the number of barrier function constraints)
        num_constraints = temp + num_robots*num_obstacles 

        # Initialize A matrix for the QP
        A[0:num_constraints, 0:2*num_robots] = 0
        # Os: Set the x,y components of the orientation of the robots
        Os[0, 0:num_robots] = np.cos(x[2, :]) # x component of current orientation
        Os[1, 0:num_robots] = np.sin(x[2, :]) # y component of current orientation

        # ps: Get the position of the safety bubble in front of the robot (projected position)
        ps[:, 0:num_robots] = x[0:2, :] + projection_distance*Os[:, 0:num_robots]

        # Ms: populate the Ms matrix based on the x,y components of the orientations of the robots and the projection distance (used to calculate h_dot)
        Ms[0, 0:2*num_robots:2] = Os[0, 0:num_robots]
        Ms[0, 1:2*num_robots:2] = Os[1, 0:num_robots]
        Ms[1, 1:2*num_robots:2] = projection_distance*Os[0, 0:num_robots]
        Ms[1, 0:2*num_robots:2] = -projection_distance*Os[1, 0:num_robots]
        
        # MDs: Maps wheel velocities to the rate of change of robots' projected positions
        MDs = (Ms.T * D).T
        temp = np.copy(MDs[1, 0:2*num_robots:2])
        MDs[1, 0:2*num_robots:2] = MDs[0, 1:2*num_robots:2]
        MDs[0, 1:2*num_robots:2] = temp
        
        count = 0
        # Loop through all robots to generate constraints for every pair of robots
        for i in range(num_robots-1):
            # diffs: difference between safety bubble centers of robots (pi(xi) - pj(xj) in barrier function)
            diffs = ps[:,i] - ps[:, i+1:num_robots] 
            
            # hs: value of the barrier function (h(x)) for each pair of robots (‖pi(xi)− pj(xj)‖² - δ²)
            hs = np.sum(np.square(diffs),0) - safety_radius**2 

            # h_dot_is: derivative of the barrier function with respect to robot i's control input (∇h(x')*g(x')*u(x'))
            h_dot_is = 2*diffs.T*MDs[:,2*i:2*i+2] 
            
            # h_dot_js: derivative of the barrier function with respect to robot j's control input (∇h(x')*g(x')*u(x'))
            h_dot_js = np.matrix(np.zeros((2,num_robots - (i+1))))
            h_dot_js[0, :] = -np.sum(2*np.multiply(diffs, MDs[:,2*(i+1):2*num_robots:2]), 0)
            h_dot_js[1, :] = -np.sum(2*np.multiply(diffs, MDs[:,2*(i+1)+1:2*num_robots:2]), 0)

            new_constraints = num_robots - i - 1
            
            # A: Populating the A matrix with the derivatives of the barrier function for each robot pair
            A[count:count+new_constraints, (2*i):(2*i+2)] = h_dot_is
            A[range(count,count+new_constraints), range(2*(i+1),2*num_robots,2)] = h_dot_js[0,:]
            A[range(count,count+new_constraints), range(2*(i+1)+1,2*num_robots,2)] = h_dot_js[1,:]
            
            # b: Populating the b vector with the safety margin, disturbance compensation and the function α(h(x'))
            b[count:count+new_constraints] = -gamma*(np.power(hs,3)).T - np.min(h_dot_is*disturb,1) - np.min(h_dot_js.T*disturb,1)
            count += new_constraints

        # If obstacles are present, generate constraints for each robot-obstacle pair
        if obstacles.size != 0:
            for i in range(num_robots):
                # diffs: difference between the robot's projected position and the obstacle position (pi(xi) - obstacle)
                diffs = (ps[:, i] - obstacles)
                # h: Value of the barrier function for the robot and obstacle pair
                h = np.sum(np.square(diffs),0) - safety_radius**2
                # h_dot_i: derivative of the barrier function for the robot and obstacle pair
                h_dot_i = 2*diffs.T*MDs[:,2*i:2*i+2]
                
                # A: populating A matrix with derivatives of the barrier function for robot obstacle pairs
                A[count:count+num_obstacles,(2*i):(2*i+2)] = h_dot_i
                # b: populating b vector with safety margin and disturbance compensation for robot obstacle pairs
                b[count:count+num_obstacles] = -gamma*(np.power(h,3)).T - np.min(h_dot_i*disturb, 1)
                count = count + num_obstacles
        
        # Adding Upper Bounds on Wheel Velocities
        A[count:count+2*num_robots,0:2*num_robots] = -np.eye(2*num_robots)
        b[count:count+2*num_robots] = -wheel_vel_limit
        count += 2*num_robots

        # Adding Lower Bounds on Wheel Velocities
        A[count:count+2*num_robots,0:2*num_robots] = np.eye(2*num_robots)
        b[count:count+2*num_robots] = -wheel_vel_limit
        count += 2*num_robots

        # L_all: Matrix L that is repeated for all robots
        L_all = np.kron(np.eye(num_robots), L)
        # dxu: Convert user input to differential drive velocities (u in the system dynamics)
        dxu = np.linalg.inv(D)*dxu
        # vhat: Reshape dxu for use in the QP [15]
        vhat = np.matrix(np.reshape(dxu ,(2*num_robots,1), order='F'))

        # H: Matrix representing the objective function of the QP (minimizing deviation from nominal input)
        H = 2*L_all.T*L_all
        # f: Vector representing the objective function of the QP [15, 16]
        f = np.transpose(-2*np.transpose(vhat)*np.transpose(L_all)*L_all)

        # The QP being solved is:
        # minimize    1/2 * vnew^T * H * vnew  + f^T * vnew  
        # subject to   A * vnew  <= b
        # Where:
        # - vnew is a vector of the optimized control inputs (wheel velocities) for all robots
        # - H is a matrix that penalizes deviations from the nominal control input.
        # - f is a vector that includes the nominal control input.
        # - A is a matrix that encodes the barrier function constraints and the wheel velocity limits.
        # - b is a vector that encodes the safety margin, disturbance compensation, and the wheel velocity limits.
        #  The QP is formulated to find the control input vnew that minimizes deviation from the nominal control, 
        # while ensuring safety by satisfying the barrier function constraints. The constraints are in the form of A*vnew <= b
        # where each row of A and b represents a barrier constraint of the form ∇h(x')>(f(x') + g(x')u(x')) -α(h(x'))-min∇h(x')>Ψ(x').
        # The QP ensures the robots follow user inputs while avoiding collisions by implementing the CBF as constraints in the optimization program.
        # ---End of QP Formulation Comments---
        
        # Solve QP program: This minimizes the difference between the nominal control and the CBF-safe control while satisfying constraints.
        vnew = solver2.solve_qp(H, -np.squeeze(np.array(f)), A[0:count,0:2*num_robots].T, np.squeeze(np.array(b[0:count])))

        # Reshape vnew to usable velocity command
        dxu = np.reshape(vnew, (2, num_robots), order='F')
        # Convert differential drive velocities to wheel velocities
        dxu = D*dxu
        return dxu
    
    return robust_barriers