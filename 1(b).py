import numpy as np
from scipy.optimize import minimize

class Robot:
    def __init__(self, l0, l1, l2, l3):
        self.l0 = l0
        self.l1 = l1
        self.l2 = l2
        self.l3 = l3

    def forward_kinematics(self, theta):
        """Computes the forward kinematics of the robot.

        Args:
            theta: A 3x1 vector of joint angles.

        Returns:
            A 4x4 transformation matrix representing the pose of the end effector.
        """

        T01 = np.array([[np.cos(theta[0]), -np.sin(theta[0]), 0, 0],
                        [np.sin(theta[0]), np.cos(theta[0]), 0, 0],
                        [0, 0, 1, self.l0],
                        [0, 0, 0, 1]])

        T12 = np.array([[np.cos(theta[1]), -np.sin(theta[1]), 0, 0],
                        [np.sin(theta[1]), np.cos(theta[1]), 0, 0],
                        [0, 0, 1, self.l1],
                        [0, 0, 0, 1]])

        T23 = np.array([[np.cos(theta[2]), -np.sin(theta[2]), 0, 0],
                        [np.sin(theta[2]), np.cos(theta[2]), 0, 0],
                        [0, 0, 1, self.l2],
                        [0, 0, 0, 1]])

        T03 = T01 @ T12 @ T23

        return T03

    def inverse_kinematics(self, p):
        """Computes the inverse kinematics of the robot.

        Args:
            p: A 3x1 vector representing the desired position of the end effector.

        Returns:
            A 3x1 vector of joint angles that achieve the desired position, or None if no solution exists.
        """

        def objective(theta):
            T03 = self.forward_kinematics(theta)
            p_hat = T03[:3, 3]
            error = np.linalg.norm(p_hat - p)
            return error

        # Set initial guess for theta
        theta_guess = np.array([0.0, np.pi/2, np.pi/4])

        # Minimize the error between desired and achieved position
        res = minimize(objective, theta_guess, method="L-BFGS-B")

        # Check if solution found
        if res.success:
            return res.x
        else:
            return None

# Example usage:

robot = Robot(l0=0.5, l1=1, l2=0.75, l3=0.5)

# Desired position of the end effector
p = np.array([0.5, 0.75, 1.0])

# Compute joint angles to achieve the desired position
theta = robot.inverse_kinematics(p)

# Check if solution found
if theta is not None:
    print(f"Joint angles: {theta}")
    T03 = robot.forward_kinematics(theta)
    print(f"Achieved end effector position: {T03[:3, 3]}")
else:
    print("No solution found for the desired position.")
