import numpy as np
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

# Example usage:

robot = Robot(l0=0.5, l1=1, l2=0.75, l3=0.5)

# Compute the pose of the end effector for the given joint angles.
theta = np.array([0, np.pi/2, np.pi/4])
T03 = robot.forward_kinematics(theta)

# Print the pose of the end effector.
print(T03)
