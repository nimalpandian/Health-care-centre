import numpy as np
import cv2

def find_origin_of_frame_e(image):
    """Finds the origin of frame E in the given image.

    Args:
        image: A numpy array representing the image.

    Returns:
        A tuple of (x, y) coordinates representing the origin of frame E.
    """
    if image is None:
        print("Error: Image not loaded.")
        return None

    # Convert the image to grayscale.
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Find the edges in the image using the Canny edge detector.
    edges = cv2.Canny(gray, 50, 150)

    # Find the contours in the image using the findContours() function.
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the contour that corresponds to frame E.
    frame_e_contour = None
    for contour in contours:
        # Check if the contour is a rectangle.
        epsilon = 0.05 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        if len(approx) == 4:
            frame_e_contour = contour
            break

    if frame_e_contour is not None:
        # Find the centroid of the contour.
        centroid = cv2.moments(frame_e_contour)

        # Ensure the moment has a non-zero area to avoid division by zero.
        if centroid['m00'] != 0:
            cx = centroid['m10'] / centroid['m00']
            cy = centroid['m01'] / centroid['m00']

            return cx, cy
        else:
            print("Error: Contour has zero area.")
            return None
    else:
        print("Error: No contour found.")
        return None

# Example usage:
image = cv2.imread('image.jpg')

# Find the origin of frame E in the image.
cx, cy = find_origin_of_frame_e(image)

if cx is not None and cy is not None:
    print(f"Origin of frame E: ({cx}, {cy})")
else:
    print("Error finding origin coordinates.")
:
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

robot = Robot(l0=10, l1=10, l2=10, l3=10)

# Compute the pose of the end effector for the given joint angles.
theta = np.array([0, np.pi/2, np.pi/4])
T03 = robot.forward_kinematics(theta)

# Print the pose of the end effector.
print(T03) 
