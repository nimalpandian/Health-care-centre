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
image = cv2.imread('vestrahorn-mountains-stokksnes-iceland_335224-667.jpg')

# Find the origin of frame E in the image.
cx, cy = find_origin_of_frame_e(image)

if cx is not None and cy is not None:
    print(f"Origin of frame E: ({cx}, {cy})")
else:
    print("Error finding origin coordinates.")
