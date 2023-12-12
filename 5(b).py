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

def find_corners_of_frame_e(image):
    """Finds the four corners of frame E in the given image.

    Args:
        image: A numpy array representing the image.

    Returns:
        An array of four tuples representing the corners of frame E.
    """
    if image is None:
        print("Error: Image not loaded.")
        return None

    # Convert the image to grayscale.
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Find the edges in the image using the Canny edge detector.
    edges = cv2.Canny(gray, 50, 150)

    # Find the contours in the image using the findContours() function.
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

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
        # Get the corners of the rectangle.
        corners = cv2.boxPoints(cv2.minAreaRect(frame_e_contour))
        corners = np.int0(corners)

        return corners
    else:
        print("Error: No contour found for frame E.")
        return None

def compute_transformation_between_frames_c2_and_e(image_c2, image_e):
    """Computes the transformation between frames C2 and E in the given images.

    Args:
        image_c2: A numpy array representing the image of frame C2.
        image_e: A numpy array representing the image of frame E.

    Returns:
        A 3x3 transformation matrix representing the transformation between frames C2 and E.
    """
    # Find the origin of frame E in the image of frame E.
    origin_of_frame_e = find_origin_of_frame_e(image_e)

    # Find the four corners of frame E in the image of frame E.
    corners_of_frame_e = find_corners_of_frame_e(image_e)

    if origin_of_frame_e is not None and corners_of_frame_e is not None:
        # Find the transformation matrix that maps the four corners of frame E in the image of frame E to the origin of frame E in the image of frame C2.
        homography, _ = cv2.findHomography(corners_of_frame_e, np.array([origin_of_frame_e]), cv2.RANSAC, 5.0)

        return homography
    else:
        print("Error: Unable to compute transformation.")
        return None

# Example usage:
image_c2 = cv2.imread('vestrahorn-mountains-stokksnes-iceland_335224-667.jpg')
image_e = cv2.imread('download.jpeg')

homography = compute_transformation_between_frames_c2_and_e(image_c2, image_e)

if homography is not None:
    print(f"Transformation matrix between frames C2 and E:\n{homography}")
else:
    print("Error: Transformation matrix not computed.")
