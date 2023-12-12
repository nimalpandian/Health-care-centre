import numpy as np
import cv2
from google.colab.patches import cv2_imshow  # For displaying images in Colab

def find_origin_of_frame_e(image_path):
    try:
        # Read the image from the file.
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Image not found: {image_path}")
    except Exception as e:
        print(f"Error: {e}")
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
        # Check if the contour is convex.
        if cv2.isContourConvex(contour):
            frame_e_contour = contour
            break

    if frame_e_contour is not None:
        # Find the centroid of the contour.
        centroid = cv2.moments(frame_e_contour)
        
        # Ensure the moment has a non-zero area to avoid division by zero.
        if centroid['m00'] != 0:
            cx = int(centroid['m10'] / centroid['m00'])
            cy = int(centroid['m01'] / centroid['m00'])
            
            return cx, cy
        else:
            print("Error: Contour has zero area.")
            return None
    else:
        print("Error: No contour found.")
        return None

# Example usage:
image_path = 'vestrahorn-mountains-stokksnes-iceland_335224-667.jpg'
origin_coordinates = find_origin_of_frame_e(image_path)

if origin_coordinates is not None:
    print(f"Origin of frame E: ({origin_coordinates[0]}, {origin_coordinates[1]})")
else:
    print("Error finding origin coordinates.")
