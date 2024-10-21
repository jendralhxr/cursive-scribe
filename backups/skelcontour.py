#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 10:12:23 2024
@author: jendralhxr
"""

import cv2 as cv
import numpy as np
import sys

def skeletonize(image, neighbor_distance=1):
    """
    Skeletonize an image by polling neighboring pixels at a given distance.
    
    :param image: Input image (grayscale or color)
    :param neighbor_distance: Distance for polling neighbors (default is 1, for a 3x3 grid)
    :return: Skeletonized image
    """
    # Convert the image to grayscale if it's not already
    if len(image.shape) == 3:
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    else:
        gray = image

    # Apply a binary threshold to get a binary image
    _, binary = cv.threshold(gray, 127, 255, cv.THRESH_BINARY)

    # Find contours in the binary image
    contours, _ = cv.findContours(gray, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    # Create an empty skeleton image
    skeleton = np.zeros_like(gray)

    # Define the neighborhood size based on the distance
    # e.g., distance 1 -> 3x3, distance 2 -> 5x5, etc.
    neighbor_size = 2 * neighbor_distance + 1

    # Polling method: iteratively remove contour points to reduce to skeleton
    for contour in contours:
        for i in range(len(contour)):
            point = contour[i][0]
            x, y = point[0], point[1]

            # Get the neighborhood of the pixel based on the specified distance
            # Ensure we don't go out of bounds
            x_start = max(0, x - neighbor_distance)
            x_end = min(binary.shape[1], x + neighbor_distance + 1)
            y_start = max(0, y - neighbor_distance)
            y_end = min(binary.shape[0], y + neighbor_distance + 1)

            # Extract the neighborhood region
            neighborhood = binary[y_start:y_end, x_start:x_end]

            # Poll neighbors (count the number of non-zero pixels in the neighborhood)
            if cv.countNonZero(neighborhood) > 2:  # Customize this threshold for thinning
                skeleton[y, x] = 255  # Set pixel in the skeleton

    return skeleton

# Load the image
image = cv.imread(sys.argv[1])

# Perform skeletonization with a larger neighborhood
neighbor_distance = 2  # This will check a 5x5 neighborhood
skeleton_image = skeletonize(image, neighbor_distance)

# Show the result
cv.imshow('Skeleton', skeleton_image)
cv.waitKey(0)
cv.destroyAllWindows()
