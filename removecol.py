#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 20 09:10:33 2024

@author: rdx
"""

from PIL import Image
import numpy as np
import sys

# Step 1: Open the image
image_path = sys.argv[1]
image = Image.open(image_path)

# Step 2: Convert the image to a NumPy array
image_array = np.array(image)

# Step 3: Remove a single column
column_to_remove = int(sys.argv[3])  # For example, remove the 50th column
image_array = np.delete(image_array, column_to_remove, axis=1)

# Step 4: Convert the modified array back to an image
modified_image = Image.fromarray(image_array)

# Step 5: Save or display the modified image
modified_image.save(sys.argv[2])
modified_image.show()
