import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from skimage.morphology import skeletonize
from skimage import io, color, filters

# 1. Load your image
# Replace 'your_image.jpg' with your actual file path
path = 'p01v1-line2.png' 
img = io.imread(path)

# 2. Convert to grayscale and Binarize
# We need the object to be white (1) and background black (0)
gray_img = color.rgb2gray(img)
plt.imshow(gray_img)

thresh = filters.threshold_otsu(gray_img)
binary = gray_img < thresh
plt.imshow(binary, cmap='gray')

# NOTE: If your object is darker than the background, 
# you might need to flip the logic: binary = gray_img < thresh

# 3. Compute Skeleton and Distance Transform
skeleton = skeletonize(binary)
plt.imshow(skeleton, cmap='gray')

distance_map = ndi.distance_transform_edt(binary)
plt.imshow(distance_map, cmap='nipy_spectral')

