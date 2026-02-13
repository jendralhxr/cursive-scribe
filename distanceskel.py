import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from skimage.morphology import skeletonize
from skimage import io, color, filters, morphology

# 1. Load your image
path = 'p01v1-line2.png' 
img = io.imread(path)

# 2. Convert to grayscale and Binarize
gray_img = color.rgb2gray(img)
plt.imshow(gray_img)

# thresholding, coba juga local thresholding macam
# thresh_local = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY, 11, 2)
thresh = filters.threshold_otsu(gray_img)
binary = gray_img < thresh
plt.imshow(binary, cmap='gray')

# 3. Compute Skeleton and Distance Transform
skeleton = skeletonize(binary)
plt.imshow(skeleton, cmap='gray')

from skimage import feature

# Find edges using Canny
edges = feature.canny(binary)
plt.imshow(edges, cmap='gray')

# edges using erosion
eroded = morphology.binary_erosion(binary)
edges = binary ^ eroded  # These are the pixels at the very boundary

# 3. Calculate Distance to Edges for ALL foreground pixels
# We create a mask where edges are 0 and everything else is 1.
# distance_transform_edt calculates the distance to the nearest 0.
edge_mask = np.ones_like(binary, dtype=float)
edge_mask[edges] = 0

# Compute the distance transform
distance_to_edges = ndi.distance_transform_edt(edge_mask)

# 4. Mask the result so we only see distances INSIDE the object
# (This sets the background pixels to 0)
foreground_distance_map = distance_to_edges * binary
plt.contour(skeleton, [0.5], colors='white', linewidths=0.1) # artistic rendering
plt.imshow(foreground_distance_map, cmap='nipy_spectral')


# harddo
combined_view = foreground_distance_map.copy()
combined_view[skeleton] = np.max(foreground_distance_map) * 1.5 
plt.imshow(combined_view, cmap='nipy_spectral')

# 5. Visualization
plt.imshow(foreground_distance_map, cmap='nipy_spectral')
#plt.axis('off')
#plt.colorbar(label='Distance to nearest edge (pixels)')
#plt.title('Distance Transform: All Foreground Pixels to Edges')
