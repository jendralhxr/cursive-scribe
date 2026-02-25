#import library
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize, remove_small_objects
from skimage import io, color, filters, measure
import numpy as np

# Membaca dan Mengonversi Gambar
image_path = "kayu.png"
image = io.imread(image_path)

# Check if the image has an alpha channel and remove it if present
if image.shape[2] == 4:
    image = image[:, :, :3]  # Keep only the first 3 channels (RGB)

# Convert to grayscale
gray = color.rgb2gray(image)

# Binarisasi Gambar dengan Thresholding
thresh = filters.threshold_otsu(gray)
binary = gray < thresh

# Melabeli dan Menganalisis Komponen Gambar
labeled_image, num_labels = measure.label(binary, return_num=True, connectivity=2)

# Memisahkan Huruf Utama dan Titik
props = measure.regionprops(labeled_image)

main_components = np.zeros_like(binary)
dot_components = np.zeros_like(binary)

# Memilah Huruf dan Titik Berdasarkan Ukurannya
for prop in props:
    if prop.area > 50:  # Huruf utama ditentukan berdasarkan threshold yang disesuaikan dengan ukuran huruf
        main_components[labeled_image == prop.label] = 1
    elif prop.area > 5:  # Titik huruf atau noise
        # Cek apakah titik ini dekat dengan huruf utama
        min_distance = np.min([np.linalg.norm(np.array(prop.centroid) - np.array(main_prop.centroid))
                               for main_prop in props if main_prop.area > 45])
        if min_distance < 27:  # Jika dekat dengan huruf utama, kemungkinan besar titik huruf
            dot_components[labeled_image == prop.label] = 1

# Gabungkan huruf utama dan titik yang valid
cleaned_binary = main_components + dot_components

# skeletonization
skeleton = skeletonize(cleaned_binary)

# Hasil
fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True)
ax = axes.ravel()

ax[0].imshow(binary, cmap=plt.cm.gray)
ax[0].set_title('Original Binary Image', fontsize=15)
ax[0].axis('off')

ax[1].imshow(cleaned_binary, cmap=plt.cm.gray)
ax[1].set_title('Cleaned Binary (No Noise)', fontsize=15)
ax[1].axis('off')

ax[2].imshow(skeleton, cmap=plt.cm.gray)
ax[2].set_title('Skeletonized Image', fontsize=15)
ax[2].axis('off')

fig.tight_layout()
plt.show()