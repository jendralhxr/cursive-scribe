import matplotlib.pyplot as plt
from skimage.morphology import skeletonize
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
props = measure.regionprops(labeled_image)

# Memisahkan Huruf Utama dan Titik
main_components = np.zeros_like(binary)
dot_components = np.zeros_like(binary)

for prop in props:
    if prop.area > 50:
        main_components[labeled_image == prop.label] = 1
    elif prop.area > 5:
        min_distance = np.min([np.linalg.norm(np.array(prop.centroid) - np.array(main_prop.centroid))
                               for main_prop in props if main_prop.area > 45])
        if min_distance < 27:
            dot_components[labeled_image == prop.label] = 1

# Gabungkan huruf utama dan titik yang valid
cleaned_binary = main_components + dot_components

# Skeletonization
skeleton = skeletonize(cleaned_binary)

# Fungsi untuk Menemukan Titik Penting pada Skeleton
def find_endpoints(skel):
    endpoints = []
    intersections = []
    turns = []

    for i in range(1, skel.shape[0] - 1):
        for j in range(1, skel.shape[1] - 1):
            if skel[i, j]:  # Jika pixel termasuk dalam skeleton
                neighborhood = skel[i-1:i+2, j-1:j+2]  # Ambil 3x3 sekitar pixel
                num_neighbors = np.sum(neighborhood) - 1  # Kurangi pusatnya

                if num_neighbors == 1:  # Endpoint (hanya 1 tetangga)
                    endpoints.append((i, j))
                elif num_neighbors >= 3:  # Persimpangan (tiga atau lebih tetangga)
                    intersections.append((i, j))
                else:
                    # Mendeteksi belokan tajam 90°
                    horizontal = skel[i, j-1] + skel[i, j+1]  # Cek kiri & kanan
                    vertical = skel[i-1, j] + skel[i+1, j]  # Cek atas & bawah
                    if horizontal == 1 and vertical == 1:
                        turns.append((i, j))  # Belokan tajam

    return np.array(endpoints), np.array(intersections), np.array(turns)

# Menentukan Fitur Setiap Huruf
letter_features = {}
for region_label in range(1, num_labels + 1):
    single_letter = (labeled_image == region_label)
    letter_skeleton = skeleton * single_letter

    endpoints, intersections, turns = find_endpoints(letter_skeleton)

    if len(endpoints) >= 2:
        start = max(endpoints, key=lambda x: x[1])  # Titik paling kanan (RTL)
        end = min(endpoints, key=lambda x: x[1])  # Titik paling kiri
    else:
        start, end = None, None

    # Menggunakan Bounding Box untuk titik tengah
    props = measure.regionprops(single_letter.astype(int))[0]
    minr, minc, maxr, maxc = props.bbox  # Bounding box (min & max koordinat)
    centroid = ((minr + maxr) / 2, (minc + maxc) / 2)  # Titik tengah bounding box

    letter_features[region_label] = {
        "start": start,
        "end": end,
        "intersections": intersections,
        "turns": turns,
        "centroid": centroid
    }

# Plot hasil
fig, ax = plt.subplots(figsize=(10, 5))
ax.imshow(skeleton, cmap=plt.cm.gray)

for letter, features in letter_features.items():
    if features["start"] is not None and len(features["start"]) > 0:
        start_y, start_x = features["start"]
        ax.plot(start_x, start_y, marker='o', markersize=5, color='blue', label="Awal Huruf" if 'Awal Huruf' not in ax.get_legend_handles_labels()[1] else "")
    if features["end"] is not None and len(features["end"]) > 0:
        end_y, end_x = features["end"]
        ax.plot(end_x, end_y, marker='o', markersize=5, color='yellow', label="Akhir Huruf" if 'Akhir Huruf' not in ax.get_legend_handles_labels()[1] else "")
    if len(features["intersections"]) > 0:
        for inter_y, inter_x in features["intersections"]:
            ax.plot(inter_x, inter_y, marker='o', markersize=3, color='green', label="Persimpangan" if 'Persimpangan' not in ax.get_legend_handles_labels()[1] else "")
    if len(features["turns"]) > 0:
        for turn_y, turn_x in features["turns"]:
            ax.plot(turn_x, turn_y, marker='o', markersize=3, color='cyan', label="Belokan 90°" if 'Belokan 90°' not in ax.get_legend_handles_labels()[1] else "")
    if features["centroid"] is not None:
        cent_y, cent_x = features["centroid"]
        ax.plot(cent_x, cent_y, marker='o', markersize=5, color='red', label="Centroid Huruf" if 'Centroid Huruf' not in ax.get_legend_handles_labels()[1] else "")

ax.set_title("Titik Penting pada Skeleton Huruf")
ax.axis("off")
plt.legend()
plt.show()