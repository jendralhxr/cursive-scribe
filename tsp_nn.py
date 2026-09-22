import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize
from skimage import io, color, filters
from skimage.measure import label, regionprops
from scipy.spatial import distance

# Load Image
image_path = "kayu.png"
image = io.imread(image_path)

# Convert to grayscale if needed
if image.shape[-1] == 4:
    image = image[:, :, :3]
gray = color.rgb2gray(image)

# Apply Otsu thresholding to convert to binary
thresh = filters.threshold_otsu(gray)
binary = gray < thresh

# Labeling komponen yang terhubung
labeled_image = label(binary)

# Dapatkan properti dari setiap komponen
props = regionprops(labeled_image)

# Memisahkan Huruf Utama dan Titik
main_components = np.zeros_like(binary)
dot_components = np.zeros_like(binary)

for prop in props:
    if prop.area > 50:
        # Simpan sebagai huruf utama
        main_components[labeled_image == prop.label] = 1
    elif prop.area > 5:
        # Periksa apakah titik berada dekat dengan huruf utama
        min_distance = np.min([np.linalg.norm(np.array(prop.centroid) - np.array(main_prop.centroid))
                               for main_prop in props if main_prop.area > 45])
        if min_distance < 27:
            dot_components[labeled_image == prop.label] = 1

# Gabungkan huruf utama dan titik yang valid
cleaned_binary = main_components + dot_components

# Skeletonization
skeleton = skeletonize(cleaned_binary)

# Extract pixel coordinates of the skeleton
y_coords, x_coords = np.where(skeleton)
nodes = list(zip(y_coords, x_coords))

# Menentukan titik awal sebagai titik paling kanan dalam skeleton (x terbesar)
start_node = max(nodes, key=lambda n: n[1])

# Function for Nearest Neighbor TSP
def nearest_neighbor_tsp(start, nodes):
    remaining_nodes = set(nodes)
    path = [start]
    remaining_nodes.remove(start)

    while remaining_nodes:
        last_node = path[-1]
        nearest = min(remaining_nodes, key=lambda n: distance.euclidean(last_node, n))
        path.append(nearest)
        remaining_nodes.remove(nearest)

    path.append(start)  # Kembali ke titik awal
    return path

# Menjalankan algoritma TSP dengan titik awal yang telah ditentukan
tsp_path = nearest_neighbor_tsp(start_node, nodes)

# Titik akhir sebelum kembali ke titik awal
end_node = tsp_path[-2]

# Menampilkan skeleton gambar
fig, ax = plt.subplots(figsize=(6, 6))
ax.imshow(skeleton, cmap="gray")

# Visualisasi jalur TSP
for i in range(len(tsp_path) - 1):
    (y1, x1), (y2, x2) = tsp_path[i], tsp_path[i + 1]
    ax.plot([x1, x2], [y1, y2], marker='o', markersize=2, color='red')

# Tandai titik awal dan akhir
ax.plot(start_node[1], start_node[0], marker='o', markersize=6, color='green', label="Titik Awal")  # Hijau untuk titik awal & akhir
ax.plot(end_node[1], end_node[0], marker='o', markersize=6, color='blue', label="Titik Akhir")  # Biru sebelum kembali ke awal

ax.set_title("TSP Using Nearest Neighbor")
ax.legend()
ax.axis("off")
plt.show()