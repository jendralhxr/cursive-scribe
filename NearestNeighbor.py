# usage: python -u line2hist.py <inputimage>
import os
#os.chdir("/shm")
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import sys
import math
from skimage.morphology import skeletonize
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean

PHI= 1.6180339887498948482 # ppl says this is a beautiful number :)
def freeman(x, y):
    if (y==0):
        y=1e-9 # so that we escape the divby0 exception
    if (x==0):
        x=-1e-9 # biased to the left as the text progresses leftward
    if (abs(x/y)<pow(PHI,2)) and (abs(y/x)<pow(PHI,2)): # corner angles
        if   (x>0) and (y>0):
            return(1)
        elif (x<0) and (y>0):
            return(3)
        elif (x<0) and (y<0):
            return(5)
        elif (x>0) and (y<0):
            return(7)
    else: # square angles
        if   (x>0) and (abs(x)>abs(y)):
            return(int(0))
        elif (y>0) and (abs(y)>abs(x)):
            return(2)
        elif (x<0) and (abs(x)>abs(y)):
            return(4)
        elif (y<0) and (abs(y)>abs(x)):
            return(6)
        
RESIZE_FACTOR=2
SLIC_SPACE= 3
SLIC_SPACE= SLIC_SPACE*RESIZE_FACTOR

THREVAL= 60
RASMVAL= 160

CHANNEL= 2

def draw(img): # draw the bitmap
    plt.figure(dpi=600)
    plt.grid(False)
    if (len(img.shape)==3):
        plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
    elif (len(img.shape)==2):
        plt.imshow(cv.cvtColor(img, cv.COLOR_GRAY2RGB))
        
filename= sys.argv[1]
#filename= 'topanribut.png'
imagename, ext= os.path.splitext(filename)
image = cv.imread(filename)
resz = cv.resize(image, (RESIZE_FACTOR*image.shape[1], RESIZE_FACTOR*image.shape[0]), interpolation=cv.INTER_LINEAR)
image= resz.copy()
image=  cv.bitwise_not(image)
height= image.shape[0]
width= image.shape[1]

image_gray= cv.cvtColor(image, cv.COLOR_BGR2GRAY)
image_gray= image[:,:,CHANNEL]
_, gray = cv.threshold(image_gray, 0, THREVAL, cv.THRESH_OTSU) # less smear
#_, gray= cv.threshold(selective_eroded, 0, THREVAL, cv.THRESH_TRIANGLE) # works better with dynamic-selective erosion
#draw(gray)
render = cv.cvtColor(gray, cv.COLOR_GRAY2BGR)

#SLIC
cue = gray.copy()
slic = cv.ximgproc.createSuperpixelSLIC(cue,algorithm = cv.ximgproc.SLICO, region_size = SLIC_SPACE)
slic.iterate()
mask= slic.getLabelContourMask()
result_mask = cv.bitwise_and(cue, mask)
num_slic = slic.getNumberOfSuperpixels()
lbls = slic.getLabels()

# moments calculation for each superpixels, either voids or filled (in-stroke)
moments = [np.zeros((1, 2)) for _ in range(num_slic)]
moments_void = [np.zeros((1, 2)) for _ in range(num_slic)]
# tabulating the superpixel labels
for j in range(height):
    for i in range(width):
        if cue[j,i]!=0:
            moments[lbls[j,i]] = np.append(moments[lbls[j,i]], np.array([[i,j]]), axis=0)
            render[j,i,0]= 140-(10*(lbls[j,i]%6))
        else:
            moments_void[lbls[j,i]] = np.append(moments_void[lbls[j,i]], np.array([[i,j]]), axis=0)

#moments[0][1] = [0,0] # random irregularities, not quite sure why
# some badly needed 'sanity' check
def remove_zeros(moments):
    temp=[]
    v= len(moments)
    if v==1:
        return temp
    else:
        for p in range(v):
            if moments[p][0]!=0. and moments[p][1]!=0.:
                temp.append(moments[p])
        return temp

for n in range(len(moments)):
    moments[n]= remove_zeros(moments[n])

# draw(render)

######## // image preprocessing ends here

# generating nodes
scribe= nx.Graph() # start anew, just in case

# valid superpixel
filled=0
for n in range(num_slic):
    if ( len(moments[n])>SLIC_SPACE ): # remove spurious superpixel with area less than 2 px 
        cx= int( np.mean( [array[0] for array in moments[n]] )) # centroid
        cy= int( np.mean( [array[1] for array in moments[n]] ))
        if (cue[cy,cx]!=0):
            render[cy,cx,1] = 255 
            scribe.add_node(int(filled), label=int(lbls[cy,cx]), area=(len(moments[n])-1)/pow(SLIC_SPACE,2), hurf='', pos_bitmap=(cx,cy), pos_render=(cx,-cy), color='#FFA500', rasm=True)
            #print(f'point{n} at ({cx},{cy})')
            filled=filled+1

def pdistance(point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return distance

# connected componentscv.circle(disp, pos[compodef line_iterator(img, point0, point1):
from dataclasses import dataclass, field
from typing import List
from typing import Optional

@dataclass
class ConnectedComponents:
    rect: (int,int,int,int) # from bounding rectangle
    centroid: (int,int) # centroid moment
    area: Optional[int] = field(default=0)
    nodes: List[int] = field(default_factory=list)
    mat: Optional[np.ndarray] = field(default=None, repr=False)
    node_start: Optional[int] = field(default=-1)    # right-up
    distance_start: Optional[int] = field(default=0) # right-up
    node_end: Optional[int] = field(default=-1)      # left-down
    distance_end: Optional[int] = field(default=0)   # left-down


pos = nx.get_node_attributes(scribe,'pos_bitmap')
components=[]
for n in range(scribe.number_of_nodes()):
    # fill
    seed= pos[n]
    ccv= gray.copy()
    cv.floodFill(ccv, None, seed, RASMVAL, loDiff=(5), upDiff=(5))
    _, ccv = cv.threshold(ccv, 100, RASMVAL, cv.THRESH_BINARY)
    mu= cv.moments(ccv)
    if mu['m00'] > pow(SLIC_SPACE,2)*PHI:
        mc= (int(mu['m10'] / (mu['m00'])), int(mu['m01'] / (mu['m00'])))
        area = mu ['m00']
        pd= pdistance(seed, mc)
        node_start = n
        box= cv.boundingRect(ccv)
        # append keypoint if the component already exists
        found=0
        for i in range(len(components)):
            if components[i].centroid==mc:
                components[i].nodes.append(n)
                # calculate the distance
                tvane= freeman(seed[0]-mc[0], mc[1]-seed[1] )
                #if seed[0]>mc[0] and pd>components[i].distance_start and (tvane==2 or tvane==4): # potential node_start for long rasm
                if seed[0]>mc[0] and pd>components[i].distance_start: # potential node_start
                    components[i].distance_start= pd
                    components[i].node_start= n
                elif seed[0]<mc[0] and pd>components[i].distance_end: # potential node_end
                    components[i].distance_end = pd
                    components[i].node_end= n
                found=1
                # print(f'old node[{n}] with component[{i}] at {mc} from {components[i].centroid} distance: {pd})')
                break
        if (found==0):
            components.append(ConnectedComponents(box, mc))
            idx= len(components)-1
            components[idx].nodes.append(n)
            components[idx].mat = ccv.copy()
            components[idx].area = int(mu['m00']/THREVAL)
            if seed[0]>mc[0]:
                components[idx].node_start= n
                components[idx].distance_start= pd
            else:
                components[idx].node_end= n
                components[idx].distance_end= pd
            #print(f'new node[{n}] with component[{idx}] at {mc} from {components[idx].centroid} distance: {pd})')


components = sorted(components, key=lambda x: x.centroid[0], reverse=True)
# for n in len(components):
#     for i in components[n].nodes:
#         distance= pdistance(components[n].centroid, pos[i])
#         print(f'{i}: {distance}')

# drawing the starting node (bitmap level)
disp = cv.cvtColor(gray, cv.COLOR_GRAY2BGR)
for n in range(len(components)):
    #print(f'{n} at {components[n].centroid} size {components[n].area}')
    # draw green line for rasm at edges, color the rasm brighter
    if components[n].area>4*PHI*pow(SLIC_SPACE,2):
        disp= cv.bitwise_or(disp, cv.cvtColor(components[n].mat,cv.COLOR_GRAY2BGR))
        seed= components[n].centroid
        cv.circle(disp, seed, 2, (0,0,120), -1)
        if components[n].node_start!=-1:
            cv.circle(disp, pos[components[n].node_start], 2, (0,120,0), -1)
        if components[n].node_end!=-1:
            cv.circle(disp, pos[components[n].node_end], 2, (120,0,0), -1)
        r= components[n].rect[0]+int(components[n].rect[2])
        l= components[n].rect[0]
        if l<width and r<width: # did we ever went beyond the frame?
            for j1 in range(int(SLIC_SPACE*PHI),height-int(SLIC_SPACE*PHI)):
                disp[j1,r,1]= 120
            for j1 in range(int(SLIC_SPACE*pow(PHI,3)),height-int(SLIC_SPACE*pow(PHI,3))):
                disp[j1,l,1]= 120
    else:        
        m= components[n].centroid[1]
        i= components[n].centroid[0]
        # draw blue line for shakil 'connection'
        for j2 in range(int(m-(2*SLIC_SPACE*PHI)), int(m+(2*SLIC_SPACE*PHI))):
            if j2<height and j2>0: 
                disp[j2,i,1]= RASMVAL/2
draw(disp) 



# SKELETON
skeleton_components = []

for n in range(len(components)):
    binary_mat = (components[n].mat == RASMVAL).astype(np.uint8)
    skeleton = skeletonize(binary_mat)
    skeleton_components.append(skeleton)

# Gabungkan semua skeleton menjadi satu gambar
combined_skeleton = np.zeros_like(skeleton_components[0], dtype=np.uint8)

for skeleton in skeleton_components:
    combined_skeleton |= skeleton

# Tampilkan semua skeleton sebagai satu gambar
plt.figure(figsize=(8, 8))
plt.imshow(combined_skeleton, cmap='gray')
plt.title("Combined Skeleton of All Components")
plt.axis('off')
plt.show()

# contours, hierarchy = cv.findContours(gray, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
# cv.Canny(gray, 30, 30)

edges = cv.Canny(gray, 30, 30)

# Temukan kontur berdasarkan gambar hasil deteksi tepi
contours, hierarchy = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

# Opsional: tampilkan hasil deteksi tepi untuk verifikasi
plt.imshow(edges, cmap="gray")
plt.title("Edges")
plt.show()




# Fungsi untuk menghitung jarak Euclidean
def euclidean(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

# Ambil koordinat dari skeleton yang sudah diekstrak
def get_skeleton_points(skeleton_image):
    points = np.column_stack(np.where(skeleton_image > 0))
    return points

# Algoritma Nearest Neighbor untuk TSP
def nearest_neighbor_tsp(points):
    num_points = len(points)
    unvisited = set(range(num_points))
    path = [unvisited.pop()]
    
    while unvisited:
        last = path[-1]
        next_point = min(unvisited, key=lambda i: euclidean(points[last], points[i]))
        path.append(next_point)
        unvisited.remove(next_point)
    
    path.append(path[0])  # Kembali ke titik awal
    return path

# Algoritma 2-opt untuk peningkatan jalur
def two_opt_tsp(points, path):
    def path_distance(path):
        return sum(euclidean(points[path[i]], points[path[i+1]]) for i in range(len(path)-1))
    
    improved = True
    while improved:
        improved = False
        for i in range(1, len(path) - 2):
            for j in range(i + 1, len(path) - 1):
                if j - i == 1: continue  # Hindari perubahan yang tidak berarti
                new_path = path[:i] + path[i:j+1][::-1] + path[j+1:]
                if path_distance(new_path) < path_distance(path):
                    path = new_path
                    improved = True
    return path

# Ambil skeleton dari gambar (contoh, gambar kamu bisa diubah)
# combined_skeleton adalah gambar yang sudah diproses sebelumnya
skel_points = get_skeleton_points(combined_skeleton)

step = 1  # Atur step untuk menyesuaikan jarak antar titik
skel_points = skel_points[::step]

# Ekstrak koordinat X dan Y
x_coords, y_coords = skel_points[:, 1], skel_points[:, 0]

# Menjalankan algoritma TSP
tsp_path = nearest_neighbor_tsp(skel_points)
tsp_path = two_opt_tsp(skel_points, tsp_path)

# Print hasil TSP
print("TSP Path :")
for i, idx in enumerate(tsp_path):
    x, y = skel_points[idx]
    print(f"{i+1}. Point {idx}: ({x}, {y})")

# Menghitung total jarak Euclidean
total_distance = 0
for i in range(len(tsp_path) - 1):
    p1 = skel_points[tsp_path[i]]
    p2 = skel_points[tsp_path[i + 1]]
    dist = euclidean(p1, p2)
    total_distance += dist

print(f"Total Euclidean Distance: {total_distance:.2f}")

# Plot hasil TSP
plt.figure(figsize=(8, 8))
for i in range(len(tsp_path) - 1):
    p1, p2 = skel_points[tsp_path[i]], skel_points[tsp_path[i+1]]
    plt.plot([p1[1], p2[1]], [p1[0], p2[0]], 'r-')

# Plot titik-titik skeleton
plt.scatter(x_coords, y_coords, s=50, c='blue')

plt.title("TSP Path on Skeleton")
plt.gca().invert_yaxis()
plt.show()

# Plot hasil dengan jarak antar titik
plt.figure(figsize=(8, 8))
for i in range(len(tsp_path) - 1):
    p1 = skel_points[tsp_path[i]]
    p2 = skel_points[tsp_path[i + 1]]
    
    # Gambar garis antar titik
    plt.plot([p1[1], p2[1]], [p1[0], p2[0]], 'r-')

    # Hitung jarak Euclidean antar titik
    dist = euclidean(p1, p2)

    # Posisi teks di tengah-tengah garis
    mid_x = (p1[1] + p2[1]) / 2
    mid_y = (p1[0] + p2[0]) / 2
    plt.text(mid_x, mid_y, f"{dist:.1f}", fontsize=6, color='green')

# Plot titik-titik skeleton
plt.scatter(x_coords, y_coords, s=50, c='blue')
plt.title(f"TSP Path on Skeleton\nTotal Distance: {total_distance:.2f}")
plt.gca().invert_yaxis()
plt.show()







# # Plot hasil TSP dengan lebih jelas
# plt.figure(figsize=(8, 8))

# # Atur batas sumbu agar ada ruang tambahan
# x_min, x_max = skel_points[:, 1].min(), skel_points[:, 1].max()
# y_min, y_max = skel_points[:, 0].min(), skel_points[:, 0].max()
# padding = 10  # Tambahkan ruang ekstra agar titik tidak terlalu berdempetan

# plt.xlim(x_min - padding, x_max + padding)
# plt.ylim(y_min - padding, y_max + padding)

# # Plot jalur dengan transparansi agar lebih jelas
# for i in range(len(tsp_path) - 1):
#     p1, p2 = skel_points[tsp_path[i]], skel_points[tsp_path[i+1]]
#     plt.plot([p1[1], p2[1]], [p1[0], p2[0]], 'r-', alpha=0.6, linewidth=1)

# # Plot titik dengan warna lebih kontras
# plt.scatter(skel_points[:, 1], skel_points[:, 0], s=30, c='cyan', edgecolors='black', zorder=3)

# plt.title("TSP Path on Skeleton")
# plt.gca().invert_yaxis()
# plt.show()