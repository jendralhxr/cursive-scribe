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
if len(skeleton_components) > 0:
    combined_skeleton = np.zeros_like(skeleton_components[0], dtype=np.uint8)

    for skeleton in skeleton_components:
        combined_skeleton |= skeleton
else:
    combined_skeleton = None

# Periksa apakah skeleton kosong sebelum menampilkan
if combined_skeleton is not None and np.any(combined_skeleton > 0):
    plt.figure(figsize=(8, 8))
    plt.imshow(combined_skeleton, cmap='gray')
    plt.title("Combined Skeleton of All Components")
    plt.axis('off')
    plt.show()
else:
    print("Skeleton kosong, tidak ada data untuk ditampilkan.")

# DETEKSI TEPI (EDGE DETECTION)
edges = cv.Canny(gray, 30, 30)

# Temukan kontur berdasarkan gambar hasil deteksi tepi
contours, hierarchy = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

# Opsional: tampilkan hasil deteksi tepi untuk verifikasi
plt.imshow(edges, cmap="gray")
plt.title("Edges")
plt.show()

# Pastikan skeleton tidak kosong sebelum melanjutkan ke TSP
if combined_skeleton is not None and np.any(combined_skeleton > 0):
    # Ekstrak titik skeleton (dibalik dari (y, x) ke (x, y))
    points = np.column_stack(np.where(combined_skeleton > 0))[:, ::-1]

    # Periksa apakah ada cukup titik untuk membuat jalur TSP
    if points.shape[0] > 1:
        # Buat graph berbasis Euclidean Distance
        G = nx.Graph()
        for i, p1 in enumerate(points):
            for j, p2 in enumerate(points[i+1:], start=i+1):
                dist = euclidean(p1, p2)
                G.add_edge(tuple(p1), tuple(p2), weight=dist)

        # Algoritma TSP dengan pendekatan Greedy (dengan siklus kembali ke awal)
        def tsp_greedy(graph, points):
            start = tuple(points[0])
            path = [start]
            unvisited = set(map(tuple, points))
            unvisited.remove(start)

            while unvisited:
                last = path[-1]
                next_node = min(unvisited, key=lambda node: graph[last][node]['weight'])
                path.append(next_node)
                unvisited.remove(next_node)

            # Tambahkan titik awal ke akhir jalur agar membentuk siklus
            path.append(start)  
            return path

        # Jalankan TSP
tsp_path = tsp_greedy(G, points)

# Gambar hasil TSP dengan titik dan garis
skeleton_rgb = cv.cvtColor((combined_skeleton * 255).astype(np.uint8), cv.COLOR_GRAY2BGR)
plt.figure(figsize=(8, 8))
plt.imshow(skeleton_rgb, cmap='Greys')
plt.title("Optimized TSP Path for Handwritten Arabic Letters")

# Gambar titik dan garis dengan koordinat yang benar (x, y)
total_distance = 0  # Variabel untuk menyimpan jarak total
for i in range(len(tsp_path) - 1):
    p1, p2 = tsp_path[i], tsp_path[i+1]
    # Menggambar garis merah antar titik
    plt.plot([p1[0], p2[0]], [p1[1], p2[1]], 'r-', linewidth=1)
    # Menggambar titik merah di setiap titik
    plt.plot(p1[0], p1[1], 'ro', markersize=2)
    
    # Menghitung jarak antar titik
    dist = euclidean(p1, p2)
    total_distance += dist
    
    # Menampilkan jarak antar titik pada garis
    mid_x = (p1[0] + p2[0]) / 2
    mid_y = (p1[1] + p2[1]) / 2
    plt.text(mid_x, mid_y, f"{dist:.2f}", fontsize=6, color='green')

# Warna titik awal biru ('bo')
plt.plot(tsp_path[0][0], tsp_path[0][1], 'bo', markersize=5)  # Titik awal (biru)

# Warna titik akhir hijau ('go')
plt.plot(tsp_path[-2][0], tsp_path[-2][1], 'go', markersize=5)  # Titik akhir (hijau)

# Menampilkan total jarak
plt.title(f"Optimized TSP Path\nTotal Distance: {total_distance:.2f}")

plt.show()

print("TSP Path (order of points):")

total_distance = 0
for i, point in enumerate(tsp_path):
    x, y = map(int, point)
    
    # Jika bukan titik terakhir, hitung jarak ke titik berikutnya
    if i < len(tsp_path) - 1:
        next_point = tsp_path[i + 1]
    else:
        # Titik terakhir, kembali ke titik awal (TSP siklik)
        next_point = tsp_path[0]
    
    dist = euclidean(point, next_point)
    total_distance += dist
    
    print(f"{i+1}. Point {i}: ({x}, {y}) | Jarak: {dist:.2f}")

print(f"Total Euclidean Distance: {total_distance:.2f}")
