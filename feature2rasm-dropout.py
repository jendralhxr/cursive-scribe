# usage: python -u features.py <inputimage> | tee <outputrasm>
import os

# os.chdir("/shm")
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

# import sys
import math
import heapq

# import csv

# freeman code going anti-clockwise like trigonometrics angle
#    3   2   1
#      \ | /
#    4 ------0
#      / | \
#    5   6   7

PHI = 1.6180339887498948482  # ppl says this is a beautiful number :)


def freeman(x, y):
    if y == 0:
        y = 1e-9  # so that we escape the divby0 exception
    if x == 0:
        x = -1e-9  # biased to the left as the text progresses leftward
    if (abs(x / y) < pow(PHI, 2)) and (abs(y / x) < pow(PHI, 2)):  # corner angles
        if (x > 0) and (y > 0):
            return 1
        elif (x < 0) and (y > 0):
            return 3
        elif (x < 0) and (y < 0):
            return 5
        elif (x > 0) and (y < 0):
            return 7
    else:  # square angles
        if (x > 0) and (abs(x) > abs(y)):
            return int(0)
        elif (y > 0) and (abs(y) > abs(x)):
            return 2
        elif (x < 0) and (abs(x) > abs(y)):
            return 4
        elif (y < 0) and (abs(y) > abs(x)):
            return 6


RESIZE_FACTOR = 1
SLIC_SPACE = 8
SLIC_SPACE = SLIC_SPACE * RESIZE_FACTOR
WHITESPACE_INTERVAL = 4

RASM_EDGE_MAXDEG = 2
RASM_CANDIDATE = SLIC_SPACE

THREVAL = 60
STROKEVAL = 160
FOCUSVAL = 240
CHANNEL = 2


def draw(img):  # draw the bitmap
    plt.figure(dpi=600)
    plt.grid(False)
    if len(img.shape) == 3:
        plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
    elif len(img.shape) == 2:
        plt.imshow(cv.cvtColor(img, cv.COLOR_GRAY2RGB))


# filename= sys.argv[1]
# filename= 'topanribut.png'
filename = "dengarkan.png"
imagename, ext = os.path.splitext(filename)
image = cv.imread(filename)
resz = cv.resize(
    image,
    (RESIZE_FACTOR * image.shape[1], RESIZE_FACTOR * image.shape[0]),
    interpolation=cv.INTER_LINEAR,
)
image = resz.copy()
# image=  cv.bitwise_not(image)
height = image.shape[0]
width = image.shape[1]

grayImg = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
image_gray = image[:, :, CHANNEL]
_, gray = cv.threshold(image_gray, 0, THREVAL, cv.THRESH_OTSU)  # less smear
# _, gray= cv.threshold(selective_eroded, 0, THREVAL, cv.THRESH_TRIANGLE) # works better with dynamic-selective erosion

# selective erosion
# kernel_size=2
# canny_threshold1=100
# canny_threshold2=200
# edges = cv.Canny(gray, canny_threshold1, canny_threshold2)
# kernel = np.ones((kernel_size, kernel_size), np.uint8)
# eroded_image = cv.erode(gray, kernel, iterations=1)
# edge_mask = cv.bitwise_not(edges)
# selective_eroded = cv.bitwise_and(eroded_image, eroded_image, mask=edge_mask)
# ret, gray= cv.threshold(selective_eroded,1,THREVAL,cv.THRESH_BINARY)
# erosion-dilation @Alifah25
# erosion_kernel = np.ones((1, 2), np.uint8)
# eroded_image = cv.erode(image_gray, erosion_kernel, iterations=1)
# cv.imwrite('eroded_text.png', eroded_image)
# cv.imwrite('dilated_text.png', dilated_image)


DILATION_Y = 2  # big enough to salvage thin lines, yet not accidentally connecting close diacritics
DILATION_X = 3  # some vertical lines are just too thin
DILATION_I = 1

# SLIC
gray = cv.dilate(
    gray, np.ones((DILATION_Y, DILATION_X), np.uint8), iterations=DILATION_I
)  # turns out gray is actually already too thin to begin with
cue = gray.copy()
stroke = cv.dilate(
    cue, np.ones((int(SLIC_SPACE), int(SLIC_SPACE)), np.uint8), iterations=1
)  # this is for the connectedcomponent check
slic = cv.ximgproc.createSuperpixelSLIC(
    cue, algorithm=cv.ximgproc.SLICO, region_size=SLIC_SPACE
)
slic.iterate()
mask = slic.getLabelContourMask()
result_mask = cv.bitwise_and(cue, mask)
num_slic = slic.getNumberOfSuperpixels()
lbls = slic.getLabels()

# moments calculation for each superpixels, either voids or filled (in-stroke)
render = cv.cvtColor(gray, cv.COLOR_GRAY2BGR)
moments = [np.zeros((1, 2)) for _ in range(num_slic)]
moments_void = [np.zeros((1, 2)) for _ in range(num_slic)]
# tabulating the superpixel labels
for j in range(height):
    for i in range(width):
        if cue[j, i] != 0:
            moments[lbls[j, i]] = np.append(
                moments[lbls[j, i]], np.array([[i, j]]), axis=0
            )
            render[j, i, 0] = THREVAL - (10 * (lbls[j, i] % 6))
        else:
            moments_void[lbls[j, i]] = np.append(
                moments_void[lbls[j, i]], np.array([[i, j]]), axis=0
            )


# moments[0][1] = [0,0] # random irregularities, not quite sure why
# some badly needed 'sanity' check
def remove_zeros(moments):
    temp = []
    v = len(moments)
    if v == 1:
        return temp
    else:
        for p in range(v):
            if moments[p][0] != 0.0 and moments[p][1] != 0.0:
                temp.append(moments[p])
        return temp


for n in range(len(moments)):
    moments[n] = remove_zeros(moments[n])

# draw(render)

######## // image preprocessing ends here

# generating nodes
scribe = nx.Graph()  # start anew, just in case

# valid superpixel
filled = 0


from skimage.morphology import skeletonize

# angle histogram
# We use a filter to blur out the noise from the image.
gaussianFilter = cv.GaussianBlur(grayImg, (5, 5), 0)  # grayImg,(5,5)
plt.imshow(gaussianFilter, cmap="gray")
cv.imwrite("GaussianFilter.png", gaussianFilter)

# binarize and invert the image.  # gaussianFilter
# _, binarizedImg = cv2.threshold(gaussianFilter, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
_, binarizedImg = cv.threshold(
    gaussianFilter, 0, 180, cv.THRESH_BINARY | cv.THRESH_OTSU
)

# plt.imshow(binarizedImg, cmap="gray")
cv.imwrite("binarizedImg.png", binarizedImg)

binarizedImg[binarizedImg < 180] = 0  # 1
binarizedImg[binarizedImg > 180] = 1  # 0
# Taking a matrix of size 5 as the kernel
kernel = np.ones((5, 5), np.uint8)
# #Erosion using skeletonize
skeletonImg = skeletonize(binarizedImg)
# cv.imwrite("skeletonImg.png", skeletonImg)
plt.imshow(skeletonImg, cmap="gray")
plt.savefig("skeletonImg.png", dpi=300)


list_space_angle_start = []
list_space_angle_end = []
print("width ", width)
step = -0.48


def search_list_area():  # cari list angle sebelum gambar diputar
    # titik_random = point  #misal area yg dipilih
    count_contigu = 0
    before = (-1, -1)
    j = width - 1  # memotong sumbu x : x = -c/m untuk y = 0, c = -192
    while j < (width) and j > 0:
        j -= 1
        k3 = j
        k4 = 0
        sum = 0
        # count +=1
        while k4 < (height - 1) and k3 > -1:  # and count <20:
            k3 = round(k3 + (1 / step))  # k3-1
            k4 = k4 + 1  # int(round(k4-step))
            # print("saja k3,k4 ",k3,k4)
            # nanti for loop titik random ada banyak, batas1_x ada banyak(append)
            # if k4==titik_random[1] and k3==titik_random[0] :
            # ambil batas area yg sebelumnya: list_split_angle_start[-1][0]
            # if len(list_space_angle_start)>0 :
            #     batas1_x = list_space_angle_start[-1][0]
            # else : #ambil width
            #     batas1_x = width
            # print("batas1_x ",batas1_x)

            try:
                sum = sum + skeletonImg[k4, k3]  # skeletonImg
                # print("k3,k4 ",k3,k4,skeletonImg[k4,k3],sum) #skeletonImg
            except:
                print()
        # arr_sum.append(sum)

        if sum == 0:
            if (
                before[0] == (j + 1)
                or before[0] == (j + 2)
                or before[0] == (j + 3)
                or before[0] == (j + 4)
            ):
                count_contigu += 1
                if (
                    count_contigu > 0
                ):  # jika ada beberapa berdampingan, dipilih yg contigu
                    if (
                        len(list_space_angle_start) > 0
                    ):  # jika sudah ada list berdampingan
                        # if list_split[-1][0]!=(j+1) and  list_split[-1][0]!=(j+2) and list_split[-1][0]!=(j+3) and list_split[-1][0]!=(j+4) : #jika elemen list terakhir tidak sama dg j+1
                        if list_space_angle_start[-1][0] - j > 5:
                            # list_split.append((j,0)) #add baru
                            list_space_angle_start.append((j, 0))
                            list_space_angle_end.append((k3, k4))
                        # elif list_split[-1][0]==(j+1) or list_split[-1][0]==(j+2) or list_split[-1][0]==(j+3) or list_split[-1][0]==(j+4) :
                        # elif list_split[-1][0] - j <= 13 :
                        else:
                            # jika elemen list terakhir sama dg j+1, atau berurutan, atau j+2,j+3,j+4
                            # list_split[-1]=(j,0)  #update j yg terakhir , untuk yg berdampingan sela 0,1,2,3,4
                            list_space_angle_start[-1] = (j, 0)
                            list_space_angle_end[-1] = (k3, k4)
                    else:
                        # list_split.append((j,0))
                        list_space_angle_start.append((j, 0))
                        list_space_angle_end.append((k3, k4))
                    # print(list_split," lstsplt ")
            else:
                count_contigu = 0
            # if count_contigu == 0 :
            before = (j, 0)

            # print(count_contigu," countigu ", before)

            # print(count,"########",j,k3,k4)
            # tidak dipilih


# batas1_x =0


def search_area(point):  # (x,y) sebelum gambar diputar, untuk satu point saja
    # titik_random = point  #misal area yg dipilih
    # found batas1 ada di range mana dari list_space_angle_start
    index_batas1 = -2  # untuk masing2 node
    print("point ", point)
    print("list_space_angle_start ", list_space_angle_start)

    if point[0] <= list_space_angle_start[0][0]:  # masuk area
        for s in range(len(list_space_angle_start)):
            if (
                point[0] <= list_space_angle_start[s][0]
                and point[0] > list_space_angle_start[s + 1][0]
            ):
                # update batas atas
                # batas1_x = list_space_angle_start[s][0]
                index_batas1 = s
                print("index_batas1 ", index_batas1, list_space_angle_start[s][0])
                # update batas bawah
                # batas2_x = list_space_angle_start[s+1][0]
                # index_batas2 = s+1
            # else....
            # print("batas1_x ", batas1_x)
    else:  # belum terisi karena di luar coverage
        # ada kasus dimana area random di luar coverage area gradient karena kemiringannya
        # update batas bawah
        # batas1_x masih = 0 , untuk diputar nanti = putdraw-width
        index_batas1.append(-1)  # berarti di area pinggir ujung habis
        # batas2_x = list_space_angle_start[0][0] nanti di penggambaran scribe
    return index_batas1


search_list_area()

for n in range(num_slic):
    if (
        len(moments[n]) > SLIC_SPACE
    ):  # remove spurious superpixel with area less than 2 px
        cx = int(np.mean([array[0] for array in moments[n]]))  # centroid
        cy = int(np.mean([array[1] for array in moments[n]]))
        if cue[cy, cx] != 0:
            render[cy, cx, 1] = 255

            inde_bts = search_area((cx, cy))
            print(
                "search n,cx,cy, idx batas1_x ",
                n,
                cx,
                cy,
                inde_bts,
                list_space_angle_start[inde_bts][0],
            )
            # tiap node punya dari posisi cxcy punya indexbts menandakan dia di area mana
            scribe.add_node(
                int(filled),
                indexbts=inde_bts,
                label=int(lbls[cy, cx]),
                area=(len(moments[n]) - 1) / pow(SLIC_SPACE, 2),
                hurf="",
                pos_bitmap=(cx, cy),
                pos_render=(cx, -cy),
                color="#FFA500",
                rasm=True,
            )
            # print(f'point{n} at ({cx},{cy})')
            filled = filled + 1


def pdistance(point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return distance


# connected components
from dataclasses import dataclass, field
from typing import List
from typing import Optional


@dataclass
class ConnectedComponents:
    rect: (int, int, int, int)  # from bounding rectangle
    centroid: (int, int)  # centroid moment
    area: Optional[int] = field(default=0)
    nodes: List[int] = field(default_factory=list)
    mat: Optional[np.ndarray] = field(default=None, repr=False)
    node_start: Optional[int] = field(default=-1)  # right-up
    distance_start: Optional[int] = field(default=0)  # right-up
    node_end: Optional[int] = field(default=-1)  # left-down
    distance_end: Optional[int] = field(default=0)  # left-down


pos = nx.get_node_attributes(scribe, "pos_bitmap")


def euclidean_distance(p1, p2):
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


min_distances = []
for key1, point1 in pos.items():
    distances = [
        euclidean_distance(point1, point2)
        for key2, point2 in pos.items()
        if key1 != key2
    ]
    min_distance = min(distances)
    min_distances.append(min_distance)
mean_closest_distance = np.mean(
    min_distances
)  # this should closely resembles SLIC_SPACE

indbts = nx.get_node_attributes(scribe, "indexbts")
print("indbts ", indbts)

components = []
for n in range(scribe.number_of_nodes()):
    # fill
    seed = pos[n]  # cari di area angle mana
    print("nodes, pos n ", n, pos[n])
    print("indbts, pos n ", n, indbts)  # cek dulu
    # print("indbts, pos n0 ",n,indbts[n][0]) #cek dulu

    ccv = cue.copy()
    cv.floodFill(ccv, None, seed, STROKEVAL, loDiff=(5), upDiff=(5))
    _, ccv = cv.threshold(ccv, 100, STROKEVAL, cv.THRESH_BINARY)
    mu = cv.moments(ccv)
    if mu["m00"] > pow(SLIC_SPACE, 2) * PHI:  # minimum area for a connectedcomponent
        mc = (int(mu["m10"] / (mu["m00"])), int(mu["m01"] / (mu["m00"])))
        area = mu["m00"]
        pd = pdistance(seed, mc)
        node_start = n
        box = cv.boundingRect(ccv)  # sejauh mana ukuran bounding box ?
        print("box ", box.rect[0])
        # append keypoint if the component already exists
        found = 0
        for i in range(len(components)):
            if components[i].centroid == mc:
                components[i].nodes.append(n)
                # calculate the distance
                tvane = freeman(seed[0] - mc[0], mc[1] - seed[1])
                # if seed[0]>mc[0] and pd>components[i].distance_start and (tvane==2 or tvane==4): # potential node_start for long rasm
                if (
                    seed[0] > mc[0] and pd > components[i].distance_start
                ):  # potential node_start
                    components[i].distance_start = pd
                    components[i].node_start = n
                elif (
                    seed[0] < mc[0] and pd > components[i].distance_end
                ):  # potential node_end
                    components[i].distance_end = pd
                    components[i].node_end = n

                found = 1
                # print(f'old node[{n}] with component[{i}] at {mc} from {components[i].centroid} distance: {pd})')
                break

        if found == 0:  # if not found
            components.append(ConnectedComponents(box, mc))
            print("box len(components)", len(components), box.rect[0])
            idx = len(components) - 1
            components[idx].nodes.append(n)
            components[idx].mat = ccv.copy()
            components[idx].area = int(mu["m00"] / THREVAL)
            if seed[0] > mc[0]:
                components[idx].node_start = n
                components[idx].distance_start = pd
            else:
                components[idx].node_end = n
                components[idx].distance_end = pd
            # print(f'new node[{n}] with component[{idx}] at {mc} from {components[idx].centroid} distance: {pd})')


components = sorted(components, key=lambda x: x.centroid[0], reverse=True)
stray = [
    c
    for c in components
    if c.area < pow(SLIC_SPACE, 2) + SLIC_SPACE * PHI and len(c.nodes) == 1
]
for i in stray:
    scribe.remove_node(i.nodes[0])
components = [
    c
    for c in components
    if c.area >= pow(SLIC_SPACE, 2) + SLIC_SPACE * PHI or len(c.nodes) > 1
]

# for n in len(components):
#     for i in components[n].nodes:
#         distance= pdistance(components[n].centroid, pos[i])
#         print(f'{i}: {distance}')

# drawing the starting node (bitmap level)
# disp = cv.cvtColor(cue, cv.COLOR_GRAY2BGR)
# for n in range(len(components)):
#     #print(f'{n} at {components[n].centroid} size {components[n].area}')
#     # draw green line for rasm at edges, color the rasm brighter
#     if components[n].area>4*PHI*pow(SLIC_SPACE,2):
#         disp= cv.bitwise_or(disp, cv.cvtColor(components[n].mat,cv.COLOR_GRAY2BGR))
#         seed= components[n].centroid
#         cv.circle(disp, seed, 2, (0,0,120), -1)
#         if components[n].node_start!=-1:
#             cv.circle(disp, pos[components[n].node_start], 2, (0,120,0), -1)
#         if components[n].node_end!=-1:
#             cv.circle(disp, pos[components[n].node_end], 2, (120,0,0), -1)
#         r= components[n].rect[0]+int(components[n].rect[2])
#         l= components[n].rect[0]
#         if l<width and r<width: # did we ever went beyond the frame?
#             for j1 in range(int(SLIC_SPACE*PHI),height-int(SLIC_SPACE*PHI)):
#                 disp[j1,r,1]= 120
#             for j1 in range(int(SLIC_SPACE*pow(PHI,3)),height-int(SLIC_SPACE*pow(PHI,3))):
#                 disp[j1,l,1]= 120
#     else:
#         m= components[n].centroid[1]
#         i= components[n].centroid[0]
#         # draw blue line for shakil 'connection'
#         for j2 in range(int(m-(2*SLIC_SPACE*PHI)), int(m+(2*SLIC_SPACE*PHI))):
#             if j2<height and j2>0:
#                 disp[j2,i,1]= STROKEVAL/2
# crop in each rasm
# rasm= components[n].mat[\
#     components[n].rect[1]:components[n].rect[1]+components[i].rect[3],\
#     components[n].rect[0]:components[n].rect[0]+components[i].rect[2]]
# cv.imwrite(str(n)+'.png', rasm)
# draw(disp)
# draw(render)

# from datetime import datetime
# now = datetime.now()
# date_time_str = now.strftime("%Y%m%d%H%M%S")
# cv.imwrite('/shm/'+date_time_str+'-render.png', render)


def draw_graph(graph, posstring, scale):
    # nodes
    plt.figure(figsize=(3 * scale, 3))
    positions = nx.get_node_attributes(graph, posstring)
    area = np.array(list(nx.get_node_attributes(graph, "area").values()))
    # edges
    node_colors = nx.get_node_attributes(graph, "color").values()
    edge_colors = nx.get_edge_attributes(graph, "color").values()
    weights = np.array(list(nx.get_edge_attributes(graph, "weight").values()))
    # plt.figure(figsize=(width/12,height/12))
    nx.draw(
        graph,
        # nodes' param
        pos=positions,
        with_labels=True,
        node_color=node_colors,
        node_size=area * 25,
        font_size=8,
        # edges' param
        edge_color=edge_colors,
        width=weights * 2,
    )


from matplotlib import font_manager

font_path = "/usr/share/fonts/truetype/noto/NotoNaskhArabic-Regular.ttf"  # Adjust to your Arabic font path
arabic_font_path = "/usr/share/fonts/truetype/noto/NotoNaskhArabic-Regular.ttf"  # Arabic-supporting font
fallback_font_path = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"  # General-purpose fallback font
)
arabic_font = font_manager.FontProperties(fname=font_path)
arabic_font = font_manager.FontProperties(fname=arabic_font_path)
fallback_font = font_manager.FontProperties(fname=fallback_font_path)


def draw_graph_edgelabel(graph, posstring, scale, filename, labelfield):
    plt.figure(figsize=(4 * scale, 4))
    # nodes
    if posstring is None:
        positions = nx.spring_600layout(graph)
    else:
        positions = nx.get_node_attributes(graph, posstring)

    # area= np.array(list(nx.get_node_attributes(graph, 'area').values()))
    node_colors = nx.get_node_attributes(graph, "color").values()
    if labelfield is not None:
        custom_labels = {node: scribe.nodes[node]["hurf"] for node in scribe.nodes}
    # edges
    edge_lbls = nx.get_edge_attributes(graph, "vane")
    edge_colors = nx.get_edge_attributes(graph, "color").values()
    weights = np.array(list(nx.get_edge_attributes(graph, "weight").values()))
    # plt.figure(figsize=(width/12,height/12))
    if labelfield is not None:
        nx.draw(
            graph,
            # nodes' param
            pos=positions,
            with_labels=True,
            node_color=node_colors,
            labels=custom_labels,
            node_size=100,
            font_size=6,
            fontproperties=arabic_font,  # Set Arabic font for edge labels
            # edges' param
            edge_color=edge_colors,
            width=weights * 2,
        )
    else:
        nx.draw(
            graph,
            # nodes' param
            pos=positions,
            node_color=node_colors,
            with_labels=True,
            node_size=100,
            font_size=6,
            # edges' param
            edge_color=edge_colors,
            width=weights * 2,
        )

    nx.draw_networkx_edge_labels(
        graph, pos=positions, edge_labels=edge_lbls, font_size=5, font_color="red"
    )
    if filename is not None:
        plt.savefig(filename, dpi=300)


def draw_graph_edgelabel_ara(graph, posstring, scale, filename, labelfield):
    plt.figure(figsize=(4 * scale, 4))

    # Set an Arabic-supporting font
    font_path = "/usr/share/fonts/truetype/noto/NotoNaskhArabic-Regular.ttf"  # Adjust to your Arabic font path
    arabic_font = font_manager.FontProperties(fname=font_path)

    # Nodes
    if posstring is None:
        positions = nx.spring_layout(graph)
    else:
        positions = nx.get_node_attributes(graph, posstring)

    node_colors = list(nx.get_node_attributes(graph, "color").values())

    if labelfield is not None:
        custom_labels = {node: graph.nodes[node][labelfield] for node in graph.nodes}
    else:
        custom_labels = None

    # Edges
    edge_lbls = nx.get_edge_attributes(graph, "vane")
    edge_colors = list(nx.get_edge_attributes(graph, "color").values())
    weights = np.array(list(nx.get_edge_attributes(graph, "weight").values()))

    # Draw the graph without labels
    nx.draw(
        graph,
        pos=positions,
        with_labels=False,  # Temporarily disable labels for custom handling
        node_color=node_colors,
        node_size=100,
        font_size=6,
        edge_color=edge_colors,
        width=weights * 2,
    )

    # Draw custom node labels with Arabic font
    if custom_labels:
        for node, (x, y) in positions.items():
            plt.text(
                x,
                y,
                custom_labels[node],
                ha="center",
                fontproperties=arabic_font,
                fontsize=6,
                color="black",
            )

    # Draw edge labels with Arabic font
    for (node1, node2), label in edge_lbls.items():
        # Calculate the midpoint for the edge label
        x = (positions[node1][0] + positions[node2][0]) / 2
        y = (positions[node1][1] + positions[node2][1]) / 2
        plt.text(
            x,
            y,
            label,
            ha="center",
            fontproperties=arabic_font,
            fontsize=5,
            color="red",
        )

    # Save the figure if filename is provided
    if filename is not None:
        plt.savefig(filename, dpi=300)

    plt.show()


def extract_subgraph(G, start):  # for a connected component
    if start != -1:
        connected_component = nx.node_connected_component(G, start)
        connected_subgraph = G.subgraph(connected_component)
    return connected_subgraph.copy()


def extract_subgraph2(G, start, end):  # for a hurf inside a rasm, naive
    paths = []
    nodes_in_paths = set()
    edges_in_paths = set()
    for path in paths:
        nodes_in_paths.update(path)
        edges_in_paths.update(zip(path, path[1:]))
    subgraph = G.subgraph(nodes_in_paths).copy()
    # subgraph = G.edge_subgraph(edges_in_paths).copy()
    return subgraph


def extract_subgraph3(G, start, end):  # for a hurf inside a rasm, handling branches
    visited = [start, end]
    queue = [start, end]

    while queue:
        node = queue.pop(0)
        crossing_start = all(
            start in path for path in nx.all_simple_paths(G, node, end)
        )
        crossing_end = all(end in path for path in nx.all_simple_paths(G, node, start))
        # print(f"{node} s{crossing_start} e{crossing_end}")
        if (
            (node not in visited)
            and (crossing_start == False)
            and (crossing_end == False)
            or node == start
            or node == end
        ):
            visited.append(node)

            for neighbor in G.neighbors(node):
                if neighbor not in visited:
                    queue.append(neighbor)

    subgraph = G.subgraph(visited).copy()
    return subgraph


def edge_attributes(G):
    if isinstance(G, nx.MultiGraph) or isinstance(G, nx.MultiDiGraph):
        for u, v, k, attrs in G.edges(keys=True, data=True):
            print(f"({u}, {v}, {k}) {attrs}")
    elif isinstance(G, nx.Graph) or isinstance(G, nx.DiGraph):
        for u, v, attrs in G.edges(data=True):
            print(f"({u}, {v}) {attrs}")


# kudos to mohikhsan @stackoverflow
# https://stackoverflow.com/questions/32328179/opencv-3-0-lineiterator
# adapted to handle integer values
def line_iterator(img, P1, P2):
    if P1 == P2:
        return 1

    imageH = img.shape[0]
    imageW = img.shape[1]
    P1X = int(P1[0])
    P1Y = int(P1[1])
    P2X = int(P2[0])
    P2Y = int(P2[1])

    # Difference and absolute difference between points
    dX = P2X - P1X
    dY = P2Y - P1Y
    dXa = np.abs(dX)
    dYa = np.abs(dY)

    itbuffer = np.empty(shape=(np.maximum(dYa, dXa), 3), dtype=np.int32)

    # Obtain coordinates along the line using Bresenham's algorithm
    negY = P1Y > P2Y
    negX = P1X > P2X
    if P1X == P2X:  # Vertical line segment
        itbuffer[:, 0] = P1X
        if negY:
            itbuffer[:, 1] = np.arange(P1Y - 1, P1Y - dYa - 1, -1)
        else:
            itbuffer[:, 1] = np.arange(P1Y + 1, P1Y + dYa + 1)
    elif P1Y == P2Y:  # Horizontal line segment
        itbuffer[:, 1] = P1Y
        if negX:
            itbuffer[:, 0] = np.arange(P1X - 1, P1X - dXa - 1, -1)
        else:
            itbuffer[:, 0] = np.arange(P1X + 1, P1X + dXa + 1)
    else:  # Diagonal line segment
        steepSlope = dYa > dXa
        if steepSlope:
            slope = dX / dY
            if negY:
                itbuffer[:, 1] = np.arange(P1Y - 1, P1Y - dYa - 1, -1)
            else:
                itbuffer[:, 1] = np.arange(P1Y + 1, P1Y + dYa + 1)
            itbuffer[:, 0] = (slope * (itbuffer[:, 1] - P1Y) + P1X).astype(np.int32)
        else:
            slope = dY / dX
            if negX:
                itbuffer[:, 0] = np.arange(P1X - 1, P1X - dXa - 1, -1)
            else:
                itbuffer[:, 0] = np.arange(P1X + 1, P1X + dXa + 1)
            itbuffer[:, 1] = (slope * (itbuffer[:, 0] - P1X) + P1Y).astype(np.int32)

    # Remove points outside of image bounds
    colX = itbuffer[:, 0]
    colY = itbuffer[:, 1]
    itbuffer = itbuffer[(colX >= 0) & (colY >= 0) & (colX < imageW) & (colY < imageH)]

    # Get intensities from img ndarray
    itbuffer = itbuffer.astype(np.int32)  # Ensure x, y coordinates are integers
    itbuffer[:, 2] = img[itbuffer[:, 1], itbuffer[:, 0]]

    nonzero = np.sum(itbuffer[:, 2] != 0)
    # if nonzero==0 or np.isnan(nonzero) or len(itbuffer)==0:
    #     return 0
    # else:
    return nonzero / len(itbuffer)


scribe.remove_edges_from(scribe.edges)  # start anew, just in case
# we need to make edges between nodes within a connectedcomponent
for k in range(len(components)):
    # establish edges from the shortest distance between nodes, forward check
    # O(n^2) complexity
    for m in components[k].nodes:
        scribe.nodes[m]["component_id"] = k
        src = scribe.nodes[m]
        # three closest nodes
        ndist = [1e9, 1e9, 1e9]
        ndst = [-1, -1, -1]
        for n in components[k].nodes:
            dst = scribe.nodes[n]
            cdist = math.sqrt(
                math.pow(dst["pos_bitmap"][0] - src["pos_bitmap"][0], 2)
                + math.pow(dst["pos_bitmap"][1] - src["pos_bitmap"][1], 2)
            )
            if m != n:
                linepart = line_iterator(stroke, src["pos_bitmap"], dst["pos_bitmap"])
            # add the checking for line segment
            if (
                (m != n)
                and cdist < SLIC_SPACE * pow(PHI, 2) * 2
                and linepart > pow(PHI, -PHI)
            ):
                if cdist < ndist[2]:  # #1 shortest
                    ndist[0] = ndist[1]
                    ndist[1] = ndist[2]
                    ndist[2] = cdist
                    ndst[0] = ndst[1]
                    ndst[1] = ndst[2]
                    ndst[2] = n
                elif cdist >= ndist[2] and cdist <= ndist[1]:
                    ndist[0] = ndist[1]
                    ndist[1] = cdist
                    ndst[0] = ndst[1]
                    ndst[1] = n
                elif cdist < ndist[0]:
                    ndist[0] = cdist
                    ndst[0] = n
        # print(f'{m} to {ndst[0]}({ndist[0]:.2f}) {ndst[1]}({ndist[1]:.2f}) {ndst[2]}({ndist[2]:.2f})')
        filled = [False, False, False]
        for i in range(2, -1, -1):
            if ndist[i] != 1e9 and ndst[i] != -1:
                dst = scribe.nodes[ndst[i]]
                tvane = freeman(
                    dst["pos_bitmap"][0] - src["pos_bitmap"][0],
                    -(dst["pos_bitmap"][1] - src["pos_bitmap"][1]),
                )
                # // (i==1 and scribe.has_edge(m,ndst[2])==True ) # for 1+alpha
                if scribe.has_edge(m, ndst[i]):
                    filled[i] = True
                # this is for 2+alpha
                if (
                    (i == 2)
                    or (i == 1 and scribe.has_edge(ndst[2], ndst[1]) == False)
                    or (
                        i == 0
                        and scribe.has_edge(ndst[2], ndst[0]) == False
                        and scribe.has_edge(ndst[1], ndst[0]) == False
                    )
                ):
                    scribe.add_edge(
                        m,
                        ndst[i],
                        color="#00FF00",
                        weight=1e2 / ndist[i] / SLIC_SPACE,
                        vane=tvane,
                    )
                    # print(f'{m} to {ndst[i]}: {ndist[i]}')
                if (
                    filled[2] == False
                    and filled[1] == False
                    and i == (3 - RASM_EDGE_MAXDEG)
                ):
                    break

# draw_graph_edgelabel(scribe, 'pos_render', 8, '/shm/withedges.png', None)


def prune_edges(graph, hop):
    G = graph.copy()
    temp = G.copy()
    # the search
    for u, v in G.edges():
        tempU = G.degree(u)
        tempV = G.degree(v)
        if tempU >= 3 and tempV >= 3:
            temp.remove_edge(u, v)
            # print(f'edge {u} {v}: from {tempU} {tempV} to {temp.degree(u)} {temp.degree(v)} ')
            if (
                nx.has_path(temp, u, v)
                and temp.degree(u) < tempU
                and temp.degree(v) < tempV
            ):
                minlen = len(nx.shortest_path(temp, u, v))
                if (
                    minlen >= hop + 1
                ):  # minlen is number of nodes involved, so number of edges involved +1
                    G.remove_edge(u, v)
    return G


# scribe= prune_edges(scribe, 3)
# scribe= nx.minimum_spanning_tree(scribe, algorithm='kruskal')


def hex_or(color1, color2):
    int1 = int(color1.lstrip("#"), 16)
    int2 = int(color2.lstrip("#"), 16)
    result_int = int1 | int2
    result_hex = f"#{result_int:06X}"
    return result_hex


def hex_and(color1, color2):
    int1 = int(color1.lstrip("#"), 16)
    int2 = int(color2.lstrip("#"), 16)
    return int1 & int2


degree_rasm = scribe.degree()
scribe_dia = scribe.copy()
baseline_pos = np.mean(np.array([value[1] for value in pos.values()]))

for k in range(len(components)):
    components[k].node_start = components[k].nodes[0]
    intersect = False
    for n in range(len(components[k].nodes)):
        if abs(pos[components[k].nodes[n]][1] - baseline_pos) < SLIC_SPACE:
            intersect = True

    # valid rasm
    # large size
    # more likely to be close to or intersecting the baseline
    if intersect == True and components[k].area > pow(SLIC_SPACE, 2) * pow(
        PHI, 4
    ):  # or (abs(components[k].centroid-baseline_pos)[1] < SLIC_SPACE*pow(PHI,3) and components[k].area>pow(SLIC_SPACE,2)*pow(PHI,3)):
        for j in components[k].nodes:
            scribe_dia.nodes[j]["rasm"] = True

        components[k].node_start = components[k].nodes[0]
        for n in components[k].nodes:
            if (
                pos[n][0] > pos[components[k].node_start][0]
            ):  # rightmost node as starting node if it is still missing
                components[k].node_start = n
        scribe_dia.nodes[components[k].node_start][
            "color"
        ] = "#F00000"  # initialize with red
        # scribe.nodes[components[k].node_start]['color']= '#FFA500'

        # actually optimizing the starting node
        # scribe.nodes[components[k].node_start]['color']= '#FFA500' # reset to orange
        scribe_dia.nodes[components[k].node_start]["color"] = "#FFA500"
        graph = extract_subgraph(scribe, components[k].node_start)

        # Check if the component is tall
        if components[k].rect[3] / components[k].rect[2] > pow(PHI, 2):
            # If tall, prefer starting from the top
            smallest_degree_nodes = [
                node
                for node, _ in sorted(graph.degree(), key=lambda item: item[1])[
                    :RASM_CANDIDATE
                ]
            ]
            # node_start = min(smallest_degree_nodes, key=lambda node: pos[node][0]) # cari yang paling kanan (Zulhaj)
            node_start = min(
                smallest_degree_nodes, key=lambda node: pos[node][1]
            )  # cari yang paling atas (Fitri)
        else:
            # if stumpy, prefers starting close to median more to the right, but far away from centroid
            rightmost_nodes = sorted(
                [
                    node
                    for node in graph.nodes
                    if pos[node][0] > (components[k].centroid[0] - SLIC_SPACE)
                ],
                key=lambda node: pos[node][0],
                reverse=True,
            )[: int(RASM_CANDIDATE * PHI)]
            # Step 1: Get the rightmost nodes
            topmost_nodes = sorted(
                [
                    node
                    for node in rightmost_nodes
                    if pos[node][1] < (baseline_pos + SLIC_SPACE)
                ],
                key=lambda node: pos[node][1],
            )[: int(RASM_CANDIDATE * PHI)]
            # Zulhaj @jendralhxr
            smallest_degree_nodes = sorted(
                [node for node in topmost_nodes], key=lambda node: graph.degree(node)
            )[: int(RASM_CANDIDATE)]
            rightmost_nodes = sorted(
                [node for node in smallest_degree_nodes],
                key=lambda node: pos[node][0],
                reverse=True,
            )[: int(RASM_CANDIDATE / PHI)]
            # node_start = max(smallest_degree_nodes, key=lambda node: pdistance(pos[node], components[k].centroid))
            node_start = min(rightmost_nodes, key=lambda node: pos[node][1])
        # @FadhilatulFitriyah
        # Step 2: Get the topmost nodes from the rightmost nodes
        # topmost_nodes = sorted(rightmost_nodes, key=lambda node: pos[node][1])[:int(RASM_CANDIDATE)]
        # Step 3: Get the node with the node start from topmost nodes
        # node_start  = min(rightmost_nodes, key=lambda node: graph.degree(node))

        components[k].node_start = node_start
        # scribe.nodes[node_start]['color']= '#F00000' # starting node is red
        scribe_dia.nodes[node_start]["color"] = "#F00000"

    # valid diacritics
    # small size, but not too small (dirt)
    # away from median, but still relatively close
    # rather stumpy
    if (
        intersect == False
        and abs(components[k].centroid - baseline_pos)[1] < SLIC_SPACE * pow(PHI, 4)
        and abs(components[k].centroid - baseline_pos)[1] > SLIC_SPACE
        and (
            components[k].rect[3] / components[k].rect[2] < pow(PHI, 2)
            and components[k].rect[2] / components[k].rect[3] < pow(PHI, 2)
        )
        and (
            components[k].area < pow(SLIC_SPACE, 2) * pow(PHI, 5)
            or (
                len(components[k].nodes) == 1
                and components[k].area > pow(SLIC_SPACE, 2)
            )
        )
    ):  # small components (diacritics)
        # A: 1 dots, B: 2 dots, C: 3 dots; D: (perhaps) hamza
        for j in components[k].nodes:
            scribe_dia.nodes[j]["rasm"] = False
            scribe_dia.nodes[j]["color"] = "#008888"
            if components[k].area > pow(SLIC_SPACE, 2) * pow(PHI, 4):
                scribe_dia.nodes[j]["dia_size"] = "C"
            elif components[k].area > pow(SLIC_SPACE, 2) * pow(PHI, 3):
                scribe_dia.nodes[j]["dia_size"] = "B"
            else:
                scribe_dia.nodes[j][
                    "dia_size"
                ] = "A"  # approx pow(SLIC_SPACE,2)*pow(PHI,3)

        src_comp = k
        src_node = -1
        closest_comp = -1
        closest_dist = 1e9
        closest_node = -1
        closest_vane = -1
        for l in range(len(components)):
            if (k != l) and pdistance(
                components[k].centroid, components[l].centroid
            ) < SLIC_SPACE * pow(PHI, 5):
                for m in components[k].nodes:
                    for n in components[l].nodes:
                        tdist = pdistance(pos[m], pos[n])
                        tvane = freeman(pos[n][0] - pos[m][0], pos[n][1] - pos[m][1])
                        if (
                            tdist < closest_dist
                            and (tvane == 2 or tvane == 6)
                            and scribe.nodes[n]["rasm"] == True
                        ):
                            closest_comp = l
                            src_node = m
                            closest_node = n
                            closest_vane = tvane
                            closest_dist = tdist
        # print(f'comp {k} to {closest_comp} \t node {m} to {n}\t: {closest_dist} {closest_vane}')
        if closest_dist < SLIC_SPACE * pow(PHI, 4):
            scribe_dia.add_edge(
                src_node,
                closest_node,
                color="#0000FF",
                weight=1e2 / closest_dist / SLIC_SPACE,
                vane=closest_vane,
            )  # blue connecting edge
            if closest_vane == 6:  # diacritics over
                # scribe.nodes[closest_node]['color']= hex_or(scribe.nodes[closest_node]['color'], '#0000FF')
                scribe_dia.nodes[closest_node]["color"] = hex_or(
                    scribe_dia.nodes[closest_node]["color"], "#0000FF"
                )  # dark blue
            else:  # diacritics below
                # scribe.nodes[closest_node]['color']= hex_or(scribe.nodes[closest_node]['color'], '#000080')
                scribe_dia.nodes[closest_node]["color"] = hex_or(
                    scribe_dia.nodes[closest_node]["color"], "#000080"
                )  # light blue

    # edge cases, treat as rasm
    else:
        for j in components[k].nodes:
            scribe_dia.nodes[j]["rasm"] = True
        scribe_dia.nodes[components[k].node_start][
            "color"
        ] = "#F00000"  # initialize with red

# merging close diacritics
for i in range(len(components)):
    for j in range(i + 1, len(components)):
        dia_dist = pdistance(components[i].centroid, components[j].centroid)
        if dia_dist < SLIC_SPACE * PHI:
            components[i].area = components[i].area + components[j].area
            components[i].nodes = components[i].nodes + components[j].nodes
            for n in components[i].nodes:
                if components[i].area > pow(SLIC_SPACE, 2) * pow(PHI, 5):
                    scribe_dia.nodes[n]["dia_size"] = "C"
                else:
                    scribe_dia.nodes[n][
                        "dia_size"
                    ] = "B"  # approx pow(SLIC_SPACE,2)*pow(PHI,4)
            components[i].rect = (
                components[i].rect[0],
                components[i].rect[1],
                components[i].rect[2] + components[j].rect[2],
                components[i].rect[3] + components[j].rect[3],
            )

            components[j].centroid = (0, 0)  # components to be removed
            # print(f"gonna remove {j}")

# removing merged diacritics components
components = sorted(components, key=lambda x: x.centroid[0], reverse=True)
while components[-1].centroid == (0, 0):
    del components[-1]

degree_dia = scribe.degree()

# substroke identification
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d


# baris 797-893  ############################START HISTOGRAM
def find_histogram_min(img, ANGLE):
    projection_hist = np.zeros(img.shape[1], np.uint8)
    for x in range(img.shape[1] - 1, 0, -1):
        x_start = x
        x_end = x_start - math.tan(ANGLE) * img.shape[0]  # can start beyond image width
        for y_pos in range(img.shape[0]):
            x_pos = int(x_start - math.tan(ANGLE) * y_pos)
            if y_pos < img.shape[0] and x_pos < img.shape[1]:
                if ccv[y_pos][x_pos][0] == STROKEVAL:
                    projection_hist[x_start] += 1

    projection_hist_smoothed = gaussian_filter1d(projection_hist, pow(PHI, 3))
    valleys = find_peaks(-projection_hist_smoothed)[0]
    # plt.plot(projection_hist_smoothed, label="projection histogram")
    # plt.title("slanted projection histogram (smoothed) at angle "+f"{ANGLE:.4f}"+" rad")
    # plt.scatter(valleys, [projection_hist_smoothed[i] for i in valleys], color='red', marker='o', s=10)
    # peaks= find_peaks(projection_hist_smoothed)[0]

    return projection_hist_smoothed, valleys


# slanted projection histogram for segmenting the strokes
SLANT1 = 0
SLANT2 = 3.1415 / pow(PHI, 3)
COLOR_TRANS1 = "#10F010"
COLOR_TRANS2 = "#10A010"


ccv = cv.cvtColor(cue, cv.COLOR_GRAY2BGR)  # hasil = ccv
for n in range(len(components)):
    if scribe_dia.nodes[components[n].nodes[0]]["rasm"] == True:
        seed = pos[components[n].nodes[0]]
        cv.floodFill(
            ccv, None, seed, (STROKEVAL, THREVAL, THREVAL), loDiff=(5), upDiff=(5)
        )

hist1, valleys1 = find_histogram_min(ccv, SLANT1)  # red
hist2, valleys2 = find_histogram_min(ccv, SLANT2)  # green

# plt.plot(hist1, color='red', label="projection angle "+f"{SLANT1:.4f}"+" rad")
# plt.plot(hist2, color='green', label="projection angle "+f"{SLANT2:.4f}"+" rad")
# plt.scatter(valleys1, [hist1[i] for i in valleys1], color='red', marker='o', s=10)
# plt.scatter(valleys2, [hist2[i] for i in valleys2], color='green', marker='o', s=10)
# plt.legend()
# plt.title("slanted projection histogram")


def find_closest_node(G, midx, midy):
    min_distance = float("inf")
    closest_node = None
    for n in G.nodes:
        pos = G.nodes[n]["pos_bitmap"]
        distance = np.sqrt((pos[0] - midx) ** 2 + (pos[1] - midy) ** 2)
        if distance < min_distance:
            min_distance = distance
            closest_node = n
    return closest_node


for x_start in valleys1:
    x_end = x_start - math.tan(SLANT1) * ccv.shape[0]
    # bitmap position
    print("valleys1 x_start,x_end", x_start, x_end)
    if x_end >= 0:
        active_stroke = False
        for y_pos in range(ccv.shape[0]):
            x_pos = int(x_start - math.tan(SLANT1) * y_pos)
            if y_pos < ccv.shape[0] and x_pos < ccv.shape[1]:
                if ccv[y_pos][x_pos][0] == STROKEVAL:
                    ccv[y_pos][x_pos][2] = FOCUSVAL
                    if active_stroke == False:
                        active_stroke = True
                        cut_start = (x_pos, y_pos)
                        print("cut_start", cut_start)

                else:
                    if active_stroke == True:
                        active_stroke = False
                        cut_end = (x_pos, y_pos)
                        midpoint = (
                            (cut_end[0] + cut_start[0]) / 2,
                            (cut_end[1] + cut_start[1]) / 2,
                        )
                        # bitmap pos
                        print("midpoint", midpoint)
                        transition_node = find_closest_node(
                            scribe_dia, midpoint[0], midpoint[1]
                        )
                        print("transition_node", transition_node)
                        if scribe_dia.nodes[transition_node]["rasm"] == True:
                            scribe_dia.nodes[transition_node]["color"] = COLOR_TRANS1
print(
    "SLANT 2 shape ", SLANT2, ccv.shape[0], ccv.shape[1]
)  # SLANT2= 3.1415 / pow(PHI,3)
for x_start in valleys2:
    x_end = x_start - math.tan(SLANT2) * ccv.shape[0]
    print("valleys2 x_start,x_end", x_start, x_end)

    if x_end >= 0:
        active_stroke = False
        for y_pos in range(ccv.shape[0]):
            x_pos = int(x_start - math.tan(SLANT2) * y_pos)
            if y_pos < ccv.shape[0] and x_pos < ccv.shape[1]:
                if ccv[y_pos][x_pos][0] == STROKEVAL:
                    ccv[y_pos][x_pos][1] = FOCUSVAL
                    if active_stroke == False:
                        active_stroke = True
                        cut_start = (x_pos, y_pos)
                        print("valleys2 cut_start", cut_start)
                else:
                    if active_stroke == True:
                        active_stroke = False
                        cut_end = (x_pos, y_pos)
                        midpoint = (
                            (cut_end[0] + cut_start[0]) / 2,
                            (cut_end[1] + cut_start[1]) / 2,
                        )
                        print("valleys2 midpoint", midpoint)
                        transition_node = find_closest_node(
                            scribe_dia, midpoint[0], midpoint[1]
                        )
                        print("valleys2 transition_node", transition_node)
                        if scribe_dia.nodes[transition_node]["rasm"] == True:
                            scribe_dia.nodes[transition_node]["color"] = COLOR_TRANS2

draw(ccv)

###### graph construction from line image ends here
###### ----------------------------------------------------
###### path finding routines starts here


def path_vane_nodes(G, path):  # if path is written as series of nodes
    pathstring = ""
    for i in range(len(path) - 1):
        src = G.nodes[path[i]]
        dst = G.nodes[path[i + 1]]
        tvane = freeman(
            dst["pos_bitmap"][0] - src["pos_bitmap"][0],
            -(dst["pos_bitmap"][1] - src["pos_bitmap"][1]),
        )
        pathstring += str(tvane)
    return pathstring


def get_edge_colors_of_node(G, node):
    edges = G.edges(node, data=True)
    colors = [
        (edge[0], edge[1], edge[2].get("color", "No color assigned")) for edge in edges
    ]
    return colors


def find_diacritics_edges(G, node):
    neighbors = list(G.neighbors(node))
    if len(neighbors) >= 2:
        for neighbor in neighbors:
            if G.edges[(node, neighbor)]["color"] == "#0000FF":
                size = G.nodes[neighbor]["dia_size"]
                if pos[neighbor][1] > pos[node][1]:
                    size = size.lower()
                return size
    return ""


def path_cut(path, i):  # path dari masing2 component i akan dipotong menjadi path_cut
    path_cut = []
    for n in path:
        # vane code
        src = n[0]
        dst = n[1]
        print("n", n[0], n[1])
        # misal ambil dari src, search di area berapa pada hist angle area
        # jika ketemu area baru , bikin path_cut baru
        # jika ketemu area yg sama, append path_cut yg lama
        # hasilnya path_cut merupakan potongan


def path_vane_edges(G, path):  # if path is written is written as series of edges
    visited = []
    pathstring = ""
    for n in path:
        # vane code
        src = n[0]
        dst = n[1]
        print("n", n[0], n[1])
        # freeman(dx, dy)  ada pos[dst][0]
        tvane = freeman(pos[dst][0] - pos[src][0], -(pos[dst][1] - pos[src][1]))
        print("tvane ", tvane)
        print("pos", pos[dst][0], pos[src][0], pos[dst][1], pos[src][1])
        G.edges[n]["vane"] = tvane
        if G.edges[n]["color"] == "#00FF00":  # main stroke
            pathstring += str(tvane)

        mark = ""
        if src not in visited:
            mark = find_diacritics_edges(G, src)
            if mark != "":
                pathstring += mark
        visited.append(src)
        if dst not in visited:
            mark = find_diacritics_edges(G, dst)
            if mark != "":
                pathstring += mark
        visited.append(dst)

    return pathstring


def bfs_with_closest_priority(G, start_node):
    visited = set()  # track visited nodes
    edges = []  # traversed edges
    priority_queue = []  # Use heapq for priority queue
    heapq.heappush(
        priority_queue, (0, start_node)
    )  # Push the start node with priority 0
    while priority_queue:
        # Get the node with the highest priority (smallest distance)
        _, current_node = heapq.heappop(priority_queue)

        if current_node not in visited:
            visited.add(current_node)
            # print(f"Visited {current_node}")  # Do something with the node

            # Explore neighbors
            for neighbor in G.neighbors(current_node):
                if neighbor not in visited:
                    distance = pdistance(pos[neighbor], pos[neighbor])
                    heapq.heappush(priority_queue, (distance, neighbor))
                    edges.append((current_node, neighbor))

    # return visited # if handling the nodes
    return edges  # if handling the edges


# drawing the rasm graph
from PIL import ImageFont, ImageDraw, Image

FONTSIZE = 24
print("len(components)):", len(components))
for i in range(len(components)):
    rasm = ""
    if scribe_dia.nodes[components[i].node_start]["rasm"] == True:
        rasm = ""
        remainder_stroke = ""

        # path finding along rasm
        node_start = components[i].node_start
        path = list(
            bfs_with_closest_priority(extract_subgraph(scribe, node_start), node_start)
        )
        # cut the path for each component
        # info path

        # bikin file csv untuk stiap component, untuk dipotong2 berdasarkan valley dan angle
        remainder_stroke = path_vane_edges(scribe_dia, path)

        if len(remainder_stroke) > 2:
            print("i rs ", i, remainder_stroke)

        # refer to rasm2hurf.py
        # rule-based minimum feasible pattern
        # rasm= stringtorasm_LEV(remainder_stroke)
        # using LSTM model
        # rasm=stringtorasm_LSTM(remainder_stroke)
        # using LCS table
        # rasm= stringtorasm_LCS(remainder_stroke)
        # rasm= stringtorasm_MC_jagokandang(remainder_stroke)

        # ccv= cv.cvtColor(cue, cv.COLOR_GRAY2BGR)
        # seed= pos[components[i].node_start]
        # cv.floodFill(ccv, None, seed, (STROKEVAL,STROKEVAL,STROKEVAL), loDiff=(5), upDiff=(5))
        # pil_image = Image.fromarray(cv.cvtColor(ccv, cv.COLOR_BGR2RGB))
        # font = ImageFont.truetype("/usr/share/fonts/truetype/noto/NotoSansArabic-Regular.ttf", FONTSIZE)
        # fontsm = ImageFont.truetype("/usr/share/fonts/truetype/noto/NotoSansArabic-Regular.ttf", FONTSIZE-12)
        # drawPIL = ImageDraw.Draw(pil_image)
        # #drawPIL.text((components[i].centroid[0]-FONTSIZE/2, components[i].centroid[1]-FONTSIZE), rasm, font=font, fill=(0, 200, 0))
        # drawPIL.text((components[i].centroid[0]-FONTSIZE/2, components[i].centroid[1]), str(i), font=fontsm, fill=(0, 200, 200))
        # # Convert back to Numpy array and switch back from RGB to BGR
        # ccv= np.asarray(pil_image)
        # ccv= cv.cvtColor(ccv, cv.COLOR_RGB2BGR)
        # draw(ccv)
        # cv.imwrite(imagename+'highlight'+str(i).zfill(2)+'.png', ccv)

graphfile = "graph-" + imagename + ext
draw_graph_edgelabel(scribe_dia, "pos_render", 8, graphfile, None)


# ada path, ccv, cue, scribe
# ada valley1, valley2


# #ambil data hasil csv dari output angle-histogram.py
# rot_pos = []
# rot_pos_end = []
# filename2 = 'out-rot_pos.csv' #out-timer-Larutan 200 (1) - Target.mp4.csv'
# with open(filename2, newline='') as csvfile:
#     spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
#     for row in spamreader:
#         print(row)
#         area_x_start = (row[0])
#         area_y_start = (row[1])
#         area_x_end = (row[2])
#         area_y_end = (row[3])
#         print("area_x_start ",area_x_start)
#         print("area_y_start ",area_y_start)
#         print("area_x_end ",area_x_end)
#         print("area_y_end ",area_y_end)
#         #writerow((int(rot_pos[r][0]),int(rot_pos[r][1]),int(rot_pos_end[r][0]),int(rot_pos_end[r][1])))
#         rot_pos.append((row[0],row[1])) #kumpulan titik hasil rotasi
#         rot_pos_end.append((row[2],row[3]))


# # import titik random dari csv?
# import csv
# #output rotpos as csv
# with open('out-rot_pos'+''+'.csv', 'w', newline='') as csvfile:
#     spamwriter = csv.writer(csvfile, delimiter=' ',quotechar='|', quoting=csv.QUOTE_MINIMAL)


# lalu batas1_x dan batas2_x diaplikasikan pada rot_pos
# input kumpulan nilai batas1_x dan batas2_x , lalu di gambar kotak2nya atau di crop

# rot_pos adalah hasil putaran

# yang ini :
# kumpulan nilai batas1_x dan batas2_x   ( ada banyak nilai, diambil dari csv)
# ........
#     if batas2_x == list_space_angle_start[r][0] and r>0 :
#         print("RECTA ",(rot_pos[r-1]),(rot_pos_end[r]))  #rectangle(img,(x1,y1),(x2,y2),color)
#         putdraw = cv2.rectangle(putdraw,(int(rot_pos[r][0]),int(rot_pos[r-1][1])),(int(rot_pos_end[r-1][0]),int(rot_pos_end[r][1])) , (0,255,255))
#     elif batas2_x == list_space_angle_start[r][0] and r==0 :
#     # if r  == 0 :
#         # x2 = putdraw.shape[1] #width
#         print("RECTA WIDTH")
#         putdraw = cv2.rectangle(putdraw,(int(rot_pos[r][0]),int(rot_pos[r-1][1])),(putdraw.shape[1],int(rot_pos_end[r][1])) , (0,255,255))


# # ##################################
# # ## ambil data dari hasil CNN
# import cnn48syairperahu
# import rasm2hurf

# from keras.models import load_model
# model = load_model('syairperahu.keras')

# IMG_WIDTH= 48 # window size of the trained model
# IMG_HEIGHT= IMG_WIDTH

# # scale takes into account SLIC_SPACE and RESIZE_FACTOR
# # can also be checked from mean_closest_distance
# def predictfromimage(cueimage, pos, scale):
#     cx= int (pos[0])
#     cy= int (pos[1])
#     cy_min= int(cy -IMG_WIDTH/2*scale )
#     cy_max= int(cy +IMG_WIDTH/2*scale )
#     cx_min= int(cx -IMG_WIDTH/2*scale )
#     cx_max= int(cx +IMG_WIDTH/2*scale )
#     if (cy_min<0):
#         cy_min= 0
#         cy_max= IMG_HEIGHT * scale
#     if (cy_max > cueimage.shape[0]):
#         cy_max= cueimage.shape[0]
#         cy_min= cy_max - IMG_HEIGHT*scale
#     if (cx_min<0):
#         cx_min= 0
#         cx_max= IMG_WIDTH * scale
#     if (cx_max > cueimage.shape[1]):
#         cx_max= cueimage.shape[1]
#         cx_min= cx_max - IMG_HEIGHT*scale

#     roi= cueimage[int(cy_min):int(cy_max),int(cx_min):int(cx_max)]
#     roi= cv.resize(roi, None, fx=1/scale, fy=1/scale, interpolation=cv.INTER_LINEAR)
#     #draw(roi)
#     roi = roi / THREVAL
#     roi = np.expand_dims(roi, axis=0)
#     roi = np.expand_dims(roi, axis=-1)

#     prediction = model.predict(roi)
#     predicted_class = np.argmax(prediction, axis=-1)[0]
#     return hurf[int(predicted_class)]

# for i in scribe_dia.nodes:
#     scribe_dia.nodes[i]['hurf']= ' '

# _, cuecnn = cv.threshold(cue, 20, 255, cv.THRESH_BINARY)

# for i in scribe_dia.nodes:
#     if scribe_dia.nodes[i]['rasm']==True:
#         pos= scribe_dia.nodes[i]['pos_bitmap']
#         scribe_dia.nodes[i]['hurf']= predictfromimage(cuecnn, scribe_dia.nodes[i]['pos_bitmap'], 1.333333333333333333)
#         print(f"node{i}: {scribe_dia.nodes[i]['hurf']} ada di {pos[0]},{pos[1]}")
#     else:
#         scribe_dia.nodes[i]['hurf']= ' '

# draw_graph_edgelabel(scribe_dia, 'pos_render', 8, '/shm/dengarkan-nomer.png', None)
# draw_graph_edgelabel_ara(scribe_dia, 'pos_render', 8, '/shm/dengarkan-hurfcnn.png', 'hurf')
