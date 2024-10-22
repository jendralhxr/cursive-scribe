# usage: python -u features.py <inputimage> | tee <outputrasm>
import os
#os.chdir("/shm")
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import sys
import math
import heapq
from collections import deque 

# freeman code going anti-clockwise like trigonometrics angle
#    3   2   1
#      \ | /
#    4 ------0
#      / | \
#    5   6   7

PHI= 1.6180339887498948482 # ppl says this is a beautiful number :)

RESIZE_FACTOR=2
SLIC_SPACE= 3
SLIC_SPACE= SLIC_SPACE*RESIZE_FACTOR
WHITESPACE_INTERVAL= 4

RASM_EDGE_MAXDEG= 2
RASM_CANDIDATE= 6

THREVAL= 60
CHANNEL= 2

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

kernel_size=2
canny_threshold1=100
canny_threshold2=200
edges = cv.Canny(gray, canny_threshold1, canny_threshold2)
kernel = np.ones((kernel_size, kernel_size), np.uint8)
eroded_image = cv.erode(gray, kernel, iterations=1)
edge_mask = cv.bitwise_not(edges)
selective_eroded = cv.bitwise_and(eroded_image, eroded_image, mask=edge_mask)
ret, gray= cv.threshold(selective_eroded,1,THREVAL,cv.THRESH_BINARY)
dilation_kernel = np.ones((1,2), np.uint8) # alifah dan fitri
gray = cv.dilate(gray, dilation_kernel, iterations=1)
cue= gray.copy()
render = cv.cvtColor(gray, cv.COLOR_GRAY2BGR)

#SLIC
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

STROKEVAL= 160

pos = nx.get_node_attributes(scribe,'pos_bitmap')
components=[]
for n in range(scribe.number_of_nodes()):
    # fill
    seed= pos[n]
    ccv= gray.copy()
    cv.floodFill(ccv, None, seed, STROKEVAL, loDiff=(5), upDiff=(5))
    _, ccv = cv.threshold(ccv, 100, STROKEVAL, cv.THRESH_BINARY)
    mu= cv.moments(ccv)
    if mu['m00'] > pow(SLIC_SPACE,2)*PHI:
        mc= (int(mu['m10'] / (mu['m00'])), int(mu['m01'] / (mu['m00'])))
        pd= pdistance(seed, mc)
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
# disp = cv.cvtColor(gray, cv.COLOR_GRAY2BGR)
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

#     rasm= components[n].mat[\
#         components[n].rect[1]:components[n].rect[1]+components[i].rect[3],\
#         components[n].rect[0]:components[n].rect[0]+components[i].rect[2]]
#     cv.imwrite(str(n)+'.png', rasm)
# draw(disp) 
#cv.imwrite(imagename+'-disp.png', disp)    

# draw each components separately, sorted right to left
# for n in range(len(components)):
#     ccv= cv.cvtColor(gray, cv.COLOR_GRAY2BGR)
#     if components[n].node_start!=-1:
#         seed= pos[components[n].node_start]
#         cv.floodFill(ccv, None, seed, (STROKEVAL,STROKEVAL,STROKEVAL), loDiff=(5), upDiff=(5))
#         cv.putText(ccv, str(n), components[n].centroid, cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 2)
#         draw(ccv) # along with the neighbor

def draw_graph(graph, posstring, scale):
    # nodes
    plt.figure(figsize=(3*scale,3)) 
    positions = nx.get_node_attributes(graph,posstring)
    area= np.array(list(nx.get_node_attributes(graph, 'area').values()))
    # edges
    node_colors = nx.get_node_attributes(graph,'color').values()
    edge_colors = nx.get_edge_attributes(graph,'color').values()
    weights = np.array(list(nx.get_edge_attributes(graph,'weight').values()))
    #plt.figure(figsize=(width/12,height/12)) 
    nx.draw(graph, 
            # nodes' param
            pos=positions, 
            with_labels=True, 
            node_color= node_colors,
            node_size=area*25,
            font_size=8,
            # edges' param
            edge_color=edge_colors, 
            width=weights*2,
            )

def draw_graph_edgelabel(graph, posstring, scale, filename):
    plt.figure(figsize=(4*scale,4)) 
    # nodes
    if posstring is None:
        positions = nx.spring_600layout(graph)
    else:
        positions = nx.get_node_attributes(graph,posstring)
    
    #area= np.array(list(nx.get_node_attributes(graph, 'area').values()))
    edge_lbls= nx.get_edge_attributes(graph, 'vane')
    # edges
    node_colors = nx.get_node_attributes(graph,'color').values()
    edge_colors = nx.get_edge_attributes(graph,'color').values()
    weights = np.array(list(nx.get_edge_attributes(graph,'weight').values()))
    #plt.figure(figsize=(width/12,height/12)) 
    nx.draw(graph, 
            # nodes' param
            pos=positions, 
            with_labels=True, 
            node_color= node_colors,
            node_size=100,
            font_size=6,
            # edges' param
            edge_color=edge_colors, 
            width=weights*2,
            )
    nx.draw_networkx_edge_labels(graph, 
            pos= positions,
            edge_labels=edge_lbls, 
            font_size=5,
            font_color='red')
    if filename is not None:
        plt.savefig(filename, dpi=300)

def line_iterator(img, point0, point1):
    for n in range(WHITESPACE_INTERVAL,1,-1):
        dx= (point1[0]-point0[0])/n
        dy= (point1[1]-point0[1])/n
        has_dark= False
        for i in range(1,n):
            x= int (point0[0]+i*dx)
            y= int (point0[1]+i*dy)
            # 3x3 kernel to account for the floating point rounding
            #print(f"{n} {i} -- ({x},{y}) {img[y,x]}")
            if img[y,x]==0 and img[y,x+1]==0 and img[y,x-1]==0 and\
                img[y+1,x]==0 and img[y+1,x+1]==0 and img[y+1,x-1]==0 and\
                img[y-1,x]==0 and img[y-1,x+1]==0 and img[y-1,x-1]==0:
                has_dark= True
                break
        #print(f"{n} space {has_dark}")
        #if has_dark==False:  # suka nyambung/lengket
        if has_dark==True:	 # suka putus 		
            break
    return has_dark
    
scribe.remove_edges_from(scribe.edges) # start anew, just in case
# we need to make edges between nodes within a connectedcomponent
for k in range(len(components)):
    # establish edges from the shortest distance between nodes, forward check
    # O(n^2) complexity
    for m in components[k].nodes:
        scribe.nodes[m]['component_id']=k
        src= scribe.nodes()[m]
        # three closest nodes
        ndist=[1e9, 1e9, 1e9]
        ndst= [-1, -1, -1]
        for n in components[k].nodes:
            dst= scribe.nodes()[n]
            cdist= math.sqrt( math.pow(dst['pos_bitmap'][0]-src['pos_bitmap'][0],2) + math.pow(dst['pos_bitmap'][1]-src['pos_bitmap'][1],2) )
            has_dark= line_iterator(cue, src['pos_bitmap'], dst['pos_bitmap'])
            # add the checking for line segment
            if (m!=n) and cdist<SLIC_SPACE*pow(PHI,2): #and has_dark==False:
                if cdist<ndist[2]: # #1 shortest
                    ndist[0]= ndist[1]
                    ndist[1]= ndist[2]
                    ndist[2]= cdist
                    ndst[0]= ndst[1]
                    ndst[1]= ndst[2]
                    ndst[2]= n
                elif cdist>=ndist[2] and cdist<=ndist[1]:
                    ndist[0]= ndist[1]
                    ndist[1]= cdist
                    ndst[0]= ndst[1]
                    ndst[1]= n
                elif cdist<ndist[0]:
                    ndist[0]= cdist
                    ndst[0]= n
        #print(f'{m} to {ndst[0]}({ndist[0]:.2f}) {ndst[1]}({ndist[1]:.2f}) {ndst[2]}({ndist[2]:.2f})')
        filled=[False, False, False]
        for i in range(2, -1, -1):
            if ndist[i]!=1e9 and ndst[i]!=-1:
                dst= scribe.nodes()[ndst[i]]
                tvane= freeman(dst['pos_bitmap'][0]-src['pos_bitmap'][0], -(dst['pos_bitmap'][1]-src['pos_bitmap'][1]))
                # // (i==1 and scribe.has_edge(m,ndst[2])==True ) # for 1+alpha
                if scribe.has_edge(m, ndst[i]):
                    filled[i]= True
                # this is for 2+alpha
                if (i==2) or \
                   (i==1 and scribe.has_edge(ndst[2],ndst[1])==False ) or \
                   (i==0 and scribe.has_edge(ndst[2],ndst[0])==False and scribe.has_edge(ndst[1],ndst[0])==False):
                    scribe.add_edge(m, ndst[i], color='#00FF00', weight=1e2/ndist[i]/SLIC_SPACE, vane=tvane)
                    #print(f'{m} to {ndst[i]}: {ndist[i]}')            
                if filled[2]==False and filled[1]==False and i==(3-RASM_EDGE_MAXDEG):
                    break
                    
degree_rasm= scribe.degree()

def prune_edges(graph, hop):
    G= graph.copy()
    temp= G.copy()
    # the search
    for u, v in G.edges():
        tempU= G.degree(u)
        tempV= G.degree(v)
        if tempU>=3 and tempV>=3:
            temp.remove_edge(u, v)
            #print(f'edge {u} {v}: from {tempU} {tempV} to {temp.degree(u)} {temp.degree(v)} ')
            if nx.has_path(temp, u, v) and temp.degree(u)<tempU and temp.degree(v)<tempV:
                minlen= len(nx.shortest_path(temp, u, v))
                if minlen >= hop+1: # minlen is number of nodes involved, so number of edges involved +1
                    G.remove_edge(u, v)
    return(G)

#scribe= prune_edges(scribe, 3)
#scribe= nx.minimum_spanning_tree(scribe, algorithm='kruskal')

degree_rasm= scribe.degree()
scribe_dia= scribe.copy()

def hex_or(color1, color2):
    int1 = int(color1.lstrip('#'), 16)
    int2 = int(color2.lstrip('#'), 16)
    result_int = int1 | int2
    result_hex = f'#{result_int:06X}'
    return result_hex

def hex_and(color1, color2):
    int1 = int(color1.lstrip('#'), 16)
    int2 = int(color2.lstrip('#'), 16)
    return int1 & int2
    


# finding diacritics connection for small components
# and update extreme nodes for large components
for k in range(len(components)):
    if components[k].area<pow(SLIC_SPACE,2)*pow(PHI,4) or len(components[k].nodes)<=4: # small components (diacritics)
        for j in components[k].nodes:
            scribe.nodes[j]['rasm']=False
        src_comp= k
        src_node= -1
        closest_comp= -1
        closest_dist= 1e9
        closest_node= -1
        closest_vane= -1
        for l in range(len(components)):
            if (k!=l) and pdistance(components[k].centroid, components[l].centroid)<SLIC_SPACE*pow(PHI,5):
                for m in components[k].nodes:
                    for n in components[l].nodes:
                        tdist= pdistance(pos[m], pos[n])
                        tvane= freeman(pos[n][0]-pos[m][0], pos[n][1]-pos[m][1])
                        if tdist<closest_dist and (tvane==2 or tvane==6) and scribe.nodes[n]['rasm']==True:
                            closest_comp= l
                            src_node= m
                            closest_node= n
                            closest_vane= tvane
                            closest_dist= tdist
        #print(f'comp {k} to {closest_comp} \t node {m} to {n}\t: {closest_dist} {closest_vane}')            
        if closest_dist<SLIC_SPACE*pow(PHI,4):
            scribe_dia.add_edge(src_node, closest_node, color='#0000FF', weight=1e2/closest_dist/SLIC_SPACE, vane=closest_vane)
            if closest_vane==6: # diacritics over
                scribe.nodes[closest_node]['color']= hex_or(scribe.nodes[closest_node]['color'], '#0000FF')
                scribe_dia.nodes[closest_node]['color']= hex_or(scribe_dia.nodes[closest_node]['color'], '#0000FF')
            else: # diacritics below
                scribe.nodes[closest_node]['color']= hex_or(scribe.nodes[closest_node]['color'], '#000080')
                scribe_dia.nodes[closest_node]['color']= hex_or(scribe_dia.nodes[closest_node]['color'], '#000080')
    
    else: # large ones, updating the starting node
        raddist_start=[]
        # calculate the distances
        for m in components[k].nodes:
            if pos[m][0] > components[k].centroid[0]:
                raddist_start.append( (pdistance(pos[components[k].node_end], pos[m]), m) )
            #    print(f'comp{k} node{m}')
            radmax3_start= heapq.nlargest(3, raddist_start, key=lambda x:x[0])
            #print(radmax3_start)
        # find the more appropriate starting node
        flag= False
        for d in range(1,RASM_EDGE_MAXDEG): # starting node needs to have the smallest degree if possible
            for e in radmax3_start:
                if degree_rasm(e[1])==d and e[1]!=-1:
                    #print(f'comp{k}_start: {components[k].node_start} -> {e[1]}')
                    components[k].node_start= e[1]
                    scribe.nodes[components[k].node_start]['color']= '#F00000'
                    scribe_dia.nodes[components[k].node_start]['color']= '#F00000'
                    flag= True
                    break
            if flag:
                break
        
degree_dia= scribe.degree()

# draw_graph(scribe_dia, 'pos_render', 8)

def extract_subgraph(G, start): # for a connected component
    if start!=-1:
        connected_component = nx.node_connected_component(G, start)
        connected_subgraph = G.subgraph(connected_component)
    return connected_subgraph.copy()


def extract_subgraph2(G, start, end): # for a hurf inside a rasm, naive
    paths = []
    nodes_in_paths = set()
    edges_in_paths = set()
    for path in paths:
        nodes_in_paths.update(path)
        edges_in_paths.update(zip(path, path[1:]))
    subgraph = G.subgraph(nodes_in_paths).copy()
    #subgraph = G.edge_subgraph(edges_in_paths).copy()
    return subgraph

def extract_subgraph3(G, start, end): # for a hurf inside a rasm, handling branches
    visited = [start,end]
    queue = [start, end]

    while queue:
        node = queue.pop(0)
        crossing_start= all(start in path for path in nx.all_simple_paths(G, node, end))
        crossing_end  = all(end in path for path in nx.all_simple_paths(G, node, start))
        #print(f"{node} s{crossing_start} e{crossing_end}")
        if (node not in visited) and (crossing_start==False) and (crossing_end==False) \
            or node==start  or node==end:
            visited.append(node)
    
            for neighbor in G.neighbors(node):
                if neighbor not in visited:
                    queue.append(neighbor)
    
    subgraph = G.subgraph(visited).copy()
    return subgraph

    
def edge_attributes(G):
    if isinstance(G,nx.MultiGraph) or isinstance(G,nx.MultiDiGraph):
        for u,v,k, attrs in G.edges(keys=True, data=True):
            print(f"({u}, {v}, {k}) {attrs}")
    elif isinstance(G,nx.Graph) or isinstance(G,nx.DiGraph):
        for u, v, attrs in G.edges(data=True):
            print(f"({u}, {v}) {attrs}")
    
def path_vane_nodes(G, path): # if path is written as series of nodes
    pathstring=''
    for i in range(len(path) - 1):
        src= G.nodes()[path[i]]
        dst= G.nodes()[path[i+1]]
        tvane= freeman(dst['pos_bitmap'][0]-src['pos_bitmap'][0], -(dst['pos_bitmap'][1]-src['pos_bitmap'][1]))
        pathstring+=str(tvane)
    return pathstring

def get_edge_colors_of_node(G, node):
    edges = G.edges(node, data=True)
    colors = [(edge[0], edge[1], edge[2].get('color', 'No color assigned')) for edge in edges]
    return colors

def path_vane_edges(G, path): # if path is written is written as series of edges
    pathstring=''
    for n in path:
        # vane code
        src= G.nodes()[n[0]]
        dst= G.nodes()[n[1]]
        tvane= freeman(dst['pos_bitmap'][0]-src['pos_bitmap'][0], -(dst['pos_bitmap'][1]-src['pos_bitmap'][1]))
        G.edges[n]['vane']=tvane
        scribe_dia.edges[n]['vane']=tvane
        if (G.edges[n]['color']=='#00FF00'): # main stroke
            pathstring+=str(tvane)
        
        # diacritics mark
        # may need duplicate check so that not printed twice if node is revisited from another edge(s)
        diacolor= hex_and(src['color'], '#0000FF')
        if diacolor:
            if diacolor == 255:
                pathstring+='-'
            elif diacolor == 128:
                pathstring+='+'
        else:
            diacolor= hex_and(dst['color'], '#0000FF')
            if diacolor:
                if diacolor == 255:
                    pathstring+='-'
                elif diacolor == 128:
                    pathstring+='+'
    return pathstring

###### graph construction from line image ends here

###### path finding routines starts here
# breadth-first search
def custom_bfs_dfs(graph, start_node):
    queue = deque([start_node])  # Initialize the queue with the start node
    visited = set([start_node])  # Set to keep track of visited nodes
    edges = []  # List to store edges

    def dfs(node):
        stack = [node]  # Use a stack for the DFS traversal
        branch_edges = []
        while stack:
            current = stack.pop()
            if current not in visited:
                visited.add(current)
                # Explore all neighbors of the current node
                for neighbor in graph.neighbors(current):
                    if neighbor not in visited:
                        stack.append(neighbor)
                        branch_edges.append((current, neighbor))  # Add the edge to the branch
        return branch_edges

    while queue:
        if (pos[queue[-1]][0] > pos[queue[0]][0]):
            node = queue.pop()
        else:
            node = queue.popleft()  # Get the next node from the BFS queue
        neighbors = list(graph.neighbors(node))  # Get neighbors of the current node
        unvisited_neighbors = [neighbor for neighbor in neighbors if neighbor not in visited]
        unvisited_neighbors.sort(key=lambda x: pos[x][0], reverse=True)
        #unvisited_neighbors.sort(key=lambda x: pos[x][0])
        
        # print all the branch first then traverse
        if unvisited_neighbors:
            for neighbor in unvisited_neighbors:
                # Add the edge from the current node to the neighbor
                edges.append((node, neighbor))
                if len(list(graph.neighbors(neighbor))) > 1:  # If there's a branch
                    branch_edges = dfs(neighbor)  # Explore the branch using DFS
                    edges.extend(branch_edges)  # Add the edges found during DFS
                else:
                    visited.add(neighbor)  # Mark the neighbor as visited
                    queue.append(neighbor)  # Add the neighbor to the BFS queue

    return edges

def bfs_with_closest_priority(G, start_node):
    visited = set()  # track visited nodes
    edges = [] # traversed edges
    priority_queue = []  # Use heapq for priority queue
    heapq.heappush(priority_queue, (0, start_node))  # Push the start node with priority 0

    while priority_queue:
        # Get the node with the highest priority (smallest distance)
        _, current_node = heapq.heappop(priority_queue)
        
        if current_node not in visited:
            visited.add(current_node)
            #print(f"Visited {current_node}")  # Do something with the node
            
            # Explore neighbors
            for neighbor in G.neighbors(current_node):
                if neighbor not in visited:
                    distance = pdistance(pos[neighbor], pos[neighbor])
                    heapq.heappush(priority_queue, (distance, neighbor))
                    edges.append((current_node, neighbor))
    
    #return visited # if handling the nodes
    return edges # if handling the edges


# drawing the rasm graph
from PIL import ImageFont, ImageDraw, Image
from rasm2hurf import stringtorasm_MC_jagokandang

FONTSIZE= 24

for i in range(len(components)):
    rasm=''
    remainder_stroke=''
    if len(components[i].nodes)>=2: # small alifs are often sometimes only 2-nodes big
        if components[i].node_start==-1: # in case of missing starting node
            #node_start_pos=(0,0)
            components[i].node_start=components[i].nodes[0]
            for n in components[i].nodes:
                if pos[n][0] > pos[components[i].node_start][0]: # rightmost node as starting node if it is still missing
                    components[i].node_start= n
        
        else: # actually optimizing the starting node
            scribe.nodes[components[i].node_start]['color']= '#FFA500' # reset to orange
            scribe_dia.nodes[components[i].node_start]['color']= '#FFA500'
            graph= extract_subgraph(scribe, components[i].node_start)

            # Check if the component is tall
            if (components[i].rect[3] / components[i].rect[2] > pow(PHI, 2)):
                # If tall, prefer starting from the top
                smallest_degree_nodes = [node for node, _ in sorted(graph.degree(), key=lambda item: item[1])[:RASM_CANDIDATE]]
                #node_start = min(smallest_degree_nodes, key=lambda node: pos[node][0]) # cari yang paling kanan (Zulhaj)
                node_start = min(smallest_degree_nodes, key=lambda node: pos[node][1]) # cari yang paling atas (Fitri)

			else: 
                # if stumpy, prefers starting close to median more to the right, but far away from centroid
                rightmost_nodes = sorted([node for node in graph.nodes if pos[node][0] > (components[i].centroid[0] - SLIC_SPACE)],key=lambda node: pos[node][0], reverse=True)[:int(RASM_CANDIDATE * PHI)]
                # Step 1: Get the rightmost nodes
                topmost_nodes = sorted([node for node in rightmost_nodes],key=lambda node: pos[node][1])[:int(RASM_CANDIDATE)]
                # Zulhaj @jendralhxr
                # smallest_degree_nodes = sorted([node for node in topmost_nodes], key=lambda node: graph.degree(node))[:int(RASM_CANDIDATE/PHI)]
                #node_start = max(smallest_degree_nodes, key=lambda node: pdistance(pos[node], components[i].centroid))
                #node_start = max(rightmost_nodes, key=lambda node: pos[node][1] )
				
				# @FadhilatulFitriyah
                # Step 2: Get the topmost nodes from the rightmost nodes
                # topmost_nodes = sorted(rightmost_nodes, key=lambda node: pos[node][1])[:int(RASM_CANDIDATE)]
                # Step 3: Get the node with the node start from topmost nodes
                node_start  = min(topmost_nodes, key=lambda node: graph.degree(node))
                
			# Set the node_start as the selected node
            components[i].node_start= node_start
			scribe.nodes[components[i].node_start]['color']= '#F00000' # starting node is red
			scribe_dia.nodes[components[i].node_start]['color']= '#F00000'
        
        # path finding
        #remainder_stroke= path_vane_edges(scribe, list(custom_bfs_dfs(extract_subgraph(scribe, node_start), node_start)))
        remainder_stroke= path_vane_edges(scribe, list(bfs_with_closest_priority(extract_subgraph(scribe, node_start), node_start)))
        print(remainder_stroke)
        
        # refer to rasm2hurf.py
        # rule-based minimum feasible pattern
        #rasm= stringtorasm_LEV(remainder_stroke)
        # using LSTM model
        # rasm=stringtorasm_LSTM(remainder_stroke)
        # using LCS table
        # rasm= stringtorasm_LCS(remainder_stroke)
        # rasm= stringtorasm_MC_jagokandang(remainder_stroke)
        
        ccv= cv.cvtColor(gray, cv.COLOR_GRAY2BGR)
        seed= pos[components[i].node_end]
        cv.floodFill(ccv, None, seed, (STROKEVAL,STROKEVAL,STROKEVAL), loDiff=(5), upDiff=(5))
        pil_image = Image.fromarray(cv.cvtColor(ccv, cv.COLOR_BGR2RGB))
        font = ImageFont.truetype("/usr/share/fonts/truetype/noto/NotoSansArabic-Regular.ttf", FONTSIZE)
        fontsm = ImageFont.truetype("/usr/share/fonts/truetype/noto/NotoSansArabic-Regular.ttf", FONTSIZE-12)
        drawPIL = ImageDraw.Draw(pil_image)
        #drawPIL.text((components[i].centroid[0]-FONTSIZE/2, components[i].centroid[1]-FONTSIZE), rasm, font=font, fill=(0, 200, 0))
        drawPIL.text((components[i].centroid[0]-FONTSIZE/2, components[i].centroid[1]), str(i), font=fontsm, fill=(0, 200, 200))
        # Convert back to Numpy array and switch back from RGB to BGR
        ccv= np.asarray(pil_image)
        ccv= cv.cvtColor(ccv, cv.COLOR_RGB2BGR)
        draw(ccv)
        # cv.imwrite(imagename+'highlight'+str(i).zfill(2)+'.png', ccv)
    else:
        rasm= '' # so the next rasm gets it fresh
        
    
graphfile= 'graph-'+imagename+ext
draw_graph_edgelabel(scribe_dia, 'pos_render', 8, '/shm/'+graphfile)

## ambil data dari hasil CNN
import cnn48syairperahu
import rasm2hurf

from keras.models import load_model
model = load_model('syairperahu.keras')


IMG_WIDTH= 48
IMG_HEIGHT= IMG_WIDTH

def predictfromimage(grayimage, pos):
    cx= pos[0]
    cy= pos[1]
    roi= grayimage[int(cy-IMG_WIDTH/2):int(cy+IMG_WIDTH/2), int(cx-IMG_WIDTH/2):int(cx+IMG_WIDTH/2)]
    roi = roi / THREVAL
    roi = np.expand_dims(roi, axis=0) 
    roi = np.expand_dims(roi, axis=-1)
    prediction = model.predict(roi)
    predicted_class = np.argmax(prediction, axis=-1)
    return hurf[int(predicted_class)]

for i in scribe.nodes():
    pos= scribe.nodes[i]['pos_bitmap']
    if pos[1]>IMG_HEIGHT and pos[1]<gray.shape[0]-IMG_HEIGHT and  \
       pos[0]>IMG_WIDTH and pos[0]<gray.shape[1]-IMG_WIDTH:       
       scribe.nodes[i]['hurf']= predictfromimage(gray, scribe.nodes[i]['pos_bitmap'])
       print(f"node{i}: {scribe.nodes[i]['hurf']} ada di {pos[0]}{pos[1]}")
    else:
       scribe.nodes[i]['hurf']= '' 




# #### scratchpad
# path_vane_edges(scribe, list(custom_bfs_dfs(extract_subgraph(scribe, node_start), node_start)))

# def non_returning_tspNORE(graph, start_node):
#     visited_nodes = set()  # Set to keep track of visited nodes
#     visited_edges = set()  # Set to keep track of visited edges
#     tour = []  # List to store the edges of the tour

#     def dfs(node):
#         visited_nodes.add(node)
#         neighbors = sorted(graph.neighbors(node), key=lambda neighbor: pos[neighbor][0], reverse=True)
#         for neighbor in neighbors:
#             edge = (node, neighbor) if node < neighbor else (neighbor, node)  # Normalize edge (undirected graph)
#             if edge not in visited_edges:
#                 # Mark this edge as visited
#                 visited_edges.add(edge)
#                 # Store the edge in the tour
#                 tour.append(edge)
#                 # Continue DFS traversal on the neighbor
#                 if neighbor not in visited_nodes:
#                     dfs(neighbor)

#     # Start the DFS traversal from the start node
#     dfs(start_node)

#     # Return the tour (edges traversed in the TSP-like traversal)
#     return tour

# def non_returning_tsp(graph, start_node):
#     visited_nodes = set()  # Set to keep track of visited nodes
#     tour = []  # List to store the edges of the tour

#     def dfs(node):
#         visited_nodes.add(node)
#         neighbors = sorted(graph.neighbors(node), key=lambda neighbor: pos[neighbor][0], reverse=True)
#         for neighbor in neighbors:
#             edge = (node, neighbor) if node < neighbor else (neighbor, node)  # Normalize edge (undirected graph)
            
#             # Store the edge in the tour even if it's been visited
#             tour.append(edge)
#             #print(f"Traversing edge: {edge}")
            
#             # Continue DFS traversal on the neighbor
#             if neighbor not in visited_nodes:
#                 dfs(neighbor)

#     # Start the DFS traversal from the start node
#     dfs(start_node)

#     cond_tour = []
#     for t in tour:
#         if t not in cond_tour :
#             cond_tour .append(t)
        
#     #return tour
#     return cond_tour


# # kalo keciiil
# i=2
# G=extract_subgraph(scribe, 132)
# draw_graph_edgelabel(G, 'pos_render', 2, "ntsp-sungguh3-asis.png")
# path_vane_edges(G, list(non_returning_tsp(G, components[i].node_start)))

# Gk= nx.minimum_spanning_tree(G, algorithm='kruskal')
# draw_graph_edgelabel(Gk, 'pos_render', 2, "ntsp-sungguh3-kruskal.png")
# path_vane_edges(Gk, list(non_returning_tsp(Gk, components[i].node_start)))

# Gk= nx.minimum_spanning_tree(G, algorithm='boruvka')
# draw_graph_edgelabel(Gk, 'pos_render', 2, "ntsp-sungguh3-boruvka.png")
# path_vane_edges(Gk, list(non_returning_tsp(Gk, components[i].node_start)))

# Gk= nx.minimum_spanning_tree(G, algorithm='prim')
# draw_graph_edgelabel(Gk, 'pos_render', 2, "TSPrepeat-sungguh3-prim.png")
# path_vane_edges(Gk, list(non_returning_tsp(Gk, components[i].node_start)))

# Gk= prune_edges(G, 2)
# draw_graph_edgelabel(Gk, 'pos_render', 2, "ntsp-sungguh3-hop2.png")
# path_vane_edges(Gk, list(non_returning_tsp(Gk, components[i].node_start)))

# Gk= prune_edges(G, 5)
# draw_graph_edgelabel(Gk, 'pos_render', 2, "ntsp-sungguh3-hop5.png")
# path_vane_edges(Gk, list(non_returning_tsp(Gk, components[i].node_start)))

# # kalo gede
# i=1
# G=extract_subgraph(scribe, 431)
# draw_graph_edgelabel(G, 'pos_render', 2, "sungguh3-asis.png")
# path_vane_edges(G, list(custom_bfs_dfs(G, components[i].node_start)))

# Gk= nx.minimum_spanning_tree(G, algorithm='kruskal')
# draw_graph_edgelabel(Gk, 'pos_render', 2, "sungguh3-kruskal.png")
# path_vane_edges(Gk, list(custom_bfs_dfs(Gk, components[i].node_start)))

# Gk= nx.minimum_spanning_tree(G, algorithm='boruvka')
# draw_graph_edgelabel(Gk, 'pos_render', 2, "sungguh3-boruvka.png")
# path_vane_edges(Gk, list(custom_bfs_dfs(Gk, components[i].node_start)))

# Gk= nx.minimum_spanning_tree(G, algorithm='prim')
# draw_graph_edgelabel(Gk, 'pos_render', 2, "sungguh3-prim.png")
# path_vane_edges(Gk, list(custom_bfs_dfs(Gk, components[i].node_start)))

# Gk= prune_edges(G, 2)
# draw_graph_edgelabel(Gk, 'pos_render', 2, "sungguh3-hop2.png")
# path_vane_edges(Gk, list(custom_bfs_dfs(Gk, components[i].node_start)))

# Gk= prune_edges(G, 4)
# draw_graph_edgelabel(Gk, 'pos_render', 2, "sungguh3-hop4.png")
# path_vane_edges(Gk, list(custom_bfs_dfs(Gk, components[i].node_start)))

