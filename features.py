# a tribute for watatita

# freeman code going anti-clockwise like trigonometrics angle
"""
3   2   1
  \ | /
4 ------0
  / | \
5   6   7
"""
SLIC_SPACE= 6
PHI= 1.6180339887498948482 # ppl says this is a beautiful number :)


def freeman(x, y):
    if (y==0):
        y=1e-9 # so that we escape the divby0 exception
    if (x==0):
        x=-1e-9 # biased to the left as the text progresses leftward
    if (abs(x/y)<PHI) and (abs(y/x)<PHI): # corner angles
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

import os
os.chdir("/shm")
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import sys
import math

plt.figure(dpi=300)

def draw1(img): # draw the intensity
    plt.figure(dpi=300)
    plt.imshow(img)
  
def draw2(img): # draw the bitmap
    plt.figure(dpi=300)
    if (len(img.shape)==3):
        plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
    elif (len(img.shape)==2):
        plt.imshow(cv.cvtColor(img, cv.COLOR_GRAY2RGB))
        

filename= sys.argv[1]
imagename, ext= os.path.splitext(filename)
image = cv.imread(filename)
image=  cv.bitwise_not(image)
height= image.shape[0]
width= image.shape[1]

THREVAL= 80
CHANNEL= 2
#image_gray= cv.cvtColor(image, cv.COLOR_BGR2GRAY)
image_gray= image[:,:,CHANNEL]
_, gray = cv.threshold(image_gray, 0, THREVAL, cv.THRESH_OTSU) # less smear
#_, gray= cv.threshold(image_gray, 0, 1, cv.THRESH_TRIANGLE)


# # ORB
# orb = cv.ORB_create()
# keypoints, descriptors = orb.detectAndCompute(gray, None)
# image_with_keypoints = cv.drawKeypoints(gray, keypoints, None)

# def sharpen(img):
#     kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
#     sharp= cv.filter2D(img, -1, kernel)
#     return(sharp)

# sharp1= sharpen(gray)
# sharp2= sharpen(sharp1)

# # SIFT
# sift = cv.SIFT_create()
# kp0, descriptors = sift.detectAndCompute(gray, None)
# ks0 = cv.drawKeypoints(gray, keypoints, None, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
# kp1, descriptors = sift.detectAndCompute(sharp1, None)
# ks1 = cv.drawKeypoints(gray, keypoints, None, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
# kp2, descriptors = sift.detectAndCompute(sharp2, None)
# ks2 = cv.drawKeypoints(gray, keypoints, None, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

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
        if cue.item(j,i)!=0:
            moments[lbls[j,i]] = np.append(moments[lbls[j,i]], np.array([[i,j]]), axis=0)
            render.itemset((j,i,0), 120-(10*(lbls[j,i]%6)))
        else:
            moments_void[lbls[j,i]] = np.append(moments_void[lbls[j,i]], np.array([[i,j]]), axis=0)

# draw render here
# draw2(renders)
# generating nodes
scribe= nx.Graph() # start anew, just in case

# valid superpixel
filled=0
for n in range(num_slic):
    if ( len(moments[n])>SLIC_SPACE): # remove spurious superpixel with area less than 2 px 
        cx= int( np.mean(moments[n][1:,0]) ) # centroid
        cy= int( np.mean(moments[n][1:,1]) )
        if (cue.item(cy,cx)!=0):
            render.itemset((cy,cx,1), 255) 
            scribe.add_node(int(filled), label=int(lbls[cy,cx]), area=(len(moments[n])-1)/pow(SLIC_SPACE,2), pos_bitmap=(cx,cy), pos_render=(cx,-cy) )
            #print(f'point{n} at ({cx},{cy})')
            filled=filled+1



# connected components
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

components=[]

# component = ConnectedComponents(bgn_x=0, bgn_y=0, end_x=100, end_y=100, mid_x=50, mid_y=50, nodes=[2])
STROKEVAL= 200

pos = nx.get_node_attributes(scribe,'pos_bitmap')
for n in range(scribe.number_of_nodes()):
    # fill
    seed= pos[n]
    ccv= gray.copy()
    cv.floodFill(ccv, None, seed, STROKEVAL, loDiff=(5), upDiff=(5))
    _, ccv = cv.threshold(ccv, 100, STROKEVAL, cv.THRESH_BINARY)
    mu= cv.moments(ccv)
    if mu['m00'] > SLIC_SPACE*PHI:
        mc= (int(mu['m10'] / (mu['m00'])), int(mu['m01'] / (mu['m00'])))
        box= cv.boundingRect(ccv)
        #print(f'keypoint[{n}] at ({mc[0]},{mc[1]})')
        # append keypoint if the component already exists
        found=0
        for i in range(len(components)):
            if components[i].centroid==mc:
                components[i].nodes.append(n)
                found=1
                break
        if (found==0):
            components.append(ConnectedComponents(box, mc))
            components[len(components)-1].nodes.append(n)
            components[len(components)-1].mat = ccv.copy()
            components[len(components)-1].area = int(mu['m00']/THREVAL)
            

components.sort(key=lambda item:item.centroid[0], reverse=True)

disp = cv.cvtColor(gray, cv.COLOR_GRAY2BGR)
for n in range(len(components)):
    #print(f'{n} at {components[n].centroid} size {components[n].area}')
    # draw green line for rasm at edges, color the rasm brighter
    if components[n].area>4*PHI*pow(SLIC_SPACE,2):
        disp= cv.bitwise_or(disp, cv.cvtColor(components[n].mat,cv.COLOR_GRAY2BGR))
        seed= components[n].centroid
        disp.itemset((seed[1],seed[0],2), 200)
        r= components[n].rect[0]+int(components[n].rect[2])
        l= components[n].rect[0]
        if l<width and r<width:
            for j1 in range(int(SLIC_SPACE*PHI),height-int(SLIC_SPACE*PHI)):
                disp.itemset((j1,r,1), 120)
            for j1 in range(int(SLIC_SPACE*pow(PHI,3)),height-int(SLIC_SPACE*pow(PHI,3))):
                disp.itemset((j1,l,1), 120)
    else:        
        m= components[n].centroid[1]
        i= components[n].centroid[0]
        # draw blue line for shakil at mid
        for j2 in range(int(m-(2*SLIC_SPACE*PHI)), int(m+(2*SLIC_SPACE*PHI))):
            if j2<height and j2>0: 
                disp.itemset((j2,i,0), 120)

    #rasm= components[n].mat[\
    #    components[n].rect[1]:components[n].rect[1]+components[i].rect[3],\
    #    components[n].rect[0]:components[n].rect[0]+components[i].rect[2]]
    #cv.imwrite(str(n)+'.png', rasm)
#cv.imwrite(imagename+'-disp.png', disp)    
        
#-------

def draw1_graph(graph, posstring):
    positions = nx.get_node_attributes(graph,posstring)
    colors = nx.get_edge_attributes(graph,'color').values()
    plt.figure(figsize=(width/12,height/12)) 
    nx.draw(graph, 
            # nodes' param
            pos=positions,
            node_color='orange', 
            node_size=24, 
            with_labels=True, font_size=8,
            # edges' param
            edge_color=colors, 
            width=12,
            )

def draw2_graph(graph, posstring):
    # nodes
    positions = nx.get_node_attributes(graph,posstring)
    area= np.array(list(nx.get_node_attributes(graph, 'area').values()))
    # edges
    colors = nx.get_edge_attributes(graph,'color').values()
    weights = np.array(list(nx.get_edge_attributes(graph,'weight').values()))
    nx.draw(graph, 
            # nodes' param
            pos=positions, 
            with_labels=True, node_color='orange',
            node_size=area*25,
            font_size=8,
            # edges' param
            edge_color=colors, 
            width=weights*2,
            )
    
def draw3_graph(graph, posstring):
    # nodes
    if posstring is None:
        positions = nx.spring_layout(graph)
    else:
        positions = nx.get_node_attributes(graph,posstring)
    area= np.array(list(nx.get_node_attributes(graph, 'area').values()))
    edge_lbls= nx.get_edge_attributes(graph, 'vane')
    # edges
    colors = nx.get_edge_attributes(graph,'color').values()
    weights = np.array(list(nx.get_edge_attributes(graph,'weight').values()))
    nx.draw(graph, 
            # nodes' param
            pos= positions,
            with_labels=True, node_color='orange',
            node_size=200,
            font_size=8,
            # edges' param
            edge_color=colors, 
            width=weights*2,
            )
    nx.draw_networkx_edge_labels(graph, 
            pos= positions,
            edge_labels=edge_lbls, 
            font_size=8,
            font_color='red')
    
scribe.remove_edges_from(scribe.edges) # start anew, just in case
# we need to make edges between nodes within a connectedcomponent
for k in range(len(components)):
    # establish edges from the shortest distance between nodes, forward check
    # O(n^2) complexity
    for m in components[k].nodes:
        scribe.nodes[m]['component_id']=k
        src= scribe.nodes()[m]
        dist1=1e9
        dist2=1e9
        vane1=-1
        vane2=-1
        dst1= scribe.nodes()[m] # first closest
        dst2= scribe.nodes()[m] # second closest
        n1= -1
        n2= -1
        for n in components[k].nodes:
            dst= scribe.nodes()[n]
            tdist= math.sqrt( math.pow(dst['pos_bitmap'][0]-src['pos_bitmap'][0],2) + math.pow(dst['pos_bitmap'][1]-src['pos_bitmap'][1],2) )
            vane= freeman(dst['pos_bitmap'][0]-src['pos_bitmap'][0], -(dst['pos_bitmap'][1]-src['pos_bitmap'][1]))
            if (m!=n) and tdist<SLIC_SPACE*pow(PHI,2):# and midval!=0:
                if tdist<dist1 and tdist<dist2: # shortest always get it
                    dst1= scribe.nodes()[n]
                    dist1= tdist
                    n1=n
                    vane1= vane
                if tdist<dist2 and tdist>=dist1 and n!=n1 and vane!=vane1:
                    dst2= scribe.nodes()[n]
                    dist2= tdist
                    n2=n
                    vane2= vane
        #print(f'{m}: {n1}/{dist1} {n2}/{dist2}')            
        #if (dst_min!=src) and (n_min!=-1):
        # add the closest
        if n1!=-1:
            scribe.add_edge(m, n1, color='#00FF00', weight=1e1/dist1/SLIC_SPACE, vane=vane1)
        # add the second closest only if n2 is not directly connected to n1
        if scribe.has_edge(n1,n2)==False and n2!=-1:
            scribe.add_edge(m, n2, color='#00FF00', weight=1e1/dist1/SLIC_SPACE, vane=vane2)

       
# cleaning and pruning
def prune_redundant_edges(G):
    mst = nx.minimum_spanning_tree(G)
    mst_edges = set(mst.edges())
    redundant_edges = set(G.edges()) - mst_edges
    G.remove_edges_from(redundant_edges)
    return G

#s2=prune_redundant_edges(scribe)

def extract_subgraph(G, start):
    connected_component = nx.node_connected_component(G, start)
    connected_subgraph = G.subgraph(connected_component)
    return connected_subgraph.copy()
    
def edge_attributes(G):
    if isinstance(G,nx.MultiGraph) or isinstance(G,nx.MultiDiGraph):
        for u,v,k, attrs in G.edges(keys=True, data=True):
            print(f"({u}, {v}, {k}) {attrs}")
    elif isinstance(G,nx.Graph) or isinstance(G,nx.DiGraph):
        for u, v, attrs in G.edges(data=True):
            print(f"({u}, {v}) {attrs}")
    
# fix the Freeman vane since it may be revesed from double assignment
def fix_vane(G):
    if isinstance(G,nx.MultiGraph) or isinstance(G,nx.MultiDiGraph):
        new_edges = []
        for u,v,k,data in G.edges(keys=True, data=True):
            src= G.nodes()[u]
            dst= G.nodes()[v]
            # the original            
            G.edges[(u,v,k)]['vane']= freeman(dst['pos_bitmap'][0]-src['pos_bitmap'][0], -(dst['pos_bitmap'][1]-src['pos_bitmap'][1]))            
            new_edge_data = data.copy()
            new_edges.append((u, v, new_edge_data))
        for u, v, data in new_edges:
            G.add_edge(v, u, **data)
            # the parallel
            G.edges[(u,v,k+1)]['vane']= freeman(src['pos_bitmap'][0]-dst['pos_bitmap'][0], -(src['pos_bitmap'][1]-dst['pos_bitmap'][1]))            
    elif isinstance(G,nx.Graph) or isinstance(G,nx.DiGraph):
        for u,v in G.edges():
            src= G.nodes()[u]
            dst= G.nodes()[v]
            G.edges[(u,v)]['vane']= freeman(dst['pos_bitmap'][0]-src['pos_bitmap'][0], -(dst['pos_bitmap'][1]-src['pos_bitmap'][1]))            


scribe_dg= scribe.to_directed() # or we can do it at rasm level
scribe_mdg = nx.MultiDiGraph(scribe_dg)
scribe_mg= nx.MultiGraph(scribe)
    
#----------------


for k in range(2,5,1):
    resz = cv.resize(image_gray, (k*image_gray.shape[1], k*image_gray.shape[0]), interpolation=cv.INTER_LINEAR)
    _, gray = cv.threshold(resz, 0, THREVAL, cv.THRESH_OTSU)
    for i in range(8):
        for j in range(8):
            kernel = np.ones((i,j), np.uint8)  # You can adjust the kernel size as needed
            erod1 = cv.erode(gray, kernel, iterations=1)
            #draw2(erod1)
            nama=f'{k}erod{i}-{j}.png'
            cv.imwrite(nama, erod1)
 
#---------

eroded_image = cv.erode(gray, kernel, iterations=1)
cleaned_image = cv.dilate(eroded_image, kernel, iterations=1)

besar=extract_subgraph(scribe, 81)
besar_mg= nx.MultiGraph(besar)

def draw_multigraph(G, pos):
    # Draw nodes and edges
    if pos is None:
        pos = nx.spring_layout(G)
    else:
        pos = nx.get_node_attributes(G,pos)
    
    nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=500, edge_color='gray')
    
    # retrieve edge lables
    edge_labels = {}
    for u, v, key, data in G.edges(keys=True, data=True):
        edge_labels[(u, v, key)] = f"{data['vane']}"

    
    # Draw edge labels
    if edge_labels:
        for (u, v, key), label in edge_labels.items():
            # Calculate midpoint for each edge
            x = (pos[u][0] + pos[v][0]) / 2
            y = (pos[u][1] + pos[v][1]) / 2
            # Add a slight offset to the y position to avoid overlapping labels
            offset = 0.15 * key
            plt.text(x, y + offset, label, horizontalalignment='center', fontsize=8, bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'))

                       