#!/opt/spyder/envs/my-env/bin/python

# freeman code going anti-clockwise like trigonometrics angle
#    3   2   1
#      \ | /
#    4 ------0
#      / | \
#    5   6   7

SLIC_SPACE= 3
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
#os.chdir("/shm")
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import sys
import math
import heapq

plt.figure(dpi=300)
  
def draw(img): # draw the bitmap
    plt.figure(dpi=300)
    if (len(img.shape)==3):
        plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
    elif (len(img.shape)==2):
        plt.imshow(cv.cvtColor(img, cv.COLOR_GRAY2RGB))
        
k=2
SLIC_SPACE= SLIC_SPACE*k

filename= sys.argv[1]
#filename= 'topanribut.png'
imagename, ext= os.path.splitext(filename)
image = cv.imread(filename)
resz = cv.resize(image, (k*image.shape[1], k*image.shape[0]), interpolation=cv.INTER_LINEAR)
image= resz.copy()
image=  cv.bitwise_not(image)
height= image.shape[0]
width= image.shape[1]

THREVAL= 80
CHANNEL= 2
#image_gray= cv.cvtColor(image, cv.COLOR_BGR2GRAY)
image_gray= image[:,:,CHANNEL]
_, gray = cv.threshold(image_gray, 0, THREVAL, cv.THRESH_OTSU) # less smear
#_, gray= cv.threshold(image_gray, 0, 1, cv.THRESH_TRIANGLE)

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

# // image [pre-]processing ends here

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
            scribe.add_node(int(filled), label=int(lbls[cy,cx]), area=(len(moments[n])-1)/pow(SLIC_SPACE,2), pos_bitmap=(cx,cy), pos_render=(cx,-cy), color='orange')
            #print(f'point{n} at ({cx},{cy})')
            filled=filled+1

def pdistance(point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return distance

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
    node_start: Optional[int] = field(default=-1)    # right-up
    distance_start: Optional[int] = field(default=0) # right-up
    node_end: Optional[int] = field(default=-1)      # left-down
    distance_end: Optional[int] = field(default=0)   # left-down

STROKEVAL= 200

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
        # r= components[n].rect[0]+int(components[n].rect[2])
        # l= components[n].rect[0]
        # if l<width and r<width: # did we ever went beyond the frame?
        #     for j1 in range(int(SLIC_SPACE*PHI),height-int(SLIC_SPACE*PHI)):
        #         disp[j1,r,1]= 120
        #     for j1 in range(int(SLIC_SPACE*pow(PHI,3)),height-int(SLIC_SPACE*pow(PHI,3))):
        #         disp[j1,l,1]= 120
    else:        
        m= components[n].centroid[1]
        i= components[n].centroid[0]
        # draw blue line for shakil 'connection'
        for j2 in range(int(m-(2*SLIC_SPACE*PHI)), int(m+(2*SLIC_SPACE*PHI))):
            if j2<height and j2>0: 
                disp[j2,i,1]= STROKEVAL/2

    #rasm= components[n].mat[\
    #    components[n].rect[1]:components[n].rect[1]+components[i].rect[3],\
    #    components[n].rect[0]:components[n].rect[0]+components[i].rect[2]]
    #cv.imwrite(str(n)+'.png', rasm)
#cv.imwrite(imagename+'-disp.png', disp)    

# draw(disp) 

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
    plt.figure(figsize=(5*scale,5)) 
    # nodes
    if posstring is None:
        positions = nx.spring_layout(graph)
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
        plt.savefig(filename)
    
scribe.remove_edges_from(scribe.edges) # start anew, just in case
# we need to make edges between nodes within a connectedcomponent
for k in range(len(components)):
    # establish edges from the shortest distance between nodes, forward check
    # O(n^2) complexity
    for m in components[k].nodes:
        scribe.nodes[m]['component_id']=k
        src= scribe.nodes()[m]
        ndist=[1e9, 1e9, 1e9]
        ndst= [-1, -1, -1]
        for n in components[k].nodes:
            dst= scribe.nodes()[n]
            cdist= math.sqrt( math.pow(dst['pos_bitmap'][0]-src['pos_bitmap'][0],2) + math.pow(dst['pos_bitmap'][1]-src['pos_bitmap'][1],2) )
            if (m!=n) and cdist<SLIC_SPACE*PHI:
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
        for i in range(3):
            if ndist[i]!=1e9 and ndst[i]!=-1:
                dst= scribe.nodes()[ndst[i]]
                #print(f'{m} to {n}: {ndist[i]}')            
                tvane= freeman(dst['pos_bitmap'][0]-src['pos_bitmap'][0], -(dst['pos_bitmap'][1]-src['pos_bitmap'][1]))
                # // (i==1 and scribe.has_edge(m,ndst[2])==True ) # for 1+alpha
                if (i==2) or \
                   (i==1 and scribe.has_edge(ndst[2],ndst[1])==False ) or \
                   (i==0 and scribe.has_edge(ndst[2],ndst[0])==False and scribe.has_edge(ndst[1],ndst[0])==False):
                    scribe.add_edge(m, ndst[i], color='#00FF00', weight=1e2/ndist[i]/SLIC_SPACE, vane=tvane) 

degree_rasm= scribe.degree()

scribe_dia= scribe.copy()

RASM_EDGE_MAXDEG= 3
# finding diacritics connection for small components
# and update extreme nodes for large components
for k in range(len(components)):
    if components[k].area<pow(SLIC_SPACE,2)*pow(PHI,3) or len(components[k].nodes)<=4:
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
                        if tdist<closest_dist and (tvane==2 or tvane==6):
                            closest_comp= l
                            src_node= m
                            closest_node= n
                            closest_vane= tvane
                            closest_dist= tdist
        #print(f'comp {k} to {closest_comp} \t node {m} to {n}\t: {closest_dist} {closest_vane}')            
        if closest_dist<SLIC_SPACE*pow(PHI,4):
            scribe_dia.add_edge(src_node, closest_node, color='#0000FF', weight=1e2/closest_dist/SLIC_SPACE, vane=closest_vane)
            if closest_vane==6: # diacritics over
                scribe.nodes[closest_node]['color']='#0080FF'
                scribe_dia.nodes[closest_node]['color']='#0080FF'
            else: # diacritics below
                scribe.nodes[closest_node]['color']='#8000FF'
                scribe_dia.nodes[closest_node]['color']='#8000FF'
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
                    scribe.nodes[components[k].node_start]['color']= 'red'
                    scribe_dia.nodes[components[k].node_start]['color']= 'red'
                    flag= True
                    break
            if flag:
                break
        
# degree_dia= scribe.degree()

# draw_graph(scribe_dia, 'pos_render', 8)

def extract_subgraph(G, start):
    if start!=-1:
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
    
def path_vane_nodes(G, path): # if path is written as series of nodes
    pathstring=''
    for i in range(len(path) - 1):
        src= G.nodes()[path[i]]
        dst= G.nodes()[path[i+1]]
        tvane= freeman(dst['pos_bitmap'][0]-src['pos_bitmap'][0], -(dst['pos_bitmap'][1]-src['pos_bitmap'][1]))
        pathstring+=str(tvane)
    return pathstring

def path_vane_edges(G, path): # if path is written is written as series of edges
    pathstring=''
    for n in path:
        src= G.nodes()[n[0]]
        dst= G.nodes()[n[1]]
        tvane= freeman(dst['pos_bitmap'][0]-src['pos_bitmap'][0], -(dst['pos_bitmap'][1]-src['pos_bitmap'][1]))
        G.edges[n]['vane']=tvane
        if (G.edges[n]['color']=='#00FF00'): # main stroke
            pathstring+=str(tvane)
        # else: #substroke
        #     if tvane==2:
        #         pathstring+='+'
        #     else:
        #         pathstring+='-'
        if dst['color']=='#8000FF': # diacritics below
            pathstring+='-'
        if dst['color']=='#0080FF': # diacritics over
            pathstring+='+'
    return pathstring

#list(nx.bfs_edges(besar, source=29)) # simplifiend
#list(nx.edge_bfs(besar, source=29)) # traverse sequence
#path_vane_edges(scribe, list(nx.edge_bfs(extract_subgraph(scribe, 29), source=29)))

for i in range(len(components)):
    if len(components[i].nodes)>3:
        if components[i].node_start==-1:
            node_pos=(0,0)
            node_start=-1
            for n in components[i].nodes:
                if pos[n][0] > node_pos[0]:
                    node_start= n
                    node_pos= pos[n]
            #node_start= components[i].nodes[0]
        else:
            node_start= components[i].node_start
        print(path_vane_edges(scribe, list(nx.edge_bfs(extract_subgraph(scribe, node_start), source=node_start))))

graphfile= imagename+'-graph'+ext
draw_graph_edgelabel(scribe_dia, 'pos_render', 8, graphfile)
