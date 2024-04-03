# USAGE: python -u scribe.py IMAGE_INPUT GRAPHNETWORK
# TODO: subgraphs matching
#       associating diactritics with the main character stroke
#       actually typing back the characters

import numpy as np
import cv2 as cv
import sys
import networkx as nx
import matplotlib.pyplot as plt
import math
#import os
import pickle

# freeman code going anti-clockwise like trigonometrics angle
"""
3   2   1
  \ | /
4 ------0
  / | \
5   6   7
"""
phi= 1.6180339887498948482 # ppl says this is a beautiful number :)

def freeman(x, y):
    if (y==0):
        y=1e-9 # so that we escape the divby0 exception
    if (x==0):
        x=-1e-9 # biased to the left as the text progresses leftward
    if (abs(x/y)<phi) and (abs(y/x)<phi): # corner angles
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

def delete_long_paths(graph, threshold):
    edges_to_remove = []
    # start and end nodes should be defined more automagicadynamically
    # to accomodate whitespaces just above/blow the main stroke
    for m in range(kosong):
        for n in range(m+1,kosong):
            if m != n:
                # Find all simple paths between start and end nodes
                if ((voids[n][0]-voids[m][0])<SLIC_SPACE*phi) and\
                   ((voids[n][1]-voids[m][1])>SLIC_SPACE*phi*4):
                    paths = nx.all_simple_paths(graph, m, n)
                    for path in paths:
                        if len(path) > threshold:
                            edges_to_remove.extend(zip(path[:-1], path[1:]))
        
    for u, v in edges_to_remove: # remove the edges
        if graph.has_edge(u, v):
            graph.remove_edge(u, v)
    
    isolated_nodes = list(nx.isolates(graph)) # remove the nodes
    graph.remove_nodes_from(isolated_nodes)


def graph_difference(G1, G2):
    diff_graph = nx.Graph()
    nodes_added = G1.nodes() - G2.nodes()
    for node in nodes_added:
        diff_graph.add_node(node, **G1.nodes[node])
    nodes_removed = G2.nodes() - G1.nodes()
    for node in nodes_removed:
        diff_graph.add_node(node, **G2.nodes[node])
    edges_added = G1.edges() - G2.edges()
    for edge in edges_added:
        diff_graph.add_edge(edge[0], edge[1], **G1[edge[0]][edge[1]])
    edges_removed = G2.edges() - G1.edges()
    for edge in edges_removed:
        diff_graph.add_edge(edge[0], edge[1], **G2[edge[0]][edge[1]])
    return diff_graph

def graph_sanity(target, source):
    for i in list(target.nodes()):
        if (len(target.nodes[i]['pos'])==0):
            target.nodes[i].update(source.nodes[i])

def draw_graph1(graph):
    positions = nx.get_node_attributes(graph,'pos')
    colors = nx.get_edge_attributes(graph,'color').values()
    plt.figure(figsize=(width/12,height/12)) 
    nx.draw(graph, 
            # nodes' param
            pos=positions,
            node_size=0, with_labels=True,
            font_size=8,
            # edges' param
            edge_color=colors, 
            width=1,
            )


def draw_graph2(graph):
    # nodes
    positions = nx.get_node_attributes(graph,'pos')
    area= np.array(list(nx.get_node_attributes(graph, 'area').values()))
    # edges
    colors = nx.get_edge_attributes(graph,'color').values()
    weights = np.array(list(nx.get_edge_attributes(graph,'weight').values()))
    
    plt.figure(figsize=(width/12,height/12)) 
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

def remap_node(G):
    mapping = {node: node*-1 for node in G.nodes()}
    G = nx.relabel_nodes(G, mapping)
    return G

######## main routine

filename= sys.argv[1]
image = cv.imread(filename)
height= image.shape[0]
width= image.shape[1]
image_gray= cv.cvtColor(cv.bitwise_not(image), cv.COLOR_BGR2GRAY)

ret, cue = cv.threshold(image_gray, 0, 120, cv.THRESH_OTSU) # other thresholding method may also work
render = cv.cvtColor(cue, cv.COLOR_GRAY2BGR)

# LSC and SLIC 
SLIC_SPACE= 3

# SEEDS parameters
num_superpixels = 5000
num_levels = 4
prior = 2
num_histogram_bins = 5
double_step = False

#slic = cv.ximgproc.createSuperpixelSEEDS(cue.shape[1], cue.shape[0], 1, num_superpixels, num_levels, prior, num_histogram_bins, double_step)
#slic.iterate(cue, num_iterations=4)

slic = cv.ximgproc.createSuperpixelSLIC(cue,algorithm = cv.ximgproc.SLICO, region_size = SLIC_SPACE)
#slic = cv.ximgproc.createSuperpixelSLIC(cue,algorithm = cv.ximgproc.SLIC, region_size = SLIC_SPACE)
#slic = cv.ximgproc.createSuperpixelSLIC(cue,algorithm = cv.ximgproc.MSLIC, region_size = SLIC_SPACE)
#slic = cv.ximgproc.createSuperpixelLSC(cue, region_size = SLIC_SPACE)
slic.iterate()

#mask= slic.getLabelContourMask()
num_slic = slic.getNumberOfSuperpixels()
lbls = slic.getLabels()

render = cv.cvtColor(cue, cv.COLOR_GRAY2BGR)

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

scribe= nx.Graph() # start anew, just in case
spaces= nx.Graph() # start anew, just in case
isi=0
kosong=0
voids=[]
# valid superpixel
for n in range(num_slic):
    if ( len(moments[n])>SLIC_SPACE): # remove spurious superpixel with area less than 2 px 
        #cx= int(moments[n][:,0][1]) # first elem
        #cy= int(moments[n][:,1][1])
        #render.itemset((cy,cx,0), 255) 
        #cx= int(moments[n][:,0][-1]) # last elem
        #cy= int(moments[n][:,1][-1])
        #render.itemset((cy,cx,2), 255) 
        cx= int( np.mean(moments[n][1:,0]) ) # centroid
        cy= int( np.mean(moments[n][1:,1]) )
        render.itemset((cy,cx,1), 255) 
        scribe.add_node(int(isi), label=int(lbls[cy,cx]), area=(len(moments[n])-1), pos=(cx,-cy) )
        #print(f'point{n} at ({cx},{cy})')
        isi= isi+1
    else:
        cx= int( np.mean(moments_void[n][1:,0]) ) # centroid
        cy= int( np.mean(moments_void[n][1:,1]) )
        voids.append( (cx,cy) )
        spaces.add_node(int(kosong), area=0, pos=(cx,-cy) )
        kosong= kosong+1

# defining the 'spaces'
spaces.remove_edges_from(spaces.edges) # start anew, just in case
for m in range(kosong):
    for n in range(m+1, kosong):
        if (abs(voids[n][1]-voids[m][1]) > SLIC_SPACE*pow(phi,2)):
            break
        vane= freeman(voids[n][0]-voids[m][0], voids[n][1]-voids[m][1])
        dist= math.sqrt( math.pow(voids[n][0]-voids[m][0],2) + math.pow(voids[n][1]-voids[m][1],2) )
        if (dist<SLIC_SPACE*phi) and ((vane==2) or (vane==6)):
            #print(f'jadi {m}-{n}:{vane}:{dist} ({voids[m][0]},{voids[m][1]}) to ({voids[n][0]},{voids[n][1]})')
            spaces.add_edge(m, n, color='#FF0000', weight=1)
            break

hops= int(height/SLIC_SPACE/phi) # taking into account the top and bottom blank space
spaces_old= spaces.copy()
#spaces= spaces_old.copy()
delete_long_paths(spaces, hops)
spaces_diff= graph_difference(spaces_old, spaces)
graph_sanity(spaces_diff, spaces) # anticipating leaky/lost nodes properties
spaces_diff.remove_nodes_from(list(nx.isolates(spaces_diff)))

draw_graph1(spaces_diff)

# the main stroke
temp= nx.get_node_attributes(scribe, 'pos')
cx=[]
cy=[]
for key, value in temp.items():
    cx.append(value[0])
    cy.append(-value[1])

# stroke and diacritics edges
scribe.remove_edges_from(scribe.edges) # start anew, just in case
distance = np.full(isi, 1e6)
dest = np.full(isi, -1, dtype=int)
distance_ud = np.full(isi, 1e6)
dest_ud = np.full(isi, -1, dtype=int)
bends= []

# establish edges from the shortest distance between nodes, forward check
for m in range(isi):
    # the search
    for n in range(isi):
        # find shortest distance
        if (m!=n):
            vane= freeman(cx[n]-cx[m], cy[n]-cy[m])
            tdist= math.sqrt( math.pow(cx[n]-cx[m],2) + math.pow(cy[n]-cy[m],2) )
            if (tdist<distance[m]) and (dest[m]!=n) and (dest[n]!=m):
                dest[m]= n
                distance[m]= tdist
            if (tdist<distance_ud[m]) and (dest_ud[m]!=n) and (dest_ud[n]!=m)\
                and ((vane==2) or (vane==6)):
                dest_ud[m]= n
                distance_ud[m]= tdist
    # the assignment, the nearest neighbor
    if (dest[m]!=m):
        midx= int((cx[dest[m]]+cx[m])/2)
        midy= int((cy[dest[m]]+cy[m])/2)
        kernel=0
        if (cue.item( midy  ,midx  ) !=0):
            kernel= kernel+1
        if (cue.item( midy+1,midx  ) !=0):
           kernel= kernel+1
        if (cue.item( midy+1,midx+1) !=0):
            kernel= kernel+1
        if (cue.item( midy  ,midx+1) !=0):
            kernel= kernel+1
        if (cue.item( midy-1,midx+1) !=0):
            kernel= kernel+1
        if (cue.item( midy-1,midx  ) !=0):
            kernel= kernel+1
        if (cue.item( midy-1,midx-1) !=0):
           kernel= kernel+1
        if (cue.item( midy  ,midx-1) !=0):
            kernel= kernel+1
        if (cue.item( midy+1,midx-1) !=0):
            kernel= kernel+1
        # the nearest vertical neighbor    
        midx= int((cx[dest_ud[m]]+cx[m])/2)
        midy= int((cy[dest_ud[m]]+cy[m])/2)
        kernel_ud=0
        if (cue.item( midy  ,midx  ) !=0):
            kernel_ud= kernel_ud+1
        if (cue.item( midy+1,midx  ) !=0):
           kernel_ud= kernel_ud+1
        if (cue.item( midy+1,midx+1) !=0):
            kernel_ud= kernel_ud+1
        if (cue.item( midy  ,midx+1) !=0):
            kernel_ud= kernel_ud+1
        if (cue.item( midy-1,midx+1) !=0):
            kernel_ud= kernel_ud+1
        if (cue.item( midy-1,midx  ) !=0):
            kernel_ud= kernel_ud+1
        if (cue.item( midy-1,midx-1) !=0):
           kernel_ud= kernel_ud+1
        if (cue.item( midy  ,midx-1) !=0):
            kernel_ud= kernel_ud+1
        if (cue.item( midy+1,midx-1) !=0):
            kernel_ud= kernel_ud+1    
        #print(f'{m}: {dest[m]}/{distance[m]}:{kernel} {dest_ud[m]}/{distance_ud[m]}:{kernel_ud}')        
        if (dest[m]==dest_ud[m]):
            #print(f"need more branch at {m}")
            bends.append(m)
        
        # diacritics connector
        if (dest_ud[m]!=-1):
            scribe.add_edge(m, dest_ud[m], color='#0000FF', weight=1e1/distance_ud[m]/2, code=vane, kernel=kernel_ud)
        # main stroke
        if ((kernel>pow(phi,3)) or ((kernel>pow(phi,2)) and cue.item(midy,midx))) and \
            (distance[m]<pow(phi,2)*SLIC_SPACE) and (dest[m]!=-1):
            scribe.add_edge(m, dest[m], color='#00FF00', weight=1e1/distance[m], code=vane, kernel=kernel)
        if ((kernel_ud>pow(phi,3)) or ((kernel_ud>pow(phi,2)) and cue.item(midy,midx))) and \
            (distance_ud[m]<pow(phi,2)*SLIC_SPACE) and (dest_ud[m]!=-1):
            scribe.add_edge(m, dest_ud[m], color='#00FF00', weight=1e1/distance[m], code=vane, kernel=kernel)
        
# additional edges missing from the O(n^2) search
for m in bends:
    bdist= 1e6
    bdest= -1
    for n in range(isi):
        if (m!=n) and (n!=dest[m]) and (m!=dest[n]):
            vane= freeman(cx[n]-cx[m], cy[n]-cy[m])
            tdist= math.sqrt( math.pow(cx[n]-cx[m],2) + math.pow(cy[n]-cy[m],2) )
            midx= int((cx[m]+cx[n])/2)
            midy= int((cy[m]+cy[n])/2)
            kernel=0
            if (cue.item( midy  ,midx  ) !=0):
                kernel= kernel+1
            if (cue.item( midy+1,midx  ) !=0):
               kernel= kernel+1
            if (cue.item( midy+1,midx+1) !=0):
                kernel= kernel+1
            if (cue.item( midy  ,midx+1) !=0):
                kernel= kernel+1
            if (cue.item( midy-1,midx+1) !=0):
                kernel= kernel+1
            if (cue.item( midy-1,midx  ) !=0):
                kernel= kernel+1
            if (cue.item( midy-1,midx-1) !=0):
               kernel= kernel+1
            if (cue.item( midy  ,midx-1) !=0):
                kernel= kernel+1
            if (cue.item( midy+1,midx-1) !=0):
                kernel= kernel+1
            if (tdist<bdist):
                bdist= tdist
                bdest= n
    if (bdest!=-1) and (bdist<pow(phi,2)*SLIC_SPACE) and\
        ((kernel>pow(phi,3)) or ((kernel>pow(phi,2)) and cue.item(midy,midx))):
        scribe.add_edge(m, bdest, color='#00FF00', weight=1e1/bdist, code=vane, kernel=kernel)
        #print(f"{m}-{bdest} {bdist} {vane} {kernel}")    
#scribe.number_of_edges()            

spaces_diff= remap_node(spaces_diff)
merged = nx.compose(scribe,spaces_diff)
merged.remove_node(0)
draw_graph2(merged)

draw_graph2(scribe)
plt.savefig("test.png", bbox_inches='tight')
#plt.savefig(sys.argv[3], bbox_inches='tight')

# save graph object to file
pickle.dump(scribe, open(sys.argv[4], 'wb'))
#scribe = pickle.load(open(sys.argv[4], 'rb'))

"""
# the artistic rendition
overlay= cv.imread(sys.argv[3])
overlay= cv.cvtColor(overlay, cv.COLOR_RGB2BGR)
canvas= cv.bitwise_not(image)
canvas = cv.resize(canvas, (overlay.shape[1], overlay.shape[0]))
blend  = cv.addWeighted(canvas, 0.5, overlay, 0.5, 0)
plt.imshow(blend) 
"""

render= cv.cvtColor(render, cv.COLOR_BGR2RGB)
plt.imshow(render) 
#plt.axis("off")
cv.imwrite(sys.argv[2], render)
