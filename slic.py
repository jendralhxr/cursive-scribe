# USAGE: python -u scribe.py IMAGE_INPUT 

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
        

######## main routine

filename= sys.argv[1]
image = cv.imread(filename)
height= image.shape[0]
width= image.shape[1]
image_gray= cv.cvtColor(cv.bitwise_not(image), cv.COLOR_BGR2GRAY)

ret, cue = cv.threshold(image_gray, 0, 120, cv.THRESH_OTSU) # other thresholding method may also work
render = cv.cvtColor(cue, cv.COLOR_GRAY2BGR)

# LSC and SLIC 
space= 3
key= 32;

# SEEDS parameters
num_superpixels = 5000
num_levels = 4
prior = 2
num_histogram_bins = 5
double_step = False

#slic = cv.ximgproc.createSuperpixelSEEDS(cue.shape[1], cue.shape[0], 1, num_superpixels, num_levels, prior, num_histogram_bins, double_step)
#slic.iterate(cue, num_iterations=4)

slic = cv.ximgproc.createSuperpixelSLIC(cue,algorithm = cv.ximgproc.SLICO, region_size = space)
#slic = cv.ximgproc.createSuperpixelSLIC(cue,algorithm = cv.ximgproc.SLIC, region_size = space)
#slic = cv.ximgproc.createSuperpixelSLIC(cue,algorithm = cv.ximgproc.MSLIC, region_size = space)
#slic = cv.ximgproc.createSuperpixelLSC(cue, region_size = space)
slic.iterate()

#mask= slic.getLabelContourMask()
num_slic = slic.getNumberOfSuperpixels()
lbls = slic.getLabels()

render = cv.cvtColor(cue, cv.COLOR_GRAY2BGR)
moments = [np.zeros((1, 2)) for _ in range(num_slic)]
# tabulating the superpixel labels
for j in range(height):
    for i in range(width):
        if cue.item(j,i)!=0:
            moments[lbls[j,i]] = np.append(moments[lbls[j,i]], np.array([[i,j]]), axis=0)
            render.itemset((j,i,0), 120-(10*(lbls[j,i]%6)))
            

scribe= nx.Graph() # start anew, just in case
isi=0
# valid superpixel
for n in range(num_slic):
    if ( len(moments[n])>3): # remove spurious superpixel with area less than 2 px 
        #cx= int(moments[n][:,0][1])
        #cy= int(moments[n][:,1][1])
        #render.itemset((cy,cx,0), 255) # first elem
        #cx= int(moments[n][:,0][-1])
        #cy= int(moments[n][:,1][-1])
        #render.itemset((cy,cx,1), 255) # centroid
        cx= int( np.mean(moments[n][1:,0]) )
        cy= int( np.mean(moments[n][1:,1]) )
        render.itemset((cy,cx,2), 255) # last elem
        scribe.add_node(int(isi), label=int(lbls[cy,cx]), area=(len(moments[n])-1), pos=(cx,-cy) )
        #print(f'point{n} at ({cx},{cy})')
        isi= isi+1

# establish edges from the shortest distance between nodes
scribe.remove_edges_from(scribe.edges) # start anew, just in case
temp= nx.get_node_attributes(scribe, 'pos')
cx=[]
cy=[]
for key, value in temp.items():
    cx.append(value[0])
    cy.append(-value[1])
for m in range(isi):
    distance= 1e3 # distance to closest neighboring keypoints
    distance_ud= 1e3 # distance to closer vertical neighbor 
    orig= 0
    dest= 0
    orig_ud= 0
    dest_ud= 0
    # the search
    for n in range(m+1, isi):
        # find shortest distance
        vane= freeman(cx[m]-cx[n], cy[m]-cy[n])
        tdist= math.sqrt( math.pow(cx[m]-cx[n],2) + math.pow(cy[m]-cy[n],2) )
        if (tdist<distance):
            orig= m
            dest= n
            distance= tdist
        if (tdist<distance_ud) and ((vane==2) or (vane==6)):
            orig_ud= m
            dest_ud= n
            distance_ud= tdist
            #print(f'edge between {orig} and {dest}')
    # the assignment
    # establish the edges if not already exist between closest a pair of nodes
    if (scribe.has_edge(orig,dest)==False) and (scribe.has_edge(dest,orig)==False) and (orig!=dest): # 1 px smear in the main stroke 
        if     (cue.item( int((cy[dest]+cy[orig])/2)    ,int((cx[dest]+cx[orig])/2))    !=0) \
            or (cue.item( int((cy[dest]+cy[orig])/2+1)  ,int((cx[dest]+cx[orig])/2))    !=0) \
            or (cue.item( int((cy[dest]+cy[orig])/2+1)  ,int((cx[dest]+cx[orig])/2+1))  !=0) \
            or (cue.item( int((cy[dest]+cy[orig])/2)    ,int((cx[dest]+cx[orig])/2+1))  !=0) \
            or (cue.item( int((cy[dest]+cy[orig])/2-1)  ,int((cx[dest]+cx[orig])/2+1))  !=0) \
            or (cue.item( int((cy[dest]+cy[orig])/2-1)  ,int((cx[dest]+cx[orig])/2))    !=0) \
            or (cue.item( int((cy[dest]+cy[orig])/2-1)  ,int((cx[dest]+cx[orig])/2-1))  !=0) \
            or (cue.item( int((cy[dest]+cy[orig])/2)    ,int((cx[dest]+cx[orig])/2-1))  !=0) \
            or (cue.item( int((cy[dest]+cy[orig])/2+1)  ,int((cx[dest]+cx[orig])/2-1))  !=0):
            fill='#00FF00' # green in-stroke, RGB
            scribe.add_edge(orig, dest, color=fill, weight=1e1/distance, code=vane)
            #print(f'stroke between {orig} and {dest}')
        else:
            # jumps over the void
            # additional check for 2 or 6 freeman code (straight up- and downward)
            fill='#0000FF' # blue void: article, harakat, or tashkeel RGB
            scribe.add_edge(orig_ud, dest_ud, color=fill, weight=1e1/distance_ud, code=vane)
            #print(f'tashkeel between {orig} and {dest}')
            
        
        
# re-fetch the attributes from drawing
# nodes
positions = nx.get_node_attributes(scribe,'pos')
area= np.array(list(nx.get_node_attributes(scribe, 'area').values()))
# edges
colors = nx.get_edge_attributes(scribe,'color').values()
weights = np.array(list(nx.get_edge_attributes(scribe,'weight').values()))

nx.draw(scribe, 
        # nodes' param
        pos=positions, 
        with_labels=True, node_color='orange',
        node_size=area*25,
        # edges' param
        edge_color=colors, 
        width=weights*4,
        font_size=8
        )

# save graph object to file
pickle.dump(scribe, open(sys.argv[3], 'wb'))
#scribe = pickle.load(open(sys.argv[3], 'rb'))

render= cv.cvtColor(render, cv.COLOR_BGR2RGB)
plt.imshow(render) 


#print(isi)
    
#render = cv.cvtColor(cue, cv.COLOR_GRAY2BGR)
#mask2 = cv.cvtColor(mask, cv.COLOR_GRAY2BGR)
#render= cv.bitwise_or(render, mask2)
cv.imwrite(sys.argv[2], render)
print(f'save to: {sys.argv[2]}')
#cv.imshow("mask", mask)
#cv.imshow("show", render)
#key = cv.waitKey(0) & 0xff

