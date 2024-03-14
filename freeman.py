# -*- coding: utf-8 -*-


# going anti-clockwise like trigonometrics angle
"""
3   2   1
  \ | /
4 ------0
  / | \
5   6   7
"""

phi= 1.6180339887498948482

def freeman(x, y):
    if (x==0):
        x=-1e-9 # biased to the left as the text progresses leftward
    if (y==0):
        y=1e-9
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
        