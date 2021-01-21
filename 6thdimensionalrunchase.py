# -*- coding: utf-8 -*-
"""
Created on Thu Nov  5 10:25:01 2020


@author: Nicolas Hudin

Program which models the semi-random movement of nth dimensional points in nth dimensional toroidal space.

Points can have attractive or repulsive behaviors relative to one another.

Closer points have stronger attraction, points further have weaker attraction.

When points exceed a certain threshold, the value wraps around, simulating toroidal space. 

"Animates" by showing points and coordinates in progressive graphs, with steps determined by input.

User defined parameters:
  Attraction behavior and strength can be defined by user.
  Random seed can be defined by user.
  Size of coordinate space can be defined by user.
  Average step size of points can be defined by user.\
  Steps per cycle can be defined by user.
  Steps per 'frame' of animation must be defined by user.

"""
import random
import math
import itertools
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
import csv
import pandas as pd

#Moving points
p0=[-1,2,-3,4,-5,1]
p1=[5,-4,3,-2,1,0]
p2=[5,-7,-2,3,1,3]
p3=[-9,8,7,-6,7,4]


#Seed for random generated values
seed_number=3


random.seed(seed_number)

# pointsdic={
#     0: p1,
#     1: p2
#     }

# itertools.permutations(range(len(pointsdic)))

pointlist=(p1,p2)
#Total range of coordinates
coordrange=20
axisrange=coordrange/2
#Average size of a step of movement of a point
stepsize=1
#number of steps calculated
step_number=50
dims=len(p0)
#number of dimensions per point
dims=6
point_number=4
maxl=(dims*((coordrange/2)**2))**.5

def periodicshift(value,maxminrange=coordrange,inv=False):
    value+=((maxminrange/2)*bool(inv)) #puts value on other end of circular range if >0
    shiftvalue=maxminrange/2
    shiftedvalue=((value+shiftvalue)%maxminrange)-shiftvalue
    return shiftedvalue

def pointdiflist(coordinatelist1,coordinatelist2,dimensions=dims):#what coordlist1 needs to move to reach coordlist2
    deltalist=[]
    for i in range(dimensions):
        icoord=(coordinatelist2[i])-(coordinatelist1[i])
        deltalist.append(icoord)
    return deltalist

def pointshiftlist(coordinatelist1,shiftinglist,dimensions=dims):#pos of coordlist1 after being shifted by shiftedlist
    deltalist=[]
    for i in range(dimensions):
        icoord=coordinatelist1[i]+shiftinglist[i]
        deltalist.append(icoord)
    return deltalist

def angularpointshiftlist(coordinatelist1,shiftinglist,anglelist,dimensions=dims):#pos of coordlist1 after being shifted by shiftedlist
    deltalist=[]
    for i in range(dimensions):
        icoord=coordinatelist1[i]+(shiftinglist[i]*math.sin(anglelist[i]))
        deltalist.append(icoord)
    return deltalist

def pointmeanlist(coordinatelist1,shiftinglist,dimensions=dims): #finds means of 2 point coordinate lists
    deltalist=[]
    for i in range(dimensions):
        icoord=coordinatelist1[i]+shiftinglist[i]
        deltalist.append((icoord*.5))
    return deltalist

def pointmeanlist2(list0,list1,list2): #finds means of 2 point coordinate lists
    deltalist=[]
    for i in range(dims):
        icoord=list0[i]+list1[i]+list2[i]
        deltalist.append((icoord*(1/(point_number-1))))
    return deltalist

def projectedim(deltalist,dimensions=dims):#deltalist=point difference list
    #projecteddimension=[]
    projectedcoordinates=[]
    for i in range(dimensions):
        tempchecklist = [(deltalist[i]-coordrange),(deltalist[i]),(deltalist[i]+coordrange)]
        tempmin=min(tempchecklist,key=abs)
        #projecteddimension.append(tempchecklist.index(tempmin))
        projectedcoordinates.append(tempmin)
    return projectedcoordinates#,projecteddimension


def listperiodicshift(coordlist,maxminrange=coordrange,inv=False,dimensions=dims):
    for i in range(dimensions):
        coordlist[i]=periodicshift(coordlist[i],maxminrange,inv)

def ndist(coordinatelist1,coordinatelist2,dimensions=dims):
    pointdifferencesum=0
    for i in range(dimensions):
        pointdifferencesum+= (coordinatelist1[i]-coordinatelist2[i])**2
    return (pointdifferencesum**.5)

def ndist2(deltalist,dimensions=dims):
    pointdifferencesum=0
    for i in range(dimensions):
        pointdifferencesum+= (deltalist[i])**2
    return (pointdifferencesum**.5)

def randompointgen(coordinaterange=coordrange,dimensions=dims):
    coordinatelist=[]
    for i in range(dimensions):
        coordinatelist.append(coordinaterange*random.random()-coordinaterange*.5)
    return coordinatelist

def wraparound(coordinatelist,dimensions=dims):
    for i in range(dimensions):
        coordinatelist[i]=periodicshift(coordinatelist[i],coordrange)
    return coordinatelist

def anglesget(diflist,dimensions=dims):#generates angles based on a difference list
    fanglelist=[]
    for i in range(dimensions):
        difference1=diflist[i]
        difference2=diflist[(i+1)%dimensions]
        #if dimension number is odd, final angle is drawn from last axis value and first.
        fanglelist.append(math.atan2(difference1,difference2))
    return fanglelist

def rananglesgen():
    angleslist=randompointgen()
    return(anglesget(angleslist))

def attrstr(trueangles,referenceangles,dimensions=dims,step=stepsize,staticmod=1,modifier=1):
    anglesdiflist=pointdiflist(trueangles,referenceangles)
    attrlist=[]
    for i in range(dimensions):
        cleanangle=periodicshift(anglesdiflist[i],math.tau) #finds proper positions of angles on a circle
        attractionvalue= step+(modifier*((step*(abs(abs(cleanangle)-math.pi)/math.pi))-step))
        attrlist.append(staticmod*attractionvalue)
    return attrlist

def s_distance(distance):
    rdis=distance/maxl
    return 1-((1/(1+math.exp(-(math.exp(2))*rdis)))**math.exp(4))#arbitrary values

def attrstr2(trueangles,referenceangles,dimensions=dims,step=stepsize,staticmod=1,modifier=1):
    anglesdiflist=pointdiflist(trueangles,referenceangles)
    attrlist=[]
    for i in range(dimensions):
        cleanangle=periodicshift(anglesdiflist[i],math.tau) #finds proper positions of angles on a circle
        attractionvalue= staticmod*(abs(math.pi-abs(cleanangle*modifier)))/math.pi
        attrlist.append(attractionvalue)
    return attrlist

def modifierstrength2(distance):
    maxl=(dims*((coordrange/2)**2))**.5
    m=(math.log(maxl+1)-math.log(distance+1))+1
    return m

def modifierstrength(distance):
    m=(math.log(maxl*1.5)-math.log(distance+(maxl/2)))+.5
    #print("distance",distance)
    return m


    
    
def attrregion(trueangles,referenceangles,distance,dimensions=dims,step=stepsize,staticmod=1,):
    anglesdiflist=pointdiflist(trueangles,referenceangles)
    attrlist=[]
    sdis=(s_distance(distance))*4 #arbitrary
    sdis1=sdis+1
    
    
    for i in range(dimensions):
        invangle=periodicshift(anglesdiflist[i],math.tau) #finds proper positions of angles on a circle
        attractionvalue= step+((step*(abs(abs(invangle)-math.pi)/math.pi))-step)
        attrlist.append(staticmod*attractionvalue)
    return attrlist

#step+((attrx2-1))
#forgot about sin
    

def movement(movepointlist,refpointlist1,staticmod1,refpointlist2,staticmod2,refpointlist3,staticmod3):#args??
    
    def mattr(reflist, mod):
    
        shortestprojection1=projectedim(pointdiflist(movepointlist,reflist))
    
        d1=ndist2(shortestprojection1)
    
        sdis=s_distance(d1)
    
        projectionangles1=anglesget(shortestprojection1)
    
        return attrstr2(moveangles,projectionangles1,staticmod=mod,modifier=sdis)
    
    moveangles=rananglesgen()

    moveattr1=mattr(refpointlist1,staticmod1)

    moveattr2=mattr(refpointlist2,staticmod2)
    
    moveattr3=mattr(refpointlist3,staticmod3)
    
    moveattrsum=pointmeanlist2(moveattr1,moveattr2,moveattr3)
    
    ppos = angularpointshiftlist(movepointlist,moveattrsum,moveangles)
    
#    move = pointdiflist(movepointlist,ppos)
    
    #print(move)
    
    #return wraparound(pointmeanlist(ppos1,ppos2))
    return wraparound(ppos)
    

plist=[]

ilist=[]


#makes 3 random coordinates:
p0=randompointgen()
p1=randompointgen()
p2=randompointgen()
p3=randompointgen()

for i in range(step_number):
    
    timelist=[p0,p1,p2,p3]
    
    plist.extend(timelist)
    
    rp0=p0.copy()
    rp1=p1.copy()
    rp2=p2.copy()
    rp3=p3.copy()
    
    #these values determine the attraction or repulsion behaviors of the points. Negative values are repulsed. Values further from 0 have stronger behavior.
    p0=movement(rp0,rp1,1,rp2,-1,rp3,-1)
    p1=movement(rp1,rp0,-1,rp2,.5,rp3,-1)
    p2=movement(rp2,rp0,1,rp1,-.5,rp3,-1)
    p3=movement(rp3,rp0,1,rp1,1,rp2,1)

pca=PCA(n_components=6)
plist=pca.fit_transform(plist)
rplist=plist.copy()

parr=np.array(plist)
parr=parr.reshape(step_number,point_number,dims)#middle value is number of points
print(parr)


"""

graph_number=(dims//3)

if dims%3 > 0:
    graph_number+=1

fig = plt.figure(figsize=plt.figaspect(1/graph_number))

ax0 = fig.add_subplot(1,2,1, projection="3d")

ax0.scatter((*zip(*parr[0:step_number,0,0:3])), c='r', marker='o',label="p0")
ax0.scatter((*zip(*parr[0:step_number,1,0:3])), c='b', marker='o',label="p1")
ax0.scatter((*zip(*parr[0:step_number,2,0:3])), c='g', marker='o',label="p2")
ax0.scatter((*zip(*parr[0:step_number,3,0:3])), c='y', marker='o',label="p3")

ax0.set_xlabel('dimension 1')
ax0.set_ylabel('dimension 2')
ax0.set_zlabel('dimension 3')
ax0.legend()

ax1 = fig.add_subplot(1,2,2, projection='3d')

ax1.scatter((*zip(*parr[0:step_number,0,3:6])), c='r', marker='o',label="p0")
ax1.scatter((*zip(*parr[0:step_number,1,3:6])), c='b', marker='o',label="p1")
ax1.scatter((*zip(*parr[0:step_number,2,3:6])), c='g', marker='o',label="p2")
ax1.scatter((*zip(*parr[0:step_number,3,3:6])), c='y', marker='o',label="p3")

ax1.set_xlabel('dimension 4')
ax1.set_ylabel('dimension 5')
ax1.set_zlabel('dimension 6')
ax1.legend()

"""

#Displays movement of points over time in 'animated' sense
fn=int(input("steps per frame"))
for i in range(step_number//fn):
    input("next frame")
        
    graph_number=(dims//3)
    
    if dims%3 > 0:
        graph_number+=1
    
    fig = plt.figure(figsize=plt.figaspect(1/graph_number))
    
    ax0 = fig.add_subplot(1,2,1, projection='3d')
    
    ax0.scatter((*zip(*parr[0:((i+1)*fn),0,0:3])), c='r', marker='o',label="p0")
    ax0.scatter((*zip(*parr[0:((i+1)*fn),1,0:3])), c='b', marker='o',label="p1")
    ax0.scatter((*zip(*parr[0:((i+1)*fn),2,0:3])), c='g', marker='o',label="p2")
    ax0.scatter((*zip(*parr[0:((i+1)*fn),3,0:3])), c='y', marker='o',label="p3")
    
    ax0.set_xlabel('dimension 1')
    ax0.set_ylabel('dimension 2')
    ax0.set_zlabel('dimension 3')
    ax0.legend()
    
    ax1 = fig.add_subplot(1,2,2, projection='3d')
    
    ax1.scatter((*zip(*parr[0:((i+1)*fn),0,3:6])), c='r', marker='o',label="p0")
    ax1.scatter((*zip(*parr[0:((i+1)*fn),1,3:6])), c='b', marker='o',label="p1")
    ax1.scatter((*zip(*parr[0:((i+1)*fn),2,3:6])), c='g', marker='o',label="p2")
    ax1.scatter((*zip(*parr[0:((i+1)*fn),3,3:6])), c='y', marker='o',label="p3")
    
    ax1.set_xlabel('dimension 4')
    ax1.set_ylabel('dimension 5')
    ax1.set_zlabel('dimension 6')
    ax1.legend()
    
    plt.show()