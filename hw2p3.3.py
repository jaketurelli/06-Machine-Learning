# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 18:23:05 2018

@author: Jake
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 21:34:13 2018

@author: Jake
"""
import numpy as np
import matplotlib.pyplot as plt
from pandas import *


def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)

def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return(x, y)


def getValues(n,thk,rad,sep):
    n_2=int(n/2)
    th_r = np.random.rand(n_2,1)*np.pi
    r_r = rad + np.random.rand(n_2,1)*thk
    [x_r,y_r] = pol2cart(r_r,th_r)
    y_r += sep/2
    t_r = -np.ones([n_2,1])
    c_r = np.tile('r',(n_2,1))
    inputs_r = np.concatenate((x_r,y_r,t_r,c_r),axis=1)

    th_b = np.random.rand(n_2,1)*-np.pi
    r_b = rad + np.random.rand(n_2,1)*thk
    [x_b,y_b] = pol2cart(r_b,th_b)
    y_b -= sep/2
    x_b += rad + thk/2
    t_b = np.ones([n_2,1])
    c_b = np.tile('b',(n_2,1))
    inputs_b = np.concatenate((x_b,y_b,t_b,c_b),axis=1)

    inputs = np.concatenate((inputs_r,inputs_b),axis=0)


#    xb = np.random.rand(n,1)-.5
#    yb = xb*slope + np.random.rand(n,1)+.05
#    tb = np.ones([n,1])
#    cb = np.tile('k',(n,1))
##    xr = (np.random.rand(n,1)*2-1)/2+0.5
##    yr = (np.random.rand(n,1)*2-1)/2-0.5
#    xr = np.random.rand(n,1)-.5
#    yr = xr*slope - np.random.rand(n,1)-.05
#    tr = -np.ones([n,1])
#    cr = np.tile('r',(n,1))
#    b = np.concatenate((xb,yb,tb,cb),axis=1)
#    r = np.concatenate((xr,yr,tr,cr),axis=1)
#    inputs = np.concatenate((b,r),axis=0)
#    myDF = DataFrame({
#    'x1' : inputs[:,0],
#    'x2' : inputs[:,1],
#    'Target' : inputs[:,2],
#    'Color': inputs[:,3]
#    })

    return(inputs)


nValues = 2000
nIter = 100000
thk = 5
rad = 10
sep = -5
myValues = getValues(nValues,thk,rad,sep)
#plt.scatter(myValues[:,0],myValues[:,1],c=myValues[:,3],s=25)
myRate = .01
slope = (sep/2/(thk+rad))

#print(myValues)

w_pocket = np.random.rand(2)
w=w_pocket
#w = [0., 0.]
epochs=0
learned = False
#fig = plt.figure()
#ax = fig.add_subplot(1,1,1)
Ein = [0.]
while not learned:
    epochs+=1
    print('epoch: ', epochs)
    totalErr = 0.0
    Ein_curr=0.0

    for row in myValues:
        target = float(row[2])
        x1 = float(row[0])
        x2 = float(row[1])
        w1 = float(w[0])
        w2 = float(w[1])
        r = w1*x1 + w2*x2
        if r >= 0:
            y=1.0
        else:
            y=-1.0

        if y != target:
            delta = target - y
#            print('w(was): ',w)
            w[0]+=delta*x1*myRate
            w[1]+=delta*x2*myRate
            totalErr += abs(delta)
            Ein_curr += y
#            print('w(is): ', w)
#            print('total Error: ', totalErr)

    Ein_curr = Ein_curr/nValues

    if epochs==1:
        Ein_min = Ein_curr
        Ein[0]=Ein_min
        w_pocket = w
    else:
        if Ein_curr < Ein_min:
            Ein_min = Ein_curr
            w_pocket = w
            n = np.linalg.norm(w)
            ww=w/n
            ww1 = 30*np.array([ww[1],-ww[0]])
            ww2 = 30*np.array([-ww[1],ww[0]])
            plt.figure()
            plt.plot([ww1[0], ww2[0]],[ww1[1], ww2[1]],'--g')
            plt.scatter(myValues[:,0], myValues[:,1], c=myValues[:,3], s=10)
            plt.show()
            print('n= ', nValues)
            print('epochs: ', epochs)
            plt.figure()
            x_curr = np.linspace(1,epochs,epochs)
            plt.plot(x_curr, Ein)
            wait = input("PRESS ENTER TO CONTINUE.")
    print(Ein)
    print(Ein_min)
    Ein=np.concatenate((Ein,[Ein_min]),0)
    w = w_pocket
    n = np.linalg.norm(w)
    ww=w/n
    ww1 = 30*np.array([ww[1],-ww[0]])
    ww2 = 30*np.array([-ww[1],ww[0]])


#    print('epoch: ', epochs)
#    print('w: ', w)

    if totalErr==0.0 or epochs>=nIter:
        learned = True

plt.figure()
plt.plot([ww1[0], ww2[0]],[ww1[1], ww2[1]],'--g')
plt.scatter(myValues[:,0], myValues[:,1], c=myValues[:,3], s=10)
plt.show()
print('n= ', nValues)
print('epochs: ', epochs)
plt.figure()
x = np.linspace(1,epochs,epochs)
plt.plot(x, Ein)

"""
(c) Repeat everything in (b) with another randomly generated data set of size 20.
Compare your results to (b)
"""


"""
(d) Repeat everything in (b) with another randomly generated data set of size 100.
Compare your results to (b)
"""

"""
(e) Repeat everything in (b) with another randomly generated data set of size 1000.
Compare your results to (b)
"""