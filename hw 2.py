# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 21:34:13 2018

@author: Jake
"""
import numpy as np
import matplotlib.pyplot as plt
from pandas import *
"""
Help:
http://stamfordresearch.com/scikit-learn-perceptron/
http://stamfordresearch.com/python-perceptron-re-visited/
https://www.youtube.com/watch?v=4J1ccdYRhmc
https://maviccprp.github.io/a-perceptron-in-just-a-few-lines-of-python-code/
https://glowingpython.blogspot.com/2011/10/perceptron.html
"""

"""
Problem 1.4

"""

"""
(a) Generate a linearly separable data set of size 20 as indicated in Exercise 1.4.
Plot the examples {(xn,yn)} as well as the target function f on a plane.
Be sure to mark the examples from different classes differently,
and add labels to the axes of the plot.
"""
def getValues(n,slope):
#    xb = (np.random.rand(n,1)*2-1)/2-0.5
#    yb = (np.random.rand(n,1)*2-1)/2+0.5
    xb = np.random.rand(n,1)-.5
    yb = xb*slope + np.random.rand(n,1)+.05
    tb = np.ones([n,1])
    cb = np.tile('k',(n,1))
#    xr = (np.random.rand(n,1)*2-1)/2+0.5
#    yr = (np.random.rand(n,1)*2-1)/2-0.5
    xr = np.random.rand(n,1)-.5
    yr = xr*slope - np.random.rand(n,1)-.05
    tr = -np.ones([n,1])
    cr = np.tile('r',(n,1))
    b = np.concatenate((xb,yb,tb,cb),axis=1)
    r = np.concatenate((xr,yr,tr,cr),axis=1)
    inputs = np.concatenate((b,r),axis=0)
    myDF = DataFrame({
    'x1' : inputs[:,0],
    'x2' : inputs[:,1],
    'Target' : inputs[:,2],
    'Color': inputs[:,3]
    })

    return(inputs)

nValues = 1000
slope = 0.33
myValues = getValues(nValues,slope)
myRate = .01
"""
(b) Run the perceptron learning algorithm on the data set above.
Report the number of updates that the algorithm takes before convergin.
Plot the examples {(xn,yn)}, the target function f, and the final hypothesis
g in the same figure.
Comment on whether f is close to g.
"""
w = np.random.rand(2)
epochs=0
learned = False
#fig = plt.figure()
#ax = fig.add_subplot(1,1,1)

while not learned:
    totalErr = 0.0
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
#            print('w(is): ', w)
#            print('total Error: ', totalErr)

    epochs+=1
    n = np.linalg.norm(w)
    ww=w/n
    ww1 = [ww[1],-ww[0]]
    ww2 = [-ww[1],ww[0]]

#    print('epoch: ', epochs)
#    print('w: ', w)

    if totalErr==0.0:
        learned = True
print(min(myValues[:,0]))
plt.plot([ww1[0], ww2[0]],[float(ww1[0])*slope,float(ww2[0])*slope],'--b' )
plt.plot([ww1[0], ww2[0]],[ww1[1], ww2[1]],'--g')
plt.scatter(myValues[:,0], myValues[:,1], c=myValues[:,3], s=10)
plt.show()
print('n= ', nValues)
print('epochs: ', epochs)


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