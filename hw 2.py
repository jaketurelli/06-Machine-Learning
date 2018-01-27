# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 21:34:13 2018

@author: Jake
"""
import numpy as np
import matplotlib.pyplot as plt
from pandas import *

def getValues(n):
    xb = (np.random.rand(n,1)*2-1)/2-0.5
    yb = (np.random.rand(n,1)*2-1)/2+0.5
    tb = (np.ones([n,1]))
    xr = (np.random.rand(n,1)*2-1)/2+0.5
    yr = (np.random.rand(n,1)*2-1)/2-0.5
    tr = -(np.ones([n,1]))
    b = np.concatenate((xb,yb,tb),axis=1)
    r = np.concatenate((xr,yr,tr),axis=1)
    inputs = np.concatenate((b,r),axis=0)

    myDF = DataFrame({
    'x1' : inputs[:,0],
    'x2' : inputs[:,1],
    'Targets' : inputs[:,2]
    })

    return(myDF)

myValues = getValues(5)
print(myValues)
colormap = np.array(['r', 'k'])
plt.scatter(myValues.x1, myValues.x2, c=colormap[myValues.Targets], s=40)
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

"""
(b) Run the perceptron learning algorithm on the data set above.
Report the number of updates that the algorithm takes before convergin.
Plot the examples {(xn,yn)}, the target function f, and the final hypothesis
g in the same figure.
Comment on whether f is close to g.
"""

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