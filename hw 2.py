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
def getValues(n):
    xb = (np.random.rand(n,1)*2-1)/2-0.5
    yb = (np.random.rand(n,1)*2-1)/2+0.5
    tb = np.ones([n,1])
    cb = np.tile('k',(n,1))
    xr = (np.random.rand(n,1)*2-1)/2+0.5
    yr = (np.random.rand(n,1)*2-1)/2-0.5
    tr = -np.ones([n,1])
    cr = np.tile('r',(n,1))
    b = np.concatenate((xb,yb,tb,cb),axis=1)
    r = np.concatenate((xr,yr,tr,cr),axis=1)
    inputs = np.concatenate((b,r),axis=0)
    myDF = DataFrame({
    'x1' : inputs[:,0],
    'x2' : inputs[:,1],
    'Targets' : inputs[:,2],
    'Color': inputs[:,3]
    })

    return(myDF)

nValues = 10
myValues = getValues(nValues)
fig = plt.figure()
#x = np.arange(min(myValues.x1), max(myValues.x1)+1, 1.0)
#np.arange()
plt.scatter(myValues.x1, myValues.x2, c=myValues.Color, s=40)
plt.show()

"""
(b) Run the perceptron learning algorithm on the data set above.
Report the number of updates that the algorithm takes before convergin.
Plot the examples {(xn,yn)}, the target function f, and the final hypothesis
g in the same figure.
Comment on whether f is close to g.
"""

def response(x,w):
    """ perceptron output """
    y = x[0]*w[0]+x[1]*w[1] # dot product between w and x
    if y >= 0:
        return 1
    else:
        return -1
    
    
w = np.random.rand(nValues,1)*2-1 # weights
learned = False
epochs = 0
while not learned:
    globalError = 0.0
    for x in myValues: # for each sample
        r = response(x,w)    
        if x[2] != r: # if we have a wrong response
            iterError = x[2] - r # desired response - actual response
            self.updateWeights(x,iterError)
            globalError += abs(iterError)
            iteration += 1
        if globalError == 0.0 or iteration >= 100: # stop criteria
            print('iterations',iteration)
            learned = True # stop learning
    
    
    
    




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