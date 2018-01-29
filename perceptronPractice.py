# -*- coding: utf-8 -*-
"""
Created on Sun Jan 28 16:31:24 2018
Perception Practice
@author: Jake
"""
import numpy as np
import matplotlib.pyplot as plt
from pandas import *
#from pylab import rand,plot,show,norm

class Perceptron:
 def __init__(self):
  """ perceptron initialization """
  self.w = np.random.rand(2)*2-1 # weights
  self.learningRate = 1

 def response(self,x):
  """ perceptron output """
  y = x[0]*self.w[0]+x[1]*self.w[1] # dot product between w and x
  if y >= 0:
   return 1
  else:
   return -1

 def updateWeights(self,x,iterError):
  """
   updates the weights status, w at time t+1 is
       w(t+1) = w(t) + learningRate*(d-r)*x
   where d is desired output and r the perceptron response
   iterError is (d-r)
  """
  self.w[0] += self.learningRate*iterError*x[0]
  self.w[1] += self.learningRate*iterError*x[1]

 def train(self,data):
  """
   trains all the vector in data.
   Every vector in data must have three elements,
   the third element (x[2]) must be the label (desired output)
  """
  learned = False
  iteration = 0
  while not learned:
   globalError = 0.0
   for x in data: # for each sample
    r = self.response(x)
    if x[2] != r: # if we have a wrong response
     iterError = x[2] - r # desired response - actual response
     self.updateWeights(x,iterError)
     globalError += abs(iterError)
   iteration += 1
   if globalError == 0.0 or iteration >= 100: # stop criteria
    print('iterations',iteration)
    learned = True # stop learning


def generateData(n):
    """
      generates a 2D linearly separable dataset with n samples.
      The third element of the sample is the label
    """
    xb = np.arange(n)
    delta = np.random.uniform(-10,10, size=(n,))
    xb = (np.random.rand(n,1)*2-1)/2-0.5
    yb = (np.random.rand(n,1)*2-1)/2+0.5
    tb = np.ones([n,1])
    xr = (np.random.rand(n,1)*2-1)/2+0.5
    yr = (np.random.rand(n,1)*2-1)/2-0.5
    tr = -np.ones([n,1])
    b = np.concatenate((xb,yb,tb),axis=1)
    r = np.concatenate((xr,yr,tr),axis=1)
    inputs = np.concatenate((b,r),axis=0)
    return inputs

# 1) Generate Data
trainset = generateData(10) # train set generation
# 2) create perception object with random weights and learning Rate
perceptron = Perceptron()   # perceptron instance
perceptron.train(trainset)  # training
testset = generateData(10)  # test set generation

# Perceptron test
for x in testset:
 r = perceptron.response(x)
 if r != x[2]: # if the response is not correct
  print('error')
 if r == 1:
  plt.plot(x[0],x[1],'ob')
 else:
  plt.plot(x[0],x[1],'or')

# plot of the separation line.
# The separation line is orthogonal to w
n = np.linalg.norm(perceptron.w)
ww = perceptron.w/n
ww1 = [ww[1],-ww[0]]
ww2 = [-ww[1],ww[0]]
plt.plot([ww1[0], ww2[0]],[ww1[1], ww2[1]],'--k')
plt.show()