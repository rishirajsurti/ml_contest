# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 21:32:16 2015

@author: rishiraj
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 17:56:58 2015

@author: rishiraj
"""
#%%
import math
import random
import statistics
import os 
import glob
import numpy
import scipy
import cv2
import pandas
from scipy.misc import imread
from matplotlib import pyplot as plt
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.cross_validation import train_test_split #optional
from sklearn.cross_validation import cross_val_score, KFold
from scipy.stats import sem
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import precision_recall_curve
import svm
import svmutil
from svm import *
from svmutil import *


os.chdir('/home/rishiraj/cs5011/ann/code/');
random.seed(0)
#%%
# calculate a random number where:  a <= rand < b
def rand(a, b):
    return (b-a)*random.random() + a

# Make a matrix (we could use NumPy to speed this up)
def makeMatrix(I, J, fill=0.0):
    m = []
    for i in range(I):
        m.append([fill]*J)
    return m

# our sigmoid function, tanh is a littlen_inputcer than the standard 1/(1+e^-x)
def sigmoid(x):
    return 1/(1+numpy.exp(-x));    
    
# derivative of our sigmoid function, in terms of the output (i.e. y)
def dsigmoid(y):
    return y*(1-y);
    
#%%
class NN:
    def __init__(self,n_input, n_hidden, n_output):
        """NN constructor.
        
       n_input, n_hidden, n_output are the number of input, hidden and output nodes.
        """
        
        #Number of input, hidden and output nodes.
        self.n_input =n_input  + 1 # +1 for bias node
        self.n_hidden = n_hidden  + 1 # +1 for bias n_outputde
        self.n_output = n_output

        # activations for nodes
        self.ai = [1.0]*self.n_input
        self.ah = [1.0]*self.n_hidden
        self.ao = [1.0]*self.n_output
        
        # create weights
        self.wi = makeMatrix(self.n_input, self.n_hidden)
        self.wo = makeMatrix(self.n_hidden, self.n_output)
        
        # set them to random vaules
        for i in range(self.n_input):
            for j in range(self.n_hidden):
                self.wi[i][j] = rand(-1, 1)
      
        for j in range(self.n_hidden):
            for k in range(self.n_output):
                self.wo[j][k] = rand(-1, 1);

    def update(self, inputs):
        
        # input activations
        for i in range(self.n_input - 1):
            self.ai[i] = inputs[i]

        # hidden activations
        for j in range(self.n_hidden - 1):
            total = 0.0
            for i in range(self.n_input):
                total += self.ai[i] * self.wi[i][j] #many i's, one j
            self.ah[j] = sigmoid(total)

        # output activations
        for k in range(self.n_output):
            total = 0.0
            for j in range(self.n_hidden):
                total += self.ah[j] * self.wo[j][k] #many j's, one k
            self.ao[k] = sigmoid(total)
            
        
        return self.ao[:]


    def backPropagate(self, targets, N):
        
        # calculate error terms for output
        output_deltas = [0.0] * self.n_output
        for k in range(self.n_output):
            output_deltas[k] = targets[k] - self.ao[k]
            output_deltas[k] = dsigmoid(self.ao[k]) * output_deltas[k]

        
        # calculate error terms for hidden
        hidden_deltas = [0.0] * self.n_hidden
        for j in range(self.n_hidden):
            error = 0.0
            for k in range(self.n_output):
                error += output_deltas[k]*self.wo[j][k]
            hidden_deltas[j] = dsigmoid(self.ah[j]) * error;

        # update output weights
        for j in range(self.n_hidden):
            for k in range(self.n_output):
                change = output_deltas[k]*self.ah[j]
                self.wo[j][k] = self.wo[j][k] + N*change ;
                 
        # update input weights
        for i in range(self.n_input):
            for j in range(self.n_hidden):
                change = hidden_deltas[j]*self.ai[i]
                self.wi[i][j] = self.wi[i][j] + N*change ;
                
        # calculate error
        error = 0.0
        for k in range(len(targets)):
            error += 0.5*((targets[k]-self.ao[k])**2)
        return error


    def test(self, features, verbose = False):
        tmp = []
        for f in xrange(len(features)):
            if verbose:
                print '-> ',[int(round(n)) for n in self.update(features[f])];
            tmp.append(self.update(features[f]))

        #return tmp

        
    def weights(self):
        print 'Input weights:'
        for i in range(self.n_input):
            print self.wi[i]
        print
        print 'Output weights:'
        for j in range(self.n_hidden):
            print self.wo[j]

    def train(self, features, target, iterations=1000, N=0.5, verbose = False):
        """Train the neural network.  
        
        N is the learning rate.
        M is momentum factor
        """
        for i in xrange(iterations):
            error = 0.0
            for f in xrange(len(features)):
                self.update(features[f])
                tmp = self.backPropagate(target[f], N)
                error += tmp
                
            if i % 1 == 0:
                print 'error %-14f' % error


#%% Data
train_X=[]
for line in open('../../train_X.csv').readlines():
    train_X.append(map(float,line.strip().split(",")));

train_Y=[]
for line in open('../../train_Y.csv').readlines():
    train_Y.append(map(float,line.strip().split(","))[0]);

test_X=[]
for line in open('../../test_X.csv').readlines():
    test_X.append(map(float,line.strip().split(",")));
'''
test_Y=[]
for line in open('../../test_data_Y.csv').readlines():
    test_Y.append(map(float,line.strip().split(" "))[0]);
## data acquired     
'''
#%% normalize the features
# zero mean, unit variance
#train_data_features_n = train_data_features;
train_X_n = train_X
dummy = [];

for j in xrange(2048):
    dummy = [];
    for i in xrange(len(train_X)):
        dummy.append(train_X[i][j]);
    
    m = numpy.mean(dummy);
    sd = numpy.std(dummy);

    for i in xrange(len(train_X)):
        train_X_n[i][j] = (float)((train_X[i][j]-m)/1+sd) ;

#test data

test_X_n = test_X
dummy = [];

for j in xrange(2048):
    dummy = [];
    for i in xrange(len(test_X)):
        dummy.append(test_X[i][j]);
    
    m = numpy.mean(dummy);
    sd = numpy.std(dummy);
    for i in xrange(len(test_X)):
        test_X_n[i][j] = ((test_X[i][j]-m)/1+sd) ;

#%%

n = NN(2048, 2048, 100);

# train it with some patterns then test it.
#pattern, iterations, Learning rate
n.train(train_X_n, train_Y, 100, 0.5);
n.test(test_X_n, verbose = True);

n.weights(); #print weights
#%% 
import csv
predicted = []
pred = []
with open('../data/result_class.csv','rb') as f:
    reader = csv.reader(f);
    for row in reader:
        predicted.append(map(int,row))

for i in xrange(len(predicted)):
    pred.append(predicted[i][0])

test_data_target[:20]
pred[:20]

precision_recall_fscore_support(test_data_target[:20], pred[:20], average='binary')
precision_recall_fscore_support(test_data_target[20:40], pred[20:40], average='binary')
precision_recall_fscore_support(test_data_target[40:60], pred[40:60], average='binary')
precision_recall_fscore_support(test_data_target[60:80], pred[60:80], average='binary')

#%%
#plot epoch vs. error
error_=[]
error = []
with open('../data/error.csv','rb') as f:
    reader = csv.reader(f);
    for row in reader:
        error_.append(row)
for i in xrange(len(error_)):
    error.append(error_[i][0])

error
plt.plot(error)
plt.ylabel('error');
plt.xlabel('epochs');
plt.show()
#%%