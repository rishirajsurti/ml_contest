# coding: utf-8
#%%
import os
import glob
import numpy as np
import scipy
import csv
from collections import Counter
os.chdir('/home/rishiraj/cs5011/contest/ml_contest/');

#%%
from sklearn import neighbors
from sklearn.metrics import accuracy_score
import numpy as np

print 'Loading train data...'
X = np.loadtxt('../train_X.csv', delimiter=',')
y = np.loadtxt('../train_Y.csv', dtype=int, delimiter=',')
# X = np.loadtxt('../../train_data_X.csv', delimiter=' ')
# y = np.loadtxt('../../train_data_Y.csv', dtype=int, delimiter=' ')

#%%
print 'Normalising train data...'
for i in range(len(X.T)):
	mean = np.mean(X[:, i])
	if mean != 0:
		variance = np.mean((X[:, i] - mean)**2)
		X[:, i] = (X[:, i] - mean)/np.sqrt(variance)

print 'Loading test data...'
testData_X = np.loadtxt('../test_X.csv', delimiter=',')
# testData_X = np.loadtxt('../../test_data_X.csv', delimiter=' ')

print 'Normalising test data...'
for i in range(len(testData_X.T)):
	mean = np.mean(testData_X[:, i])
	if mean != 0:
		variance = np.mean((testData_X[:, i] - mean)**2)
		testData_X[:, i] = (testData_X[:, i] - mean)/np.sqrt(variance)

# testData_Y = np.loadtxt('../../test_data_Y.csv', delimiter=' ')

# For the given training dataset, after splitting it into
# 80 p.c. training and 20 p.c. test sets, we obtained k = 7
# gives the highest accuracy between k = 5 to 15 considering
# the problem of computations for higher value of k
#%%
n = 1
print 'Performing kNN classification with k = %d...' % (n)
clf = neighbors.KNeighborsClassifier(n, weights='uniform')
clf.fit(X, y)

labelsPredicted = clf.predict(testData_X)
#%%

# print "%d : %f" % (n, accuracy_score(testData_Y, labelsPredicted))
np.savetxt("nn_output_rv.csv" % labelsPredicted, fmt="%.0f", delimiter='\n')

print 'Done'
#%%
f=open('1nn_output.txt','w');
for i in xrange(len(labelsPredicted)):
    f.write(str(int(labelsPredicted[i])));
    f.write("\n");
f.close();