# coding: utf-8
from sklearn import neighbors
from sklearn.metrics import accuracy_score
from numpy import *

print 'Loading train and test data...'
trainData = loadtxt('../../trainData.csv', delimiter=',', skiprows=1)
testData = loadtxt('../../testData.csv', delimiter=',',skiprows=1)

print 'Performing kNN classification'
for n_neighbors in range(5, 11):
    clf = neighbors.KNeighborsClassifier(n_neighbors, weights='uniform')
    clf.fit(trainData[:, :-1], trainData[:, -1])
    labelsPredicted = clf.predict(testData[:, :-1])
    print "%d : %f" % (n_neighbors, accuracy_score(testData[:, -1], labelsPredicted))
    

