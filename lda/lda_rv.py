# coding: utf-8
from sklearn.lda import LDA
from numpy import *
from sklearn.metrics import accuracy_score

trainData = loadtxt('../../trainData.csv', delimiter=',', skiprows=1)
testData = loadtxt('../../testData.csv', delimiter=',',skiprows=1)

clf = LDA(n_components=2048)
clf.fit(trainData[:,:-1], trainData[:,-1])

labelsPredicted = clf.predict(testData[:,:-1])
acc = accuracy_score(testData[:, -1], labelsPredicted)

print acc
# 0.783 is the accuracy score

f=open('lda_rv_output.csv', 'w')

for i in xrange(len(labelsPredicted)):
    f.write(str(labelsPredicted[i]));
    f.write("\n");

f.close();    
