# coding: utf-8
# This file takes the following files as input:
# train_data_X.csv, train_data_Y.csv, test_data_X.csv, test_data_Y.csv
# and generates the following files as output:
# trainData.csv, testData.csv

"""
Created on Sun Nov  8 13:04:29 2015

@author: rishiraj
"""

from numpy import *
from sys import argv

print 'Loading train data and train labels...'
f = open('../trainData.arff', 'w');
trainData_X = loadtxt('../train_data_X.csv', delimiter=' ')
trainData_Y = loadtxt('../train_data_Y.csv', delimiter=' ')

trainData = zeros((trainData_X.shape[0], trainData_X.shape[1] + 1))
trainData[:, :-1] = trainData_X
trainData[:, -1] = trainData_Y

print 'Saving ARFF train data file...'
f.write("@relation dataset\n\n");
for i in range(1,2049):
    f.write("@attribute feature"+str(i)+" numeric");
    f.write("\n");

f.write("@attribute class{"),
for i in range(1,100):
    f.write(str(i)+", "),

f.write(str(100));
f.write("}\n\n");
f.write("@data\n");
savetxt(f, trainData)
f.close();

print 'Loading test data and test labels...'
f = open('../testData.arff', 'w');
testData_X = loadtxt('../test_data_X.csv', delimiter=' ')
testData_Y = loadtxt('../test_data_Y.csv', delimiter=' ')

testData = zeros((testData_X.shape[0], testData_X.shape[1] + 1))
testData[:, :-1] = testData_X
testData[:, -1] = testData_Y

print 'Saving ARFF test data file...'
f.write("@relation dataset\n\n");
for i in range(1,2049):
    f.write("@attribute feature"+str(i)+" numeric");
    f.write("\n");

f.write("@attribute class{"),
for i in range(1,100):
    f.write(str(i)+", "),

f.write(str(100));
f.write("}\n\n");
f.write("@data\n");
savetxt(f, testData)
f.close();