# coding: utf-8
from sklearn.lda import LDA
import numpy as np
from sklearn.metrics import accuracy_score

print 'Loading train data...'
X = np.loadtxt('../../train_X.csv', delimiter=',')
y = np.loadtxt('../../train_Y.csv', dtype=int, delimiter=',')
# X = np.loadtxt('../../train_data_X.csv', delimiter=' ')
# y = np.loadtxt('../../train_data_Y.csv', dtype=int, delimiter=' ')

print 'Normalising train data...'
for i in range(len(X.T)):
	mean = np.mean(X[:, i])
	if mean != 0:
		variance = np.mean((X[:, i] - mean)**2)
		X[:, i] = (X[:, i] - mean)/np.sqrt(variance)

print 'Loading test data...'
testData_X = np.loadtxt('../../test_X.csv', delimiter=',')
# testData_X = np.loadtxt('../../test_data_X.csv', delimiter=' ')

print 'Normalising test data...'
for i in range(len(testData_X.T)):
	mean = np.mean(testData_X[:, i])
	if mean != 0:
		variance = np.mean((testData_X[:, i] - mean)**2)
		testData_X[:, i] = (testData_X[:, i] - mean)/np.sqrt(variance)

# testData_Y = np.loadtxt('../../test_data_Y.csv', dtype=int, delimiter=' ')

print 'Performing classification using LDA...'
clf = LDA(n_components=testData_X.shape[1])
clf.fit(X, y)

labelsPredicted = clf.predict(testData_X)

# print accuracy_score(testData_Y, labelsPredicted)
# 1.0 is the accuracy score

np.savetxt("lda_output_rv.csv", labelsPredicted, fmt="%.0f", delimiter='\n')

print 'Done'