import numpy as np
import csv
from sklearn.decomposition import PCA
from sklearn.lda import LDA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cross_validation  import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support
import Image
import glob
import numpy as np
from sklearn import preprocessing
from sklearn import svm
from sklearn import cross_validation
import pickle
from sklearn.cross_validation import KFold
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
mydata=[]
with open('train_X.csv', 'rb') as f:
    c=csv.reader(f, delimiter=',', quotechar=' ')
    for row in c:
        mydata.append(map(float,row))  

x_=np.array(mydata) 

mydata=[]
with open('train_Y.csv', 'rb') as f:
    c=csv.reader(f, delimiter=',', quotechar=' ')
    for row in c:
        mydata.append(map(float,row))  

y_=np.array(mydata) 

mydata=[]
with open('test_X.csv', 'rb') as f:
    c=csv.reader(f, delimiter=',', quotechar=' ')
    for row in c:
        mydata.append(map(float,row))  

x_t=np.array(mydata) 

x_train = preprocessing.scale(x_)
rows = len(y_)
#y_train = preprocessing.scale(y_)
#y_train = y_
x_test = preprocessing.scale(x_t)

rows,cols = np.shape(x_train)

y_train = np.zeros((rows,100))
#print np.shape(y_train)
for j in range(rows):
	y_train[j][int(y_[j])]=1


#print np.shape(x_train)
#print np.shape(y_train)

X = x_train
y = y_train



def nonlin(x, deriv=False):
    if(deriv==True):
        return 1-x*x
    return .5 * (1 + np.tanh(.5 * x))

k = np.ones(rows)

k=np.reshape(k,(rows,1))
X = np.hstack((k,X))
trows,tcols = np.shape(x_test)

o = np.ones(trows)

o = np.reshape(o,(trows,1))
test = np.hstack((o,x_test))


np.random.seed(1)

ans = []
yans = []
yans= np.argmax(y,axis=1)

alpha = 2*np.random.random((cols+1, 500)) - 1
beta = 2*np.random.random((501,100)) - 1

xaxis = []
txaxis = []
rvalue = []
testans = []
testrvalue =[]
tl2_error = []
beta_temp = beta
alpha_temp = alpha
for iteration in xrange(50):
	print iteration+1
	result = []
	for i in range (rows):
		l0 = X[i]
		l1 = np.zeros(501)
		l1[0] = 1
		#print np.shape(l0)
		#print np.shape(alpha)
		l1[1:] = nonlin(np.dot(l0,alpha))
		l2 = nonlin(np.dot(l1,beta))
		l2_error = y[i]-l2
		l2_delta = -2*l2_error*nonlin(l2,deriv = True)
		
		l1_delta = nonlin(l1[1:],deriv=True)*np.dot(beta[1:,:],l2_delta)
		for i in range(100):
			beta_temp[:,i] -= 0.001*l2_delta[i]*l1
				
		for i in range(500):
			alpha_temp[:,i] -= 0.001*l1_delta[i]*l0
		result.append(l2)
	beta = beta_temp
	alpha = alpha_temp
	xaxis.append(iteration)
	#print 'Trainerror: %d', sum(sum((y-l2)*(y-l2)))
	rvalue.append(sum(sum((y-l2)*(y-l2)))/rows)
	ans = np.argmax(result,axis=1)
	#print 'Trainaccuracy'+accuracy_score(ans,yans)

#test = x_test

tl2mat = []
#print test
for i in range (trows):		
	tl0 = test[i]
	tl1 = np.zeros(501)
	tl1[0] = 1
	tl1[1:] = nonlin(np.dot(tl0,alpha))
	tl2 = nonlin(np.dot(tl1,beta))
	tl2mat.append(tl2)
	#tl2_error.append(y_test[i]-tl2)
tl2_error = np.array(tl2_error)
#testrvalue.append(sum(sum(tl2_error*tl2_error))/trows)

tl2mat = np.array(tl2mat)
ans = np.argmax(result,axis=1)
#yans = np.argmax(y_test,axis=1)
outans = np.argmax(tl2mat,axis=1)
for x in range (len(outans)):
	outans[x] = int(outans[x])

np.savetxt("modnnout50.txt",outans)
"""
y_test=np.array(mydata) 

tl2_error = np.array(tl2_error)
y_test = np.array(y_test)
print accuracy_score(outans,yans)
precision, recall, fmeasure, support=precision_recall_fscore_support(outans,yans)


print fmeasure
"""
