import numpy as np
import matplotlib.pyplot as mp
import scipy as sc
from sklearn import linear_model
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn import preprocessing

def myknn(nn):
	knn=KNeighborsClassifier(n_neighbors=nn)
	in_train_pp = preprocessing.scale(in_train)
	knn.fit(in_train_pp, out_train)
	out_test_results = knn.predict(preprocessing.scale(in_test))
	print accuracy_score(out_test,out_test_results)
	
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
x_test = preprocessing.scale(x_t)

x_train = preprocessing.scale(x_)
rows = len(y_)
y_train = preprocessing.scale(y_)

knn(rows*cols/100)
