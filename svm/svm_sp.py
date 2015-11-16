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
x_test = preprocessing.scale(x_t)
'''
mydata=[]
with open('test_data_Y.csv', 'rb') as f:
    c=csv.reader(f, delimiter=',', quotechar=' ')
    for row in c:
        mydata.append(map(float,row)) 
 
y_test=np.array(mydata) 
'''



x_train = preprocessing.scale(x_)
rows = len(y_)
y_train = preprocessing.scale(y_)


print "Srikanth"
n=5
"""
myacc = []
def line(c):
    lin_svc=svm.SVC(kernel='linear',C=c)
    #np.savetxt("svmout.txt",lin_svc.predict(x_train))
    #print lin_svc.predict(x_train)
    scores_l=cross_validation.cross_val_score(lin_svc,x_test,y_test,scoring='accuracy',cv=n)
    mean=scores_l.mean()
    #path_pickle = './'
    #f=open(path_pickle+'lin_ker.pickle','wb')
    #pickle.dump(lin_svc,f)
    #f.close()
    print mean

def poly(c,g):
    poly_svc=svm.SVC(kernel='poly',C=c,gamma=g)
    scores_l=cross_validation.cross_val_score(poly_svc,x_test,y_test,scoring='accuracy',cv=n)
    mean=scores_l.mean()
    #path_pickle = './'
    #f=open(path_pickle+'poly_ker.pickle','wb')
    #pickle.dump(poly_svc,f)
    #f.close()
    print mean

def rbf(c,g):
    gaussian_svc=svm.SVC(kernel='rbf',C=c,gamma=g)
    np.savetxt("svmout.txt",gaussian_svc.predict(x_train))
    #scores_l=cross_validation.cross_val_score(gaussian_svc,x_test,y_test,scoring='accuracy',cv=n)
    #mean=scores_l.mean()
    #path_pickle = './'
    #f=open(path_pickle+'rbf_ker.pickle','wb')
    #pickle.dump(gaussian_svc,f)
    #f.close()
    #print mean

def sig(c,g, c0):
    sigmoid_svc=svm.SVC(kernel='sigmoid',C=c,gamma=g,coef0 = c0)
    np.savetxt("svmout.txt",sig_svc.predict(x_train))
    scores_l=cross_validation.cross_val_score(sigmoid_svc,x_test,y_test,scoring='accuracy',cv=n)
    mean=scores_l.mean()
    #path_pickle = './'
    #f=open(path_pickle+'sig_ker.pickle','wb')
    #pickle.dump(sigmoid_svc,f)
    #f.close()
    print mean
"""
rbf_svc = svm.SVC(kernel='rbf')
rbf_svc.fit(x_train,y_train)
sig_svc = svm.SVC(kernel='sigmoid')
sig_svc.fit(x_train,y_train)
rb = rbf_svc.predict(x_test)
si = sig_svc.predict(x_test)
rb = np.array(rb)
si = np.array(si)
for z in range (len(rb)):
	rb[z]=(int)(rb[z])
	si[z]=(int)(si[z])
np.savetxt("svmoutrbf.txt",rb)
np.savetxt("svmoutsig.txt",si)
#line(1)
#rbf(2.4,0.009)
#sig(13, 0.02, -2)
#poly(0.0005,0.4)

