import os
import glob
import numpy
import scipy
import csv
from matplotlib import pyplot as plt
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.cross_validation import train_test_split #optional
from sklearn.cross_validation import cross_val_score, KFold
from scipy.stats import sem

import svm
import svmutil
from svm import *
from svmutil import *

os.chdir('/home/rishiraj/cs5011/contest/ml_contest/svm');

##K-Fold cross validation function
def evaluate_cross_validation(clf, data_features, data_target, K):
    cv = KFold(len(data_target), K, shuffle=True, random_state=0);
    scores = cross_val_score(clf, data_features, data_target, cv=cv);
    print scores;
    print ("Mean score: {0:.3f} (+/-{1:.3f})").format(np.mean(scores), sem(scores));


##for results 
def calc_results(svc_):
    svc_.fit(train_data_features, train_data_target); #fit
    print svc_.score(test_data_features, test_data_target); #test    
    test_data_predict = svc_.predict(test_data_features); #predict
    
    print metrics.classification_report(test_data_target, test_data_predict)
    print metrics.confusion_matrix(test_data_target, test_data_predict)
    evaluate_cross_validation(svc_, train_data_features, train_data_target, 10) #10 fold cross-validation

##for libsvm, args need to be list/tuple:

#print len(train_data_features[0])
#note: svm_train(<class_values>, <features>)

#%% Data
train_X=[]
for line in open('../../train_data_X.csv').readlines():
    train_X.append(map(float,line.strip().split(" ")));

train_Y=[]
for line in open('../../train_data_Y.csv').readlines():
    train_Y.append(map(float,line.strip().split(" "))[0]);

test_X=[]
for line in open('../../test_data_X.csv').readlines():
    test_X.append(map(float,line.strip().split(" ")));

test_Y=[]
for line in open('../../test_data_Y.csv').readlines():
    test_Y.append(map(float,line.strip().split(" "))[0]);
## data acquired     
#%%
len(train_Y[0])    
    
#%%
#linear
m = svm_train(train_Y, train_X, '-t 0')
#m = svm_train(train_data_target, train_data_features, '-t 0 -v 5') #cross validation k=5
p_labels, p_acc, p_vals = svm_predict(test_Y, test_X, m)

f=open('svm_rs_output_polynomial.csv','w');
for i in xrange(len(p_labels)):
    f.write(str(p_labels[i]));
    f.write("\n");
f.close();

#polynomial
m2 = svm_train(train_Y, train_X, '-t 1')
#m2 = svm_train(train_data_target, train_data_features, '-t 1 -v 10') #cross validation
p_labels, p_acc, p_vals = svm_predict(test_Y, test_X,m2)
#90.649% (2598/2866) (classification)

#radial/gaussian
m3 = svm_train(train_Y, train_X, '-t 2')
#m3 = svm_train(train_data_target, train_data_features, '-t 2 -v 10')
p_labels, p_acc, p_vals = svm_predict(test_Y, test_X,m3)

#sigmoid
m4 = svm_train(train_Y, train_X, '-t 3')
p_labels, p_acc, p_vals = svm_predict(test_Y, test_X,m4)

'''
#Linear SVC
svc_l = SVC(kernel="linear")
calc_results(svc_l);

#Poly
svc_p = SVC(kernel="poly");
calc_results(svc_p);

#sigmoid
svc_s = SVC(kernel="sigmoid");
calc_results(svc_s);
'''

'''
svc_l.fit(train_data_features, train_data_target); #fit
test_data_predict = svc_l.predict(test_data_features); #predict

print svc_l.score(train_data_features, train_data_target);
print metrics.classification_report(test_data_target, test_data_predict)
print metrics.confusion_matrix(test_data_target, test_data_predict)

evaluate_cross_validation(svc_l, train_data_features, train_data_target, 10)
'''