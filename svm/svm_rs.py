# -*- coding: utf-8 -*-
"""
Created on Sun Nov  1 09:40:42 2015

@author: rishiraj
"""
#%%
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

with open('../../train1_X.csv') as csvfile:
    reader = csv.DictReader(csvfile)
            
        


#%%
#linear
m = svm_train(train_data_target, train_data_features, '-t 0')
#m = svm_train(train_data_target, train_data_features, '-t 0 -v 5') #cross validation k=5
svm_save_model('svm_linear.model',m)
p_labels, p_acc, p_vals = svm_predict(test_data_target, test_data_features,m)

#polynomial
m2 = svm_train(train_data_target, train_data_features, '-t 1')
#m2 = svm_train(train_data_target, train_data_features, '-t 1 -v 10') #cross validation
svm_save_model('svm_polynomial.model',m2)
p_labels, p_acc, p_vals = svm_predict(test_data_target, test_data_features,m2)
#Cross Validation Accuracy = 63.1213%, k=10
#Accuracy = 70% (56/80) (classification)

#radial/gaussian
m3 = svm_train(train_data_target, train_data_features, '-t 2')
#m3 = svm_train(train_data_target, train_data_features, '-t 2 -v 10')
svm_save_model('svm_radial.model',m3)
p_labels, p_acc, p_vals = svm_predict(test_data_target, test_data_features,m3)

#sigmoid
m4 = svm_train(train_data_target, train_data_features, '-t 3')
svm_save_model('svm_sigmoid.model',m4)
p_labels, p_acc, p_vals = svm_predict(test_data_target, test_data_features,m4)
#Accuracy = 25% (20/80) (classification)

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