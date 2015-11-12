# -*- coding: utf-8 -*-
"""
Created on Sun Nov  1 20:36:09 2015

@author: rishiraj
"""

#%%
import glob
import os
import numpy as np
from sklearn.naive_bayes import GaussianNB
from itertools import groupby
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB

os.chdir('/home/rishiraj/cs5011/contest/ml_contest/bayes')

#%% Data
train_X=[]
for line in open('../../train_X.csv').readlines():
    train_X.append(map(float,line.strip().split(",")));

train_Y=[]
for line in open('../../train_Y.csv').readlines():
    train_Y.append(map(float,line.strip().split(" "))[0]);

test_X=[]
for line in open('../../test_X.csv').readlines():
    test_X.append(map(float,line.strip().split(",")));


#%% normalize the features
# zero mean, unit variance
#train_data_features_n = train_data_features;
train_X_abs = train_X
dummy = [];

for j in xrange(2048):
    dummy = [];
    for i in xrange(len(train_X)):
        dummy.append(train_X[i][j]);
    
    m = numpy.mean(dummy);
    sd = numpy.std(dummy);

    for i in xrange(len(train_X)):
        train_X_abs[i][j] = (float)((train_X[i][j]-m)/(1+sd));
        train_X_abs[i][j] = abs(train_X_abs[i][j])

#test data
test_X_abs = test_X
dummy = [];

for j in xrange(2048):
    dummy = [];
    for i in xrange(len(test_X)):
        dummy.append(test_X[i][j]);
    
    m = numpy.mean(dummy);
    sd = numpy.std(dummy);
    for i in xrange(len(test_X)):
        test_X_abs[i][j] = ((test_X[i][j]-m)/(1+sd)) ;
        test_X_abs[i][j] = abs(test_X_abs[i][j])
## data acquired     
#%%
################FITTING DATA

###########Multinomial
clf = MultinomialNB()
clf.fit(train_X_n, train_Y); ##alpha = 1 default( for smoothing)
predicted = clf.predict(test_X)
accuracy_score(test_Y, predicted)
#0.81472435450104674
#clf.class_log_prior_


######### Bernoulli

clf = BernoulliNB();
clf.fit(train_X_n, train_Y); ##alpha = 1 default( for smoothing)
predicted = clf.predict(test_X)
accuracy_score(test_Y, predicted)

#%% write to file
f=open('bayes_rs_output_bernoulli.csv','w');
for i in xrange(len(predicted)):
    f.write(str(predicted[i]));
    f.write("\n");
f.close();

