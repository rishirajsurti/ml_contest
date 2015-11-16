# -*- coding: utf-8 -*-
"""
Created on Sun Nov  1 20:40:38 2015

@author: rishiraj
"""
#%%
import os
import glob
import numpy as np
import scipy
import csv
from sklearn.metrics import accuracy_score
from collections import Counter
os.chdir('/home/rishiraj/cs5011/contest/ml_contest/');
#%%
def most_common(lst):
    return max(set(lst), key=lst.count);
    
def ret_avg(lst):
    return int(np.mean(lst));
    
files = ['svm/svm_rs_output_gaussian.csv',
         'svm/svm_rs_output_polynomial.csv', 
         'svm/svm_rs_output_sigmoid.csv',
         'svm/svm_rs_output_gaussian_100.txt',
         'svm/svm_rs_output_gaussian_10000.txt',
         'svm/svm_rs_output_gaussian_100.txt',
         'svm/svm_rs_output_gaussian_10000.txt',
         'svm/svm_rs_output_gaussian_sklearn.txt',
         'svm/svm_rs_output_gaussian_sklearn_1000.txt', 
         'bayes/bayes_rs_output_bernoulli.csv', 
         'bayes/bayes_rs_output_multinomial.csv',
         'knn/7nn_output_rv.csv',
         'lda/lda_output_rv.csv',
         'nn/nn_sp_200.txt',
         'nn/nn_sp_200.txt',
         'svm/svm_rs_output_sigmoid_one_vs_rest_10000.txt',
         ] 
# add your files here;

#files = ['nn/nn_sp_200.txt']
#files= ['svm_rs_edited/svm_rs_output_gaussian_10000.txt']
#,'svm/svm_rs_output_polynomial.csv',

#'lda/lda_rv_output.csv',

test_Y=[]
for line in open('test_Y_ref.csv').readlines():
    test_Y.append(map(float,line.strip().split(","))[0]);

#%%
#test_Y2 = test_Y;    
final = np.zeros(len(test_Y));
outputs=[];

for i in xrange(len(files)):
    test_Y=[]
    for line in open(files[i]).readlines():
        test_Y.append(map(float,line.strip().split(" "))[0]);
    outputs.append(test_Y);    

test_Y=[]
for line in open('test_Y_ref.csv').readlines():
    test_Y.append(map(float,line.strip().split(","))[0]);
    
for i in xrange(len(final)):
    op=[];
    for j in xrange(len(outputs)):
        op.append(outputs[j][i]); #j'th file, i'th element
    final[i] = most_common(op); # need more files
    #final[i] = ret_avg(op);

accuracy_score(test_Y, final)
#0.91660851360781581

#%% write to file
f=open('test_Y.txt','w');
for i in xrange(len(final)):
    f.write(str(int(final[i])));
    f.write("\n");
f.close();
