# -*- coding: utf-8 -*-
"""
Created on Sun Nov  1 20:40:38 2015

@author: rishiraj
"""
#%%
import os
import glob
import numpy 
import scipy
import csv
from sklearn.metrics import accuracy_score
from collections import Counter
os.chdir('/home/rishiraj/cs5011/contest/ml_contest/');
#%%
def most_common(lst):
    return max(set(lst), key=lst.count);
    
files = ['svm/svm_rs_output.csv', 'bayes/bayes_rs_output.csv'] # add your files here;

test_Y=[]
for line in open('../test_data_Y.csv').readlines():
    test_Y.append(map(float,line.strip().split(" "))[0]);

final = np.zeros(len(test_Y));
outputs=[];

for i in xrange(len(files)):
    test_Y=[]
    for line in open(files[i]).readlines():
        test_Y.append(map(float,line.strip().split(" "))[0]);
    outputs.append(test_Y);    
    
for i in xrange(len(final)):
    op=[];
    for j in xrange(len(outputs)):
        op.append(outputs[j][i]); #j'th file, i'th element
    final[i] = most_common(op); # need more files
    

accuracy_score(test_Y, final)
#0.91660851360781581

#%% write to file
f=open('final_output.csv','w');
for i in xrange(len(final)):
    f.write(str(final[i]));
    f.write("\n");
f.close();