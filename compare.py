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
os.chdir('/home/rishiraj/cs5011/contest/ml_contest/');
#%%

files = ['/bayes/bayes_rs_output.csv', '/svm/svm_rs_output.csv'] # add your files here;

test_Y=[]
for line in open('../../test_data_Y.csv').readlines():
    test_Y.append(map(float,line.strip().split(" "))[0]);

final = np.zeros(len(test_Y));
