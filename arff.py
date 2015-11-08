# -*- coding: utf-8 -*-
"""
Created on Sun Nov  8 13:04:29 2015

@author: rishiraj
"""
#%%
import numpy
from sys import argv
f = open('output.txt', 'w');
f.write("@relation dataset\n\n");
for i in range(1,2049):
    f.write("@attribute feature"+str(i)+" numeric");
    f.write("\n");

f.write("@attribute class{"),
for i in range(1,101):
    f.write(str(i)+", "),

f.write("}\n\n");
f.write("@data\n");
f.close();