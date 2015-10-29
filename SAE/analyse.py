# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 16:23:33 2015

@author: syouse3
"""


import os
import cPickle
import numpy as np
def plotresults():
    #measure = 'lpl'
    msr = 'ci'
    path = '/home/syouse3/git/Survivalnet/SAE/results/Brain-Oct28/FinalExperiment/'        
    paths = []    
    firsts = []
    maxes = []
    lasts = []
    for f in os.listdir(path):
        if f.endswith(msr):
            if 'nl22-hs100' in f:
                print f
                paths.append(f)

    for i in range(len(paths)):
        f = file(path + paths[i], 'rb')
        loaded_objects = []
        loaded_objects = (cPickle.load(f))
        print loaded_objects        
        firsts[i] = loaded_objects[0]
        print firsts[i]
        lasts[i] = loaded_objects[-1]
        print lasts[i]        
        maxes[i] = np.max(loaded_objects)
        print maxes[i]        
        f.close()
