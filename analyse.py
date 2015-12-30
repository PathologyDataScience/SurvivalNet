# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 16:23:33 2015

@author: syouse3
"""


import os
import cPickle
import numpy as np
def analyse():
    #measure = 'lpl'
    msr = 'ci'

    path = '/Users/Ayine/pySurv/LUAD_results/sigmoid/forLee/lr/'        
    paths = []    
    firsts = []
    maxes = []
    lasts = []
    for f in os.listdir(path):
        if f.endswith(msr):
            print f
            paths.append(f)

    for p in paths:
        f = file(path + p, 'rb')
        loaded_objects = (cPickle.load(f))
        firsts.append(loaded_objects[0])
        lasts.append(loaded_objects[-1])
        maxes.append(np.max(loaded_objects))
        f.close()

    print lasts
    print "Mean: " + str(np.mean(lasts))
    print "Std: " + str(np.std(lasts))
    print firsts
    print "Mean: " + str(np.mean(firsts))
    print "Std: " + str(np.std(firsts))
    print maxes
    print "Mean: " + str(np.mean(maxes))
    print "Std: " + str(np.std(maxes))
if __name__ == '__main__':
    analyse()
