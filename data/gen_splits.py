# -*- coding: utf-8 -*-
"""
Created on Tue Dec 22 14:29:41 2015

@author: Ayine
"""

import scipy.io as sio
import numpy as np
import sys
sys.path.append('..')
from loadMatData import VA, LUAD_P, LUSC_P, Brain_P, LUAD_G, LUSC_G
from sklearn.preprocessing import scale

def generate_split(i, X, O, T):
    seeds = np.arange(1, 100, 7)
    prng = np.random.RandomState(seeds[i])
    shuffled = np.insert(X, 0, O, axis=1)
    shuffled = np.insert(shuffled, 0, T, axis=1)
    prng.shuffle(shuffled)
    O = shuffled[:, 1].astype('int32')
    T = shuffled[:, 0].astype('int32')
    X = shuffled[:, 2:]
    
    
    #outputFileName = os.path.join('VA', 'shuffle' + str(i) + '.pickle')
    #f = file(outputFileName, 'wb')
    #cPickle.dump([X, T, O], f, protocol=cPickle.HIGHEST_PROTOCOL)
    #f.close()
    ## PARSET
    sio.savemat('/Users/Ayine/pySurv/data/Brain_P/' + 'shuffle' + str(i) + '.mat', {'X':X, 'T':T, 'C':1 - O})

def gen_splits(n, X, O, T):
    for i in range(n):
        generate_split(i, X, O, T)
        
if __name__ == '__main__':
    ## PARSET
    p = Brain_P
    mat = sio.loadmat(p)
    X = mat['X']
    X = X.astype('float64')
    X = scale(X)
       
    O = mat['C']
    T = mat['T']
    
    T = np.asarray([t[0] for t in T])
    #1 means event is observed    
    O = 1 - np.asarray([c[0] for c in O], dtype='int32')
    
    gen_splits(n = 10, X = X, O = O, T = T)