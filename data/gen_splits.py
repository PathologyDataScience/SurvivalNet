# -*- coding: utf-8 -*-
"""
Created on Tue Dec 22 14:29:41 2015

@author: Ayine
"""

import scipy.io as sio
import os
import numpy as np
import sys
sys.path.append('..')

def generate_split(i):
    seeds = np.arange(1, 100, 7)
    prng = np.random.RandomState(seeds[i])
    order = prng.permutation(np.arange(628))
    sio.savemat(os.path.join(os.getcwd(), 'Brain_PC/') + 'shuffle' + str(i) + '.mat', {'order':order})

def gen_splits(n):
    for i in range(n):
        generate_split(i)
        
if __name__ == '__main__':
    ## PARSET
    clinical = False
    gen_splits(n = 10)