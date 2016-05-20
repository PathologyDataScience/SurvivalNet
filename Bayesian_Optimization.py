# -*- coding: utf-8 -*-
"""
Created on Sat Apr  2 22:17:31 2016

@author: Ayine
"""

import bayesopt
import numpy as np
from time import clock
from BayesOpt_costfunc import bayesopt_costfunc
import theano

def tune(nonlin):
    params = {}
    params['n_iterations'] = 216
    params['n_iter_relearn'] = 1
    params['n_init_samples'] = 2
    
    print "*** Model Selection with BayesOpt ***"
    
    n = 3                   # n dimensions
    #lb = np.array([1,  10,  .0])
    #ub = np.array([10, 300, .9])

    start = clock()
    #mvalue, x_out, error = bayesopt.optimize(bayesopt_costfunc, n, lb, ub, params)
    layers = [1, 3, 5, 7, 9, 10]
    hsizes = [10, 50, 100, 150, 200, 300]
    drates = [0.0, .1, .3, .5, .7, .9]
    x_set = []
    for l in layers:
	for h in hsizes:
        for d in drates:
	        x_set.append([l, h, d])
    x_set = np.array(x_set, dtype=float)
    #x_set = np.array([[1, 3, 5, 7, 9, 10],\
    #                  [10, 50, 100, 150, 200, 300],\
    #                  [0, 1, 3, 5, 7, 9]], dtype=float).transpose()
    
    mvalue, x_out, error = bayesopt.optimize_discrete(bayesopt_costfunc, x_set, params)
    print "Result", mvalue, "at", x_out
    print "Running time:", clock() - start, "seconds"
    return mvalue, x_out, error
