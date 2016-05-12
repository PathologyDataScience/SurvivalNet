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
    params['n_iterations'] = 100
    params['n_iter_relearn'] = 1
    params['n_init_samples'] = 2
    
    print "*** Model Selection with BayesOpt ***"
    
    n = 3                   # n dimensions
    lb = np.array([1,  10,  .0])
    ub = np.array([10, 50, .9])

    start = clock()
    mvalue, x_out, error = bayesopt.optimize(bayesopt_costfunc, n, lb, ub, params)
    
    #mvalue, x_out, error = bayesopt.optimize_discrete(bo_costfunc, x_set, params)
    print "Result", mvalue, "at", x_out
    print "Running time:", clock() - start, "seconds"
    return mvalue, x_out, error
    
