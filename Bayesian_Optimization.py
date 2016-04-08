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

def tune(i, nonlin):
    params = {}
    params['n_iterations'] = 300
    params['n_iter_relearn'] = 1
    params['n_init_samples'] = 2
    
    print "*** Model Selection with BayesOpt ***"
    
    n = 6                   # n dimensions
    if (nonlin == theano.tensor.nnet.relu):
        lb = np.array([1, 10,  .0005, .0009, 0, i])
        ub = np.array([5, 300, .00121,    .001,  .5, i])
    else:
        lb = np.array([1, 10,  .0005, .0005, 0, i])
        ub = np.array([5, 300, .1,    .1,  .5, i])

    start = clock()
    mvalue, x_out, error = bayesopt.optimize(bo_costfunc, n, lb, ub, params)
    
    #mvalue, x_out, error = bayesopt.optimize_discrete(bo_costfunc, x_set, params)
    print "Result", mvalue, "at", x_out
    print "Running time:", clock() - start, "seconds"
    return mvalue, x_out, error
    
