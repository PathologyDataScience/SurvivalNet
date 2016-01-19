#!/usr/bin/env python

import bayesopt
import numpy as np
import sys
from time import clock
from bo_costfunc import bo_costfunc

# Function for testing = bo_costfunc(flr, nl, hs, plr, dor)


def bayesopt_tuning(i = 0, nonlin = 'sig'):
    params = {}
    params['n_iterations'] = 300
    params['n_iter_relearn'] = 1
    params['n_init_samples'] = 2
    
    print "*** Model Selection with BayesOpt ***"
    
    n = 6                   # n dimensions
    if (nonlin == 'relu'):
        lb = np.array([1, 10,  .0005, .0009, 0, i])
        ub = np.array([5, 300, .00121,    .001,  .5, i])
    elif (nonlin == 'sig'):
        lb = np.array([1, 10,  .0005, .0005, 0, i])
        ub = np.array([5, 300, .1,    .1,  .5, i])
    else: #LEAKY RELU
        # nl, hs, ptlr, ftlr, do_rate, split_id, leakage 
        lb = np.array([1, 10,  .0005, .0009, 0, i, 5.5])
        ub = np.array([5, 300, .00121,    .001,  .5, i, 5.5])

    
    #lb = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    #ub = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    #x_set = np.array([[1, 2, 2, 3, 5, 10],\
    #                  [10, 50, 100, 150, 200, 300],\
    #                  [.0005, .001, .005, .01, .05, .1],\
    #                  [.0005, .001, .005, .01, .05, .1],\
    #                  [0, .1, .2, .3, .5, .5],\
    #                  [i, i, i, i, i, i]]).transpose()
    start = clock()
    mvalue, x_out, error = bayesopt.optimize(bo_costfunc, n, lb, ub, params)
    
    #mvalue, x_out, error = bayesopt.optimize_discrete(bo_costfunc, x_set, params)
    print "Result", mvalue, "at", x_out
    print "Running time:", clock() - start, "seconds"
    return mvalue, x_out, error
    
