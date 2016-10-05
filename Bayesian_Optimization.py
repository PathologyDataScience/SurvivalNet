# -*- coding: utf-8 -*-
"""
Created on Wed Oct 5 22:17:31 2016

@author: Safoora
"""
import sys
import bayesopt
import numpy as np
from TFCostFunction import cost_func
import theano

def tune(i):
    params = {}
    params['n_iterations'] = 30
    params['n_iter_relearn'] = 1
    params['n_init_samples'] = 2
    
    print "*** Model Selection with BayesOpt ***"
    
    n = 4                   # n dimensions
    lb = np.array([1,  10,  .0, i])
    ub = np.array([3, 500, .9, i+.5])

    mvalue, x_out, error = bayesopt.optimize(cost_func, n, lb, ub, params)
    print "Result", mvalue, "at", x_out
    return mvalue, x_out, error
if __name__=='__main__':
	tune(None)     
