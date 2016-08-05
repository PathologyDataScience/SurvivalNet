# -*- coding: utf-8 -*-
"""
Created on Sat Apr  2 22:17:31 2016

@author: Ayine
"""
import sys
sys.path.append('./../')
import bayesopt
import numpy as np
from time import clock
from .CostFunction import cost_func, aggr_st_cost_func, st_cost_func, panorg_cost_func
import theano

def tune(i):
    params = {}
    params['n_iterations'] = 100
    params['n_iter_relearn'] = 1
    params['n_init_samples'] = 2
    
    print "*** Model Selection with BayesOpt ***"
    n = 5                # n dimensions
# params: #layer, width, dropout, L2reg, nonlinearity 
    lb = np.array([1,  10, .0, 0, i])
    ub = np.array([5, 500, .9, 1, i+.5])

    start = clock()
    mvalue, x_out, error = bayesopt.optimize(panorg_cost_func, n, lb, ub, params)
    #layers = [1, 3, 5, 7, 9, 10]
    #hsizes = [10, 50, 100, 150, 200, 300]
    #drates = [0.0, .1, .3, .5, .7, .9]
    #layers = [1, 5, 10]
    #hsizes = [10, 100, 1000, 10000]
    #drates = [.0, .1, .5, .9]
	#reg = []
    #x_set = []
    #for l in layers:
	#for h in hsizes:
    #        for d in drates:
	#        x_set.append([l, h, d])
    #x_set = np.array(x_set, dtype=float)
    #x_set = np.array([[1, 3, 5, 7, 9, 10],\
    #                  [10, 50, 100, 150, 200, 300],\
    #                  [0, 1, 3, 5, 7, 9]], dtype=float).transpose()
    #print x_set.shape

    #print x_set
    #mvalue, x_out, error = bayesopt.optimize_discrete(cost_func, x_set, params)
    print "Result", mvalue, "at", x_out
    print "Running time:", clock() - start, "seconds"
    return mvalue, x_out, error
if __name__=='__main__':
	tune(0)     
