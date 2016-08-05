@author: Safoora Yousefi
"""
import numpy as np
from time import clock
from BayesOpt_costfunc import bayesopt_costfunc
import theano

def tune(nonlin):
    params = {}
    start = clock()
    layers = [1, 3, 5, 7, 9, 10]
    hsizes = [10, 50, 100, 150, 200, 300]
    drates = [.0, .1, .3, .5, .7, .9]
    x_set = []
    min_cost = 1
    best_query = [10, 10, 10]
    for l in layers:
	for h in hsizes:
            for d in drates:
	        x_set.append([l, h, d])
		params = [l, h, d]
		cur_cost = bayesopt_costfunc(params)
		if cur_cost < min_cost: 
		    min_cost = cur_cost
		    best_query = params
		print "Query: ", params, " Cost: ", cur_cost, " BestQuery: ", best_query, "MinCost: ", min_cost  
    print "Running time:", clock() - start, "seconds"
if __name__=='__main__':
	tune(None) 
