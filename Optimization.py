# -*- coding: utf-8 -*-
"""
Created on Thu Mar 17 01:26:50 2016

@author: Ayine
"""
import theano.tensor as T
class Optimization(object):
    def SGD(self, cost, params, learning_rate):
        # compute the gradients with respect to the model parameters
        gparams = T.grad(cost, params)
        
        updates = [
            (param, param + gparam * learning_rate)
            for param, gparam in zip(params, gparams)
        ]
        return updates