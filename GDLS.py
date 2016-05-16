# -*- coding: utf-8 -*-
"""
Created on Thu Mar 17 01:26:50 2016

@author: Safoora
"""
#import warnings
import theano
import time
import warnings
import numpy
import theano.tensor as T
from scipy.optimize.linesearch import LineSearchWarning
import scipy
from LineSearch import line_search_wolfe1, line_search_wolfe2, LineSearchWarning, _LineSearchError

def _line_search_wolfe12(f, fprime, xk, pk, gfk, old_fval, old_old_fval,
                         **kwargs):
    """
    Same as line_search_wolfe1, but fall back to line_search_wolfe2 if
    suitable step length is not found, and raise an exception if a
    suitable step length is not found.
    Raises
    ------
    _LineSearchError
        If no suitable step size is found
    """
    ret = line_search_wolfe1(f, fprime, xk, pk, gfk,
                             old_fval, old_old_fval,
                             **kwargs)

    if ret[0] is None:
        print 'line search failed: try different one.'
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', LineSearchWarning)
            ret = line_search_wolfe2(f, fprime, xk, pk, gfk,
                                     old_fval, old_old_fval)

    if ret[0] is None:
        raise _LineSearchError()

    return ret
            
class GDLS(object):

    def __init__(self, model, x, o, atrisk):
        self.cost = model.riskLayer.cost
        self.params = model.params
        is_tr = T.iscalar('is_train')

        #change shape of params to array   
        
        self.theta_shape = sum([self.params[i].get_value().size for i in range(len(self.params))])
        #theta_shape += model.riskLayer.W.get_value().size
        
        self.old_old_fval = None       
        N = self.theta_shape
        print self.theta_shape
        
        self.theta = theano.shared(value=numpy.zeros(self.theta_shape, dtype=theano.config.floatX))
        self.theta.set_value(numpy.concatenate([e.get_value().ravel() for e in
            self.params]), borrow = "true")
            
        self.gradient = theano.function(on_unused_input='ignore',
                                   inputs=[is_tr],
                                   outputs = T.grad(self.cost(o, atrisk), self.params),
                                   givens = {model.x:x, model.o:o, model.AtRisk:atrisk, model.is_train:is_tr},
                                   name='gradient')
        self.cost_func = theano.function(on_unused_input='ignore',
                                   inputs=[is_tr],
                                   outputs = self.cost(o, atrisk),
                                   givens = {model.x:x, model.o:o, model.AtRisk:atrisk, model.is_train:is_tr},
                                   name='cost_func')   
            
    #@profile
    def f(self, theta_val):
        #model.reset_weight(params)
        #print "theta = ", theta_val
        self.theta.set_value(theta_val)
        idx = 0
        for i in range(len(self.params)):
            p = self.theta.get_value()[idx:idx + self.params[i].get_value().size]
            p = p.reshape(self.params[i].get_value().shape)
            idx += self.params[i].get_value().size
            self.params[i].set_value(p)
        #print "params diff:", sum(sum(abs(params[0].get_value() - tmp)))

        c = -self.cost_func(1) 
        #print "cost =", c
        return c
  
    
    #@profile
    def fprime(self, theta_val):
        #model.reset_weight(params)
        self.theta.set_value(theta_val)
        idx = 0
        for i in range(len(self.params)):
            p = self.theta.get_value()[idx:idx + self.params[i].get_value().size]
            p = p.reshape(self.params[i].get_value().shape)
            idx += self.params[i].get_value().size
            self.params[i].set_value(p)

        gs = self.gradient(1)
        gf = numpy.concatenate([g.ravel() for g in gs])
        #print "gcost = ", -gf
        return -gf
   
    #Gradient Descent with line search
    def gd_ls(self, f, x0, fprime):
        #print "Next iteration of bfgs"
        self.theta_t = x0
        self.old_fval = f(self.theta_t)
        self.gf_t = fprime(x0)
        self.rho_t = -self.gf_t
        try:
            #print "starting line search:"
            self.eps_t, fc, gc, self.old_fval, self.old_old_fval, gf_next = \
                 _line_search_wolfe12(f, fprime, self.theta_t, self.rho_t, self.gf_t,
                                      self.old_fval, self.old_old_fval, amin=1e-100, amax=1e100)
        except _LineSearchError:
            print 'Line search failed to find a better solution.\n'         
            theta_next = self.theta_t + self.gf_t * .0001
            return theta_next
        print "Line Search Success! eps = ", self.eps_t
        theta_next = self.theta_t + self.eps_t * self.rho_t
        return theta_next 
 
    def GDLS(self):
	of = self.gd_ls
	theta_val = of(
                    f=self.f,
                    x0=self.theta.get_value(),
                    fprime=self.fprime,
		    )
        self.theta.set_value(theta_val)
        """theta_val,_,_ = of(
                    func=self.f,
                    x0=self.theta.get_value(),
                    fprime=self.fprime)
        self.theta.set_value(theta_val)"""
        idx = 0
        for i in range(len(self.params)):
            p = self.theta.get_value()[idx:idx + self.params[i].get_value().size]
            p = p.reshape(self.params[i].get_value().shape)
            idx += self.params[i].get_value().size
            self.params[i].set_value(p)        
        return
