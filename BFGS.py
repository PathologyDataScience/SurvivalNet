# -*- coding: utf-8 -*-
"""
Created on Thu Mar 17 01:26:50 2016

@author: Safoora
"""
#import warnings
import theano
import warnings
import numpy
import theano.tensor as T
#from scipy.optimize.linesearch import LineSearchWarning
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
            
class BFGS(object):

    def __init__(self, model, x, o, atrisk, is_tr):
        self.cost = model.riskLayer.cost
        self.params = model.params

        #print "original params class: ", len(params), params[0].get_value().shape, params[1].get_value().shape, params[2].get_value().shape
        #change shape of params to array   
        
        self.theta_shape = sum([self.params[i].get_value().size for i in range(len(self.params))])
        #theta_shape += model.riskLayer.W.get_value().size
        
        self.old_old_fval = None       
        N = self.theta_shape
        I = numpy.eye(N, dtype=int)
        self.H_t = I

        print self.theta_shape
        
        self.theta = theano.shared(value=numpy.zeros(self.theta_shape, dtype=theano.config.floatX))
        self.theta.set_value(numpy.concatenate([e.get_value().ravel() for e in
            self.params]), borrow = "true")
            
        self.gradient = theano.function(on_unused_input='ignore',
                                   inputs=[],
                                   outputs = T.grad(self.cost(o, atrisk), self.params),
                                   givens = {model.x:x, model.o:o, model.AtRisk:atrisk, model.is_train:is_tr},
                                   name='gradient')
        self.cost_func = theano.function(on_unused_input='ignore',
                                   inputs=[],
                                   outputs = self.cost(o, atrisk),
                                   givens = {model.x:x, model.o:o, model.AtRisk:atrisk, model.is_train:is_tr},
                                   name='cost_func')   
            
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

        c = -self.cost_func() 
        #print "cost =", c
        return c
  
    
    def fprime(self, theta_val):
        #model.reset_weight(params)
        self.theta.set_value(theta_val)
        idx = 0
        for i in range(len(self.params)):
            p = self.theta.get_value()[idx:idx + self.params[i].get_value().size]
            p = p.reshape(self.params[i].get_value().shape)
            idx += self.params[i].get_value().size
            self.params[i].set_value(p)

        gs = self.gradient()
        gf = numpy.concatenate([g.ravel() for g in gs])
        #print "gcost = ", -gf
        return -gf
    def bfgs_min(self, f, x0, fprime):
        I = numpy.eye(len(x0), dtype=int)
        #print "Next iteration of bfgs"
        self.theta_t = x0
        self.old_fval = f(self.theta_t)
        self.gf_t = fprime(x0)
        if (numpy.linalg.norm(self.gf_t < self.tol))
            return        
        self.rho_t = -numpy.dot(self.H_t, self.gf_t)
        #print "diff = ", sum(abs(self.rho_t + self.gf_t))
        #print self.rho_t
        #print self.gf_t
        try:
            #print "starting line search:"
            self.eps_t, fc, gc, self.old_fval, self.old_old_fval, gf_next = \
                 _line_search_wolfe12(f, fprime, self.theta_t, self.rho_t, self.gf_t,
                                      self.old_fval, self.old_old_fval, amin=1e-100, amax=1e100)
        except _LineSearchError:
            print 'Line search failed to find a better solution.\n'         
            theta_next = self.theta_t + self.gf_t * .001
            return theta_next
        print "Line Search Success! eps = ", self.eps_t
        theta_next = self.theta_t + self.eps_t * self.rho_t
        #return self.theta_t + self.eps_t * self.rho_t
    
        delta_t = theta_next - self.theta_t
        #print "delta_t = ", delta_t
        self.theta_t = theta_next
        
        self.phi_t = gf_next - self.gf_t
        #print "phi_t = ", self.phi_t

        self.gf_t = gf_next
        
        #Wright, and Nocedal 'Numerical Optimization', 1999, pg. 198.    
        denom = 1.0 / (numpy.dot(self.phi_t, delta_t)) 
        #print "denom = ", denom

        A1 = I - delta_t[:, numpy.newaxis] * self.phi_t[numpy.newaxis, :] * denom
        A2 = I - self.phi_t[:, numpy.newaxis] * delta_t[numpy.newaxis, :] * denom
        #print "***estimating H***"        
        self.H_t = numpy.dot(A1, numpy.dot(self.H_t, A2)) + (denom * delta_t[:, numpy.newaxis] *
                                             delta_t[numpy.newaxis, :])
        #print "self.H_t = ", self.H_t

        #Goodfellow et al 'Deep Learning', 2016, chapter 8. 
        #TODO
        return theta_next

    def BFGS(self):

         if self.tol = None:
             self.tol = .0000001
#        updates = [
#            (params[i], self.bfgs_min(
#                    f=f,
#                    x0=theta.get_value(),
#                    fprime=fprime))]
        theta_val = self.bfgs_min(
                    f=self.f,
                    x0=self.theta.get_value(),
                    fprime=self.fprime)
        self.theta.set_value(theta_val)
        idx = 0
        for i in range(len(self.params)):
            p = self.theta.get_value()[idx:idx + self.params[i].get_value().size]
            p = p.reshape(self.params[i].get_value().shape)
            idx += self.params[i].get_value().size
            self.params[i].set_value(p)        
        return

        
