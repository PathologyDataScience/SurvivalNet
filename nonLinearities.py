import numpy
import theano
import theano.tensor as T

def ReLU(x, _):
    return theano.tensor.switch(x<0, 0, x)

def LeakyReLU(x, alpha):
    return theano.tensor.switch(x>0, x, x/alpha)

def Sigmoid(x, _):
	return T.nnet.sigmoid(x)
