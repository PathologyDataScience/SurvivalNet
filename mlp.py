"""
This tutorial introduces the multilayer perceptron using Theano.
 A multilayer perceptron is a logistic regressor where
instead of feeding the input to the logistic regression you insert a
intermediate layer, called the hidden layer, that has a nonlinear
activation function (usually tanh or sigmoid) . One can use many such
hidden layers making the architecture deep. The tutorial will also tackle
the problem of MNIST digit classification.
.. math::
    f(x) = G( b^{(2)} + W^{(2)}( s( b^{(1)} + W^{(1)} x))),
References:
    - textbooks: "Pattern Recognition and Machine Learning" -
                 Christopher M. Bishop, section 5
"""
__docformat__ = 'restructedtext en'

import numpy

import theano
import theano.tensor as T
from nonLinearities import ReLU, LeakyReLU, Sigmoid

# start-snippet-1
class HiddenLayer(object):
    def __init__(self, rng, input, n_in, n_out, W=None, b=None,
                 activation=T.tanh, alpha=0):
        """
        Typical hidden layer of a MLP: units are fully-connected and have
        sigmoidal activation function. Weight matrix W is of shape (n_in,n_out)
        and the bias vector b is of shape (n_out,).
        NOTE : The nonlinearity used here is tanh
        Hidden unit activation is given by: tanh(dot(input,W) + b)
        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights
        :type input: theano.tensor.dmatrix
        :param input: a symbolic tensor of shape (n_examples, n_in)
        :type n_in: int
        :param n_in: dimensionality of input
        :type n_out: int
        :param n_out: number of hidden units
        :type activation: theano.Op or function
        :param activation: Non linearity to be applied in the hidden
                           layer
        """
        self.input = input
        # end-snippet-1

        # `W` is initialized with `W_values` which is uniformely sampled
        # from sqrt(-6./(n_in+n_hidden)) and sqrt(6./(n_in+n_hidden))
        # for tanh activation function
        # the output of uniform if converted using asarray to dtype
        # theano.config.floatX so that the code is runable on GPU
        # Note : optimal initialization of weights is dependent on the
        #        activation function used (among other things).
        #        For example, results presented in [Xavier10] suggest that you
        #        should use 4 times larger initial weights for sigmoid
        #        compared to tanh
        #        We have no info for other function, so we use the same as
        #        tanh.
        if W is None:
            W_values = numpy.asarray(
                rng.uniform(
                    low=-numpy.sqrt(6. / (n_in + n_out)),
                    high=numpy.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)
                ),
                dtype=theano.config.floatX
            )
            #if activation == Sigmoid:
             #   W_values *= 4
                #print W_values
            W = theano.shared(value=W_values, name='W', borrow=True)

        if b is None:
            b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)

        self.W = W
        self.b = b
        
        lin_output = T.dot(input, self.W) + self.b
        self.output = (
            lin_output if activation is None
            else activation(lin_output, alpha)
        )
        # parameters of the model
        self.params = [self.W, self.b]

    def reset_weight(self, params):
        self.W.set_value(params[0])
        self.b.set_value(params[1])

    def reset_weight_by_rate(self, rate):
        if rate != 0:
            self.W.set_value(self.W.get_value() / rate)
            self.b.set_value(self.b.get_value() / rate)


def _dropout_from_layer(rng, layer, p):
    """
    p is the probablity of dropping a unit
    """
    srng = theano.tensor.shared_randomstreams.RandomStreams(
            rng.randint(999999))
    # p=1-p because 1's indicate keep and p is prob of dropping
    mask = srng.binomial(n=1, p=1-p, size=layer.shape)
    # check = theano.printing.Print('mask')(mask)
    # The cast is important because
    # int * float32 = float64 which pulls things off the gpu
    output = layer * T.cast(mask, theano.config.floatX)
    return output


class DropoutHiddenLayer(HiddenLayer):
    def __init__(self, rng, input, n_in, n_out, is_train, pretrain_dropout,
                 activation, dropout_rate, W=None, b=None, alpha=0):
        super(DropoutHiddenLayer, self).__init__(
                rng=rng, input=input, n_in=n_in, n_out=n_out, W=W, b=b,
                activation=activation, alpha=alpha)
        train_output = _dropout_from_layer(rng, self.output, p=dropout_rate)
        test_output = self.output * (1 - dropout_rate)
        self.output = T.switch(T.eq(pretrain_dropout, 0), self.output,  # if not pretrain
                               T.switch(T.eq(is_train, 1), train_output, test_output))   # if train
