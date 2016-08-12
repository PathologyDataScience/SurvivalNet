__author__ = 'Song'
import numpy
import theano
import theano.tensor as T
import theano.tensor.extra_ops as Te

class RiskLayer(object):
    def __init__(self, input, n_in, n_out, rng):
        # rng = numpy.random.RandomState(111111)
        # initialize randomly the weights W as a matrix of shape (n_in, n_out)
        self.W  = theano.shared(
            value = numpy.asarray(
                rng.uniform(
                    low=-numpy.sqrt(6. / (n_in + n_out)),
                    high=numpy.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)
                ),
                dtype=theano.config.floatX
            ),
            name='W',
            borrow=True
        )

        self.input = input
        self.output = T.dot(self.input, self.W ).flatten()
        self.params = [self.W ]

    def cost(self, observed, at_risk):
        prediction = self.output
        factorizedPred = prediction - prediction.max() #subtract maximum to facilitate computation
        exp = T.exp(factorizedPred)[::-1]
        partial_sum = Te.cumsum(exp)[::-1]  + 1 # get the reversed partial cumulative sum
        log_at_risk = T.log(partial_sum[at_risk]) + prediction.max() #add maximum back
        diff = prediction - log_at_risk
        cost = T.sum(T.dot(observed, diff))
        return cost

    def reset_weight(self, params):
        self.W.set_value(params)

