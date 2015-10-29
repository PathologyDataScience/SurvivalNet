__author__ = 'Song'
import numpy
import theano
import theano.tensor as T
import theano.tensor.extra_ops as Te


def test():
    m = T.ivector('m')
    v = T.ivector('v')
    sum = T.sum(m - v)
    f = theano.function([m, v], sum)
    print f([[1, 2, 3, 4], [1, 2, 3, 4]])


class LogLikelihoodLayer(object):
    def __init__(self, input, n_in, n_out):
        rng = numpy.random.RandomState(89677)
        # initialize with 0 the weights W as a matrix of shape (n_in, n_out)
        self.W  = theano.shared(
            value=numpy.asarray(
                rng.uniform(
                    low=-numpy.sqrt(6. / (n_in + n_out)),
                    high=numpy.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)
                ),
                # rng.normal(size=(n_in, n_out)),
                dtype=theano.config.floatX
            ),
            name='W',
            borrow=True
        )

        # initialize the baises b as a vector of n_out 0s
        # self.b = theano.shared(
        #     value=numpy.zeros(n_out, dtype=theano.config.floatX),
        #     name='b',
        #     borrow=True
        # )
        self.input = input
        self.output = T.dot(self.input, self.W ).flatten()
        self.params = [self.W ]

    def cost(self, observed, at_risk):
        prediction = self.output
        exp = T.exp(prediction)[::-1]
        partial_sum = Te.cumsum(exp)[::-1]  + 1 # get the reversed partial cumulative sum
        log_at_risk = T.log(partial_sum[at_risk])
        diff = prediction - log_at_risk
        cost = T.sum(T.dot(observed, diff))
        return cost

    def gradient(self, observed, at_risk):
        prediction = self.output
        risk = T.exp(prediction)
        product = self.input * (risk * T.ones((1, self.input.shape[0])))
        numerator = Te.cumsum(product[::-1])[::-1][at_risk]
        denominator = Te.cumsum(risk[::-1])[::-1][at_risk] * T.ones((1, self.input.shape[0]))
        numerator = numerator.flatten()
        denominator = denominator.flatten()
        gradient = T.dot(observed, self.input - (numerator / denominator))
        return gradient

    def reset_weight(self, params):
        self.W.set_value(params)


if __name__ == '__main__':
    logLayer = LogLikelihoodLayer(input=T.matrix('x'), n_in=2, n_out=1)
    logLayer.gradient([0,0], [1,1])
