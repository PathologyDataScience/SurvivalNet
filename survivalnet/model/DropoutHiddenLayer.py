__docformat__ = 'restructedtext en'

import theano
import theano.tensor as T
from .HiddenLayer import HiddenLayer

class DropoutHiddenLayer(HiddenLayer):
    def __init__(self, rng, input, n_in, n_out, is_train,
                 activation, dropout_rate, W=None, b=None):
        super(DropoutHiddenLayer, self).__init__(
                rng=rng, input=input, n_in=n_in, n_out=n_out, W=W, b=b,
                activation=activation)
        train_output = _dropout_from_layer(rng, self.output, p=dropout_rate)
        test_output = self.output * (1 - dropout_rate)
        self.output = T.switch(T.eq(is_train, 1), train_output, test_output)   # if train
		
def _dropout_from_layer(rng, layer, p):
    """
    p is the probablity of dropping a unit
    """
    srng = T.shared_randomstreams.RandomStreams(rng.randint(999999))
    mask = theano.printing.Print("droping out =")(srng.binomial(n=1, p=1-p, size=layer.shape))
    # p=1-p because 1's indicate keep and p is prob of dropping
    # check = theano.printing.Print('mask')(mask)
    # The cast is important because
    # int * float32 = float64 which pulls things off the gpu
    output = layer * T.cast(mask, theano.config.floatX)
    return output


