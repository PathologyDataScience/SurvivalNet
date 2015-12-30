
import numpy

import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

from partialLogLikelihood import LogLikelihoodLayer
from mlp import HiddenLayer, DropoutHiddenLayer
from dA import dA
from nonLinearities import ReLU, LeakyReLU

# start-snippet-1
class SdA(object):
    def __init__(
        self,
        numpy_rng,
        theano_rng=None,
        n_ins=784,
        hidden_layers_sizes=[500, 500],
        n_outs=10,
        corruption_levels=[0.1, 0.1],
        at_risk=None,
        at_risk_test=None,
        drop_out=False,
        pretrain_dropout=False,
        dropout_rate=0.1,
        non_lin=None,
        alpha=None
    ):
        """ This class is made to support a variable number of layers.
        :type numpy_rng: numpy.random.RandomState
        :param numpy_rng: numpy random number generator used to draw initial
                    weights
        :type theano_rng: theano.tensor.shared_randomstreams.RandomStreams
        :param theano_rng: Theano random generator; if None is given one is
                           generated based on a seed drawn from `rng`
        :type n_ins: int
        :param n_ins: dimension of the input to the sdA
        :type n_layers_sizes: list of ints
        :param n_layers_sizes: intermediate layers size, must contain
                               at least one value
        :type n_outs: int
        :param n_outs: dimension of the output of the network
        :type corruption_levels: list of float
        :param corruption_levels: amount of corruption to use for each
                                  layer
        """

        self.sigmoid_layers = []
        self.dA_layers = []
        self.params = []
        self.n_layers = len(hidden_layers_sizes)
        self.drop_out = drop_out
        self.pretrain_dropout = pretrain_dropout
        self.is_train = T.iscalar('is_train')
        self.is_pretrain_dropout = T.iscalar('is_pretrain_dropout')
        #assert self.n_layers > 0
        if not theano_rng:
            theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))

        # allocate symbolic variables for the data
        self.x = T.matrix('x')  # the data is presented as rasterized images
        self.o = T.ivector('o')  # observed death or not, 1 is death, 0 is right censored
        # end-snippet-1

        # The SdA is an MLP, for which all weights of intermediate layers
        # are shared with a different denoising autoencoders
        # We will first construct the SdA as a deep multilayer perceptron,
        # and when constructing each sigmoidal layer we also construct a
        # denoising autoencoder that shares weights with that layer
        # During pretraining we will train these autoencoders (which will
        # lead to chainging the weights of the MLP as well)
        # During finetunining we will finish training the SdA by doing
        # stochastich gradient descent on the MLP

        # start-snippet-2
        if self.n_layers == 0:            
            self.logLayer = LogLikelihoodLayer(
                input=self.x,
                n_in=n_ins,
                n_out=n_outs,
                rng = numpy_rng
            )
        else:    
            for i in xrange(self.n_layers):
                # construct the sigmoidal layer
    
                # the size of the input is either the number of hidden units of
                # the layer below or the input size if we are on the first layer
                if i == 0:
                    input_size = n_ins
                else:
                    input_size = hidden_layers_sizes[i - 1]
    
                # the input to this layer is either the activation of the hidden
                # layer below or the input of the SdA if you are on the first
                # layer
                if i == 0:
                    layer_input = self.x
                else:
                    layer_input = self.sigmoid_layers[-1].output
    
                sigmoid_layer = DropoutHiddenLayer(rng=numpy_rng,
                                            input=layer_input,
                                            n_in=input_size,
                                            n_out=hidden_layers_sizes[i],
                                            activation=non_lin,
                                            alpha=alpha,
                                            dropout_rate=dropout_rate,
                                            is_train=self.is_train,
                                            pretrain_dropout=self.is_pretrain_dropout) \
                    if drop_out else HiddenLayer(rng=numpy_rng,
                                            input=layer_input,
                                            n_in=input_size,
                                            n_out=hidden_layers_sizes[i],
                                            activation=non_lin,
                                            alpha=alpha)
                # add the layer to our list of layers
                self.sigmoid_layers.append(sigmoid_layer)
                # its arguably a philosophical question...
                # but we are going to only declare that the parameters of the
                # sigmoid_layers are parameters of the StackedDAA
                # the visible biases in the dA are parameters of those
                # dA, but not the SdA
                self.params.extend(sigmoid_layer.params)
    
                # Construct a denoising autoencoder that shared weights with this
                # layer
                dA_layer = dA(numpy_rng=numpy_rng,
                              theano_rng=theano_rng,
                              input=layer_input,
                              n_visible=input_size,
                              n_hidden=hidden_layers_sizes[i],
                              W=sigmoid_layer.W,
                              bhid=sigmoid_layer.b,
                              non_lin=non_lin,
                              alpha=alpha)
                self.dA_layers.append(dA_layer)
            # end-snippet-2
            # set up the partial sum
    
            # We now need to add a logistic layer on top of the MLP
            self.logLayer = LogLikelihoodLayer(
                input=self.sigmoid_layers[-1].output,
                n_in=hidden_layers_sizes[-1],
                n_out=n_outs,
                rng = numpy_rng
            )
    
        self.params.extend(self.logLayer.params)
        self.finetune_cost = self.logLayer.cost(self.o, at_risk)
        self.test_cost = self.logLayer.cost(self.o, at_risk_test)
        self.last_gradient = self.logLayer.gradient(self.o, at_risk)

    def pretraining_functions(self, train_set_x, batch_size, pretrain_mini_batch):
        # index to a [mini]batch
        index = T.lscalar('index')  # index to a minibatch
        corruption_level = T.scalar('corruption')  # % of corruption to use
        learning_rate = T.scalar('lr')  # learning rate to use
        # begining of a batch, given `index`
        batch_begin = index * batch_size
        # ending of a batch given `index`
        batch_end = batch_begin + batch_size
        if self.pretrain_dropout:
            is_pretrain_dropout = numpy.cast['int32'](1)
            is_train = numpy.cast['int32'](1)
        else:
            is_pretrain_dropout = numpy.cast['int32'](0)
            is_train = numpy.cast['int32'](0)   # value does not matter
        if pretrain_mini_batch:
            train_set_x = train_set_x[batch_begin: batch_end]
        pretrain_fns = []
        for dA in self.dA_layers:
            # get the cost and the updates list
            cost, updates = dA.get_cost_updates(corruption_level,
                                                learning_rate)
            # compile the theano function
            fn = theano.function(
                on_unused_input='ignore',
                inputs=[
                    index,
                    theano.Param(corruption_level, default=0.2),
                    theano.Param(learning_rate, default=0.1)
                ],
                outputs=cost,
                updates=updates,
                givens={
                    self.x: train_set_x,
                    self.is_pretrain_dropout: is_pretrain_dropout,
                    self.is_train: is_train
                }
            )
            # append `fn` to the list of functions
            pretrain_fns.append(fn)

        return pretrain_fns

    def build_finetune_functions(self, train_X, test_X, train_observed, test_observed, learning_rate):
        index = T.lscalar('index')  # index to a [mini]batch
        pretrain = numpy.cast['int32'](0)   # pretrain set to 0
        is_train = numpy.cast['int32'](1)
        # compute the gradients with respect to the model parameters
        gparams = T.grad(self.finetune_cost, self.params)

        # compute list of fine-tuning updates
        # updates = [
        #     (param, param + theano.printing.Print('gradient')(gparam) * learning_rate)
        #     for param, gparam in zip(self.params, gparams)
        # ]
        #updates = [
        #    (param, theano.printing.Print('param')(param) + gparam * learning_rate)
        #    for param, gparam in zip(self.params, gparams)
        #]
        updates = [
            (param, param + gparam * learning_rate)
            for param, gparam in zip(self.params, gparams)
        ]
        train_fn = theano.function(
            on_unused_input='ignore',
            inputs=[index],
            outputs=self.finetune_cost,
            updates=updates,
            givens={
                self.x: train_X,
                self.o: train_observed,
                self.is_pretrain_dropout: numpy.cast['int32'](1 - pretrain),
                self.is_train: numpy.cast['int32'](is_train)
            },
            name='train'
        )

        output_fn = theano.function(
            on_unused_input='ignore',
            inputs=[index],
            outputs=self.logLayer.output,
            givens={
                self.x: test_X,
                self.o: train_observed,
                self.is_pretrain_dropout: numpy.cast['int32'](1 - pretrain),
                self.is_train: numpy.cast['int32'](1 - is_train)

            },
            name='output'
        )

        grad_fn = theano.function(
            on_unused_input='ignore',
            inputs=[index],
            outputs=gparams,
            givens={
                self.x: train_X,
                self.o: train_observed,
                self.is_pretrain_dropout: numpy.cast['int32'](1 - pretrain),
                self.is_train: numpy.cast['int32'](1 - is_train)
            },
            name='output'
        )

        last_out_fn = theano.function(
            on_unused_input='ignore',
            inputs=[index],
            outputs=self.logLayer.input,
            givens={
                self.x: train_X,
                self.o: train_observed,
                self.is_pretrain_dropout: numpy.cast['int32'](0),
                self.is_train: numpy.cast['int32'](1 - is_train)
            },
            name='last_output'
        )

        test_cost_fn = theano.function(
            on_unused_input='ignore',
            inputs=[index],
            outputs=self.test_cost,
            givens={
                self.x: test_X,
                self.o: test_observed,
                self.is_pretrain_dropout: numpy.cast['int32'](1 - pretrain),
                self.is_train: numpy.cast['int32'](1 - is_train)

            },
            name='test_cost'
        )

        return train_fn, output_fn, grad_fn, last_out_fn, test_cost_fn

    def reset_weight(self, params):
        for i in xrange(self.n_layers):
            self.sigmoid_layers[i].reset_weight((params[2*i], params[2*i+1]))
        self.logLayer.reset_weight(params[-1])

    def reset_weight_by_rate(self, rate):
        for i in xrange(self.n_layers):
            self.sigmoid_layers[i].reset_weight_by_rate(rate)