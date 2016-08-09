import numpy
import theano.tensor as T
import theano


def RiskBackpropagate(Model, profile):
    """
    Generates feature weights of a provided Neuralnetwork. The gradient
    vector of the Neuralnetwork with respect to its features is a vector
    of feature weights.

    Parameters:
    ----------
    Model: theano deep learning model
    a Neuralnetwork containing at least one layer, and theano functions
    to feed the model with specif profile of feature values.
    profile : numpy matrix
    profile contains feature values to be used as the input for feeding
    the model.
    Output:
    ----------
    Feature_weights : numpy nd array
    an array of the feature weights.
    """

    Feature_weights = []

    def Model_Gradient(Model):
        X = T.matrix('X')
        AtRisk = T.ivector('AtRisk')
        Observed = T.ivector('Observed')
        Is_train= T.scalar('Is_train' , dtype='int32')
        partial_derivative = theano.function(on_unused_input='ignore',
                                             inputs=[X, AtRisk, Observed,Is_train],
                                             outputs=T.grad(Model.riskLayer
                                                            .output[0], Model.x
                                                            ),
                                             givens={
                                                 Model.x: X,
                                                 Model.o: AtRisk,
                                                 Model.AtRisk: Observed,
                                                 Model.is_train:Is_train
                                                 },
                                             name='partial_derivative')
        
        return partial_derivative

    for sample_profile in profile:
        sample_O = numpy.array([0])
        sample_O = sample_O.astype(numpy.int32)
        sample_T = numpy.array([0])
        sample_T = sample_T.astype(numpy.int32)
        partial_derivative = Model_Gradient(Model)
        Feature_weights.append(partial_derivative(sample_profile, sample_O,
                                                  sample_T, 0)[0])
        
    Feature_weights = numpy.asarray(Feature_weights)
    return Feature_weights
