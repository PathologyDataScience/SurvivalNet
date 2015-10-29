__author__ = 'Song'
from loadData import load_training_data
from loadMatData import load_data
import matplotlib.pyplot as plt
from scipy.special import expit
from lifelines.utils import _naive_concordance_index
import theano.tensor as T
import numpy
import theano
from mlp import MLP
# from lifelines import KaplanMeierFitter
# kmf = KaplanMeierFitter()

VA = 'data/VA.mat'
LUAD_P = 'data/LUAD_P.mat'
LUSC_P = 'data/LUSC_P.mat'
Brain_P ='data/Brain_P.mat'


def main(learning_rate=0.0001, L1_reg=0.000, L2_reg=0.075, n_epochs=500,
             dataset=LUAD_P, n_hidden=10):
    train_set_x,  discrete_observed, survival_time, observed, test_data = load_data(dataset, step=7.0)
    # compute number of minibatches for training, validation and testing
    input_shape = train_set_x.shape[1]
    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print '... building the model'

    rng = numpy.random.RandomState(1234)
    x = T.matrix('x')
    o = T.vectors('o')

    # construct the MLP class
    classifier = MLP(
        rng=rng,
        input=x,
        n_in=input_shape,
        n_hidden=n_hidden,
        n_out=1
    )

    # start-snippet-4
    # the cost we minimize during training is the negative log likelihood of
    # the model plus the regularization terms (L1 and L2); cost is expressed
    # here symbolically
    cost = (
        classifier.negative_log_likelihood(o)
        + L1_reg * classifier.L1
        + L2_reg * classifier.L2_sqr
    )
    # start-snippet-5
    # compute the gradient of cost with respect to theta (sotred in params)
    # the resulting gradients will be stored in a list gparams
    gparams = [(T.grad(cost, param)) for param in classifier.params]

    # specify how to update the parameters of the model as a list of
    # (variable, update expression) pairs

    # given two lists of the same length, A = [a1, a2, a3, a4] and
    # B = [b1, b2, b3, b4], zip generates a list C of same size, where each
    # element is a pair formed from the two lists :
    #    C = [(a1, b1), (a2, b2), (a3, b3), (a4, b4)]
    updates = [
        (param, param - learning_rate * gparam)
        for param, gparam in zip(classifier.params, gparams)
    ]
    # compiling a Theano function `train_model` that returns the cost, but
    # in the same time updates the parameter of the model based on the rules
    # defined in `updates`
    index = T.iscalar()
    train_model = theano.function(
        on_unused_input='ignore',
        inputs=[index],
        outputs=cost,
        updates=updates,
        givens={
            x: train_set_x,
            o: discrete_observed
        }
    )

    output_fn = theano.function(
        on_unused_input='ignore',
        inputs=[index],
        outputs=classifier.outputLayer.hazard_ratio,
        givens={
            x: test_data,
            o: observed
        }
    )
    # end-snippet-5

    ###############
    # TRAIN MODEL #
    ###############
    print '... training'
    c = []
    epoch = 0
    while epoch < n_epochs:
        epoch += 1
        avg_cost = train_model(epoch)
        # learning_rate *= 0.95
        hazard_rate = output_fn(epoch)
        c_index = _naive_concordance_index(survival_time, hazard_rate, observed)
        c.append(c_index)
        print 'at epoch %d, cost is %f, test c_index is %f' % (epoch, avg_cost, c_index)
    print 'best score is: %f' % max(c)
    plt.ylim(0.2, 0.8)
    plt.plot(range(len(c)), c, c='r', marker='o', lw=2, ms=4, mfc='c')
    plt.show()
    plt.savefig('FFANNci.png', format='png')
    plt.savefig('FFANNci.eps', format='eps')
if __name__ == '__main__':
    main()
