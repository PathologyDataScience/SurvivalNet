__author__ = 'Song'

import os
import sys
import timeit
import copy
from SdA import SdA
import numpy
from lifelines.utils import _naive_concordance_index
import matplotlib.pyplot as plt
import matlab.engine
import theano
import cPickle

def test_SdA(observed, X, survival_time, at_risk_X,  K = 5, foldnum = 1, finetune_lr=0.0001, pretrain=True, pretraining_epochs=50, n_layers=10, n_hidden=140, coxphfit=True,
             pretrain_lr=0.5, training_epochs=300, pretrain_mini_batch=True, batch_size=100, augment=False,
             drop_out=True, pretrain_dropout=False, dropout_rate=0.5, grad_check=False, plot=False, resultPath = ''):
    # observed, X, survival_time, at_risk_X = load_data('C:/Users/Song/Research/biomed/Survival/trainingData.csv')
    expID = 'pt' + str(pretrain) + 'ftlr' + str(finetune_lr) + '-' + 'pt' + str(pretraining_epochs) + '-' + 'nl' + str(n_layers) + '-' + 'hs' + str(n_hidden) + '-' + \
    'ptlr' + str(pretrain_lr) + '-' + 'ft' + str(training_epochs) + '-' + 'bs' + str(batch_size) + '-' +  \
    'dor'+ str(dropout_rate) + '-' + 'do'+ str(drop_out) + str(pretrain_dropout) + '-fold' + str(foldnum)      
    print expID    
    finetune_lr = theano.shared(numpy.asarray(finetune_lr, dtype=theano.config.floatX))
    learning_rate_decay = .989    
    
    
    if augment:
        train_X, train_y, train_observed, at_risk_X, test_X, test_y, test_observed = load_augment_data()
    else:
#        observed, X, survival_time, at_risk_X = load_data()
        test_size = len(X) / K
        train_X = X[test_size:]
        train_y = survival_time[test_size:]
        train_observed = observed[test_size:]
        test_observed = observed[:test_size]
        test_X = X[:test_size]
        test_y = survival_time[:test_size]
    n_ins = train_X.shape[1]
    n_train_batches = len(train_X) / batch_size if pretrain_mini_batch else 1
    # changed to theano shared variable in order to do minibatch
    train_X = theano.shared(value=train_X, name='train_X')
    # numpy random generator
    # start-snippet-3
    numpy_rng = numpy.random.RandomState(89677)
    print '... building the model'
    # construct the stacked denoising autoencoder class
    sda = SdA(
        numpy_rng=numpy_rng,
        n_ins=n_ins,
        hidden_layers_sizes=[n_hidden] * n_layers,
        n_outs=1,
        drop_out=drop_out,
        pretrain_dropout=pretrain_dropout,
        dropout_rate=dropout_rate,
        at_risk=at_risk_X
    )
    if grad_check:
        sda_plus = SdA(
            numpy_rng=numpy_rng,
            n_ins=n_ins,
            hidden_layers_sizes=[140] * n_layers,
            n_outs=1,
            at_risk=at_risk_X
        )

        sda_minus = SdA(
            numpy_rng=numpy_rng,
            n_ins=n_ins,
            hidden_layers_sizes=[140] * n_layers,
            n_outs=1,
            at_risk=at_risk_X
        )

    # end-snippet-3 start-snippet-4
    #########################
    # PRETRAINING THE MODEL #
    #########################
    if pretrain:
        print '... getting the pretraining functions'
        pretraining_fns = sda.pretraining_functions(train_set_x=train_X,
                                                    batch_size=batch_size,
                                                    pretrain_mini_batch=pretrain_mini_batch)

        print '... pre-training the model'
        start_time = timeit.default_timer()
        # Pre-train layer-wise
        # de-noising level, set to zero for now
        corruption_levels = [.0] * n_layers
        for i in xrange(sda.n_layers):
            # go through pretraining epochs
            for epoch in xrange(pretraining_epochs):
                # go through the training set
                c = []
                for batch_index in xrange(n_train_batches):
                    c.append(pretraining_fns[i](index=batch_index,
                             corruption=corruption_levels[i],
                             lr=pretrain_lr))
                             
                #print 'Pre-training layer %i, epoch %d, cost ' % (i, epoch),
                #print numpy.mean(c)

        end_time = timeit.default_timer()

        print >> sys.stderr, ('The pretraining code for file ' +
                              os.path.split(__file__)[1] +
                              ' ran for %.2fm' % ((end_time - start_time) / 60.))
        if drop_out:
            sda.reset_weight_by_rate(drop_out)
    # end-snippet-4
    ########################
    # FINETUNING THE MODEL #
    ########################

    # get the training, validation and testing function for the model
    print '... getting the finetuning functions'
    train_fn, output_fn, grad_fn, last_out_fn = sda.build_finetune_functions(
        train_X=train_X,
        train_observed=train_observed,
        test_X=test_X,
        learning_rate=finetune_lr
    )
    if grad_check:
        train_fn_plus, output_fn_plus, grad_fn_plus, last_out_fn_plus = sda_plus.build_finetune_functions(
            train_X=train_X,
            train_observed=train_observed,
            test_X=test_X,
            learning_rate=finetune_lr
        )
        train_fn_minus, output_fn_minus, grad_fn_minus, last_out_fn_minus = sda_minus.build_finetune_functions(
            train_X=train_X,
            train_observed=train_observed,
            test_X=test_X,
            learning_rate=finetune_lr
        )

    # cox initialization
    if coxphfit:
        last_out = last_out_fn(0)
        eng = matlab.engine.start_matlab()
        cox_x = matlab.double(last_out.tolist())
        cox_y = matlab.double(train_y.tolist())
        cox_c = matlab.double((1 - train_observed).tolist())
        b = eng.coxphfit(cox_x, cox_y, 'censoring', cox_c)
        b = numpy.asarray([[w[0] for w in b]]).T
        sda.logLayer.reset_weight(b)
        # print numpy.dot(last_out, b)

    print '... finetunning the model'
    # early-stopping parameters
 #   new_lr = theano.printing.Print('new_lr')(finetune_lr)    
    c = []
    cost_list = []
    epoch = 0
    while epoch < training_epochs:
        epoch += 1
        print epoch                
        avg_cost = train_fn(epoch)
        test_harzard = output_fn(epoch)
        if grad_check:
            grad = grad_fn(epoch)
            parameter = [param.get_value() for param in sda.params]
            gradient_check(grad, parameter, sda_plus, sda_minus, train_fn_plus, train_fn_minus)
        c_index = _naive_concordance_index(test_y, test_harzard, test_observed)
        c.append(c_index)
        cost_list.append(avg_cost)
        print 'at epoch %d, cost is %f, test c_index is %f' % (epoch, avg_cost, c_index)
        decay_learning_rate = theano.function(inputs=[], outputs=finetune_lr, \
        updates={finetune_lr: finetune_lr * learning_rate_decay})    
        new_lr = decay_learning_rate()
        #print finetune_lr.eval()
    print 'best score is: %f' % max(c)
    if plot:
        plt.plot(range(len(c)), c, c='r', marker='o', lw=5, ms=10, mfc='c')
        plt.show()
        plt.plot(range(len(cost_list)), cost_list, c='r', marker='o', lw=5, ms=10, mfc='c')
        plt.show()
        
    outputFileName = resultPath + expID  + 'ci'
    f = file(outputFileName, 'wb')
    cPickle.dump(c, f, protocol=cPickle.HIGHEST_PROTOCOL)
    f.close()
    
    outputFileName = resultPath + expID  + 'lpl'
    f = file(outputFileName, 'wb')
    cPickle.dump(cost_list, f, protocol=cPickle.HIGHEST_PROTOCOL)
    f.close()
    return cost_list, c


def gradient_check(grad, parameter, sda_plus, sda_minus, train_fn_plus, train_fn_minus, e=0.01**2, epoch=1):
    for l in xrange(len(parameter)):
        param = parameter[l]
        if len(param.shape) == 1:
            for i in xrange(len(param)):
                # dimention level
                param_plus = copy.copy(param)
                param_plus[i] += e
                param_minus = copy.copy(param)
                param_minus[i] -= e
                # layer level
                parameter_plus = copy.copy(parameter)
                parameter_plus[l] = parameter_plus
                parameter_minus = copy.copy(parameter)
                parameter_minus[l] = parameter_minus
                # reset weight
                sda_plus.reset_weight(parameter_plus)
                sda_minus.reset_weight(parameter_minus)
                # get reset cost
                cost_plus = train_fn_plus(epoch)
                cost_minus = train_fn_minus(epoch)
                appro_grad = (cost_plus - cost_minus) / 2 * e
                diff = grad[l][i] - appro_grad
                print diff
        if len(param.shape) == 2:
            for i in xrange(len(param)):
                for j in xrange(len(param[i])):
                    # dimention level
                    param_plus = copy.copy(param)
                    param_plus[i][j] += e
                    param_minus = copy.copy(param)
                    param_minus[i][j] -= e
                    # layer level
                    parameter_plus = copy.copy(parameter)
                    parameter_plus[l] = param_plus
                    parameter_minus = copy.copy(parameter)
                    parameter_minus[l] = param_minus
                    # reset weight
                    sda_plus.reset_weight(parameter_plus)
                    sda_minus.reset_weight(parameter_minus)
                    # get reset cost
                    cost_plus = train_fn_plus(epoch)
                    cost_minus = train_fn_minus(epoch)
                    appro_grad = (cost_plus - cost_minus) / (2 * e)
                    diff = grad[l][i][j] - appro_grad
                    print diff / grad[l][i][j] * 100

if __name__ == '__main__':
    pretrain = True
    markers = ['o', '*', '^', '.', 'v']
    colors = ['r', 'b', 'g', 'm', 'c']
    labels = ['.7', '.5', '.3', '.1', '.0', 'No']
    do_rates = [.7, .5, .3, .1, 0]
    # do_rates = [0.0]
    for i in xrange(len(do_rates)):
        cost_list, c = test_SdA(dropout_rate=do_rates[i], pretrain=pretrain, coxphfit=True, n_layers = 10, n_hidden = 60)
        plt.plot(range(len(cost_list)), cost_list, c=colors[i], marker=markers[i], lw=5, ms=10, mfc=colors[i],
                 label=labels[i])
    cost_list, c = test_SdA(drop_out=False, pretrain=pretrain, coxphfit=False)
    plt.plot(range(len(cost_list)), cost_list, c='k', marker='s', lw=5, ms=10, mfc='k', label='No')
    plt.legend(loc=4, fontsize='x-large')
    plt.show()
