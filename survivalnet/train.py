import numpy
import os
import sys
import theano
import timeit

from model import Model
from optimization import BFGS
from optimization import GDLS
from optimization import SurvivalAnalysis 
from optimization import isOverfitting
import matplotlib.pyplot as plt


def train(pretrain_set, train_set, test_set, val_set,
             pretrain_config, finetune_config, n_layers=10, n_hidden=140, coxphfit=False,
             dropout_rate=0.5, non_lin=None, optim = 'GD', disp = True, earlystp = True):    
    finetune_lr = theano.shared(numpy.asarray(finetune_config['ft_lr'], dtype=theano.config.floatX))
    learning_rate_decay = .989    
        
    # changed to theano shared variable in order to do minibatch
    #train_set = theano.shared(value=train_set, name='train_set')
    
    # numpy random generator
    numpy_rng = numpy.random.RandomState()
    #if disp: print '... building the model'

    # construct the stacked denoising autoencoder and the corresponding regression network
    model = Model(
        numpy_rng = numpy_rng,
        n_ins = train_set['X'].shape[1],
        hidden_layers_sizes = [n_hidden] * n_layers,
        n_outs = 1,
        dropout_rate=dropout_rate,
        non_lin=non_lin)
        
    #########################
    # PRETRAINING THE MODEL #
    #########################
    if pretrain_config is not None:
        n_train_batches = len(train_set) / pretrain_config['pt_batchsize'] if pretrain_config['pt_batchsize'] else 1
            
        if disp: print '... getting the pretraining functions'
        pretraining_fns = model.pretraining_functions(pretrain_set,
                                                    pretrain_config['pt_batchsize'])
        if disp: print '... pre-training the model'
        start_time = timeit.default_timer()
        # de-noising level
        corruption_levels = [pretrain_config['corruption_level']] * n_layers
        for i in xrange(model.n_layers):            #Layerwise pre-training
            # go through pretraining epochs
            for epoch in xrange(pretrain_config['pt_epochs']):
                # go through the training set
                c = []
                for batch_index in xrange(n_train_batches):
                    c.append(pretraining_fns[i](index=batch_index,
                             corruption=corruption_levels[i],
                             lr=pretrain_config['pt_lr']))
                             
                if disp: print "Pre-training layer %i, epoch %d, cost" % (i, epoch),
                if disp: print numpy.mean(c)

        end_time = timeit.default_timer()
        
        if disp: print >> sys.stderr, ('The pretraining code for file ' +
                              os.path.split(__file__)[1] +
                              ' ran for %.2fm' % ((end_time - start_time) / 60.))

    ########################
    # FINETUNING THE MODEL #
    ########################

    #if disp: print '... getting the finetuning functions'
    forward, backward = model.build_finetune_functions(
        learning_rate=finetune_lr
    )

    #if disp: print '... finetunning the model'
    # early-stopping parameters
    cindex_train = []
    cindex_test = []
    cindex_val = []
    train_cost_list = []
    test_cost_list = []
    val_cost_list = []
    #gradient_sizes = []
    plt.figure(1)
    plt.ion()
    if optim == 'BFGS':        
        bfgs = BFGS(model, train_set['X'], train_set['O'], train_set['A'])
    elif optim == 'GDLS':
	gdls = GDLS(model, train_set['X'], train_set['O'], train_set['A'])
    survivalAnalysis = SurvivalAnalysis()    
    epoch = 0
    while epoch < finetune_config['ft_epochs']:
        #print epoch    
        train_cost, train_risk, train_features = forward(train_set['X'], train_set['O'], train_set['A'], 1)
	#print train_features.mean()
        if optim == 'BFGS':        
            bfgs.BFGS()	
            #gradient_sizes.append(numpy.linalg.norm(bfgs.gf_t))        
        elif optim == 'GDLS':        
            gdls.GDLS()	
        elif optim == 'GD':
            #idx = 0
            backward(train_set['X'], train_set['O'], train_set['A'], 1)
        	#gradients = []
        	#for i in range(len(grads)):
              #      gradients[idx:idx + grads[i].size] = grads[i].ravel()
              #      idx += grads[i].size
              #      gradient_sizes.append(numpy.linalg.norm(gradients))        
        train_c_index = survivalAnalysis.c_index(train_risk, train_set['T'], 1 - train_set['O'])
             
        test_cost, test_risk, test_features = forward(test_set['X'], test_set['O'], test_set['A'], 0)
        test_c_index = survivalAnalysis.c_index(test_risk, test_set['T'], 1 - test_set['O'])

        val_cost, val_risk, val_features = forward(val_set['X'], val_set['O'], val_set['A'], 0)
        val_c_index = survivalAnalysis.c_index(val_risk, val_set['T'], 1 - val_set['O'])
        
        cindex_train.append(train_c_index)
        cindex_test.append(test_c_index)
        cindex_val.append(val_c_index)

                
        train_cost_list.append(train_cost)
        test_cost_list.append(test_cost)
        val_cost_list.append(val_cost)
        if disp:
            x = numpy.arange(0,epoch+1)
            plt.clf()
            plt.subplot(1,2,1)        

            plt.plot(x, numpy.asarray(train_cost_list)/train_set['X'].shape[0], 'r--', x, numpy.asarray(test_cost_list)/test_set['X'].shape[0], 'b--', x, numpy.asarray(val_cost_list)/val_set['X'].shape[0], 'g--')
            plt.subplot(1,2,2)        
            plt.plot(x, numpy.asarray(cindex_train), 'r--', x, numpy.asarray(cindex_test), 'b--', x, numpy.asarray(cindex_val), 'g--')
            plt.pause(0.05)

        if disp: 
            print 'epoch = %d, trn_cost = %f, trn_ci = %f, tst_cost = %f, tst_ci = %f' % (epoch, train_cost, train_c_index, test_cost, test_c_index)
        if earlystp and epoch >= 15 and (epoch % 5 == 0):
            if disp: print "Checking overfitting!"
            check, maxIter = isOverfitting(numpy.asarray(cindex_test))
            if check:                
                print "Training Stopped Due to Overfitting! cindex = %f, MaxIter = %d" %(cindex_test[maxIter], maxIter)
                
                break
	else: maxIter = epoch
    	sys.stdout.flush()
        decay_learning_rate = theano.function(inputs=[], outputs=finetune_lr, \
        updates={finetune_lr: finetune_lr * learning_rate_decay})    
        decay_learning_rate()
        epoch += 1
    	if numpy.isnan(test_cost): break 
    if disp: print 'best score is: %f' % max(cindex_test)
    return train_cost_list, cindex_train, test_cost_list, cindex_test, model, maxIter
