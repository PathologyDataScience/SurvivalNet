# -*- coding: utf-8 -*-
"""
Created on Sat Apr  2 17:59:16 2016

@author: Ayine
"""
import os
import scipy.io as sio
from SurvivalAnalysis import SurvivalAnalysis
import Bayesian_Optimization
import cPickle
import numpy as np
from train import train
import theano
def Run():      
    #where c-index and cost function values are saved 
    resultPath = os.path.join(os.getcwd(), 'results/Brain_P_results/relu/Apr7/')
    if not os.path.exists(resultPath):
        os.makedirs(resultPath)
    #where the data (possibly multiple cross validation sets) are stored
    #we use 10 permutations of the data and consequently 10 different training 
    #and testing splits to produce the results in the paper
    pin = os.path.join(os.getcwd(), 'data/Brain_PC/')
    numberOfShuffles = len([name for name in os.listdir(pin)])        
    
    # Use Bayesian Optimization for model selection, 
    #if false ,manually set parameters will be used
    BayesOpt = False
        
    for i in range(numberOfShuffles): 
        #file names: shuffle0.mat, etc.
        order = pin + 'shuffle' + str(i) + '.mat'            
        p = pin + 'Brain_PC.mat'            
        order = sio.loadmat(order)
        order = order['order']
        order = np.asarray([i[0] for i in order.transpose()])

        mat = sio.loadmat(p)
        X = mat['X'][order]

        #C is censoring status. 0 means alive patient. We change it to O 
        #for comatibility with lifelines package        
        C = mat['C'][order]
        T = mat['T'][order]
        T = np.asarray([t[0] for t in T])
        O = 1 - np.asarray([c[0] for c in C])
        
        #Use the whole dataset fotr pretraining
        pretrain_set = X
                
        #foldsize denotes th amount of data used for testing. The same amount 
        #of data is used for model selection. The rest is used for training.
        fold_size = int(15 * len(X) / 100)
        
        train_set = {}
        test_set = {}

        #caclulate the risk group for every patient i: patients who die after i
        sa = SurvivalAnalysis()    
        train_set['X'], train_set['T'], train_set['O'], train_set['A'] = sa.calc_at_risk(X[fold_size * 2:], T[fold_size * 2:], O[fold_size * 2:]);
        test_set['X'], test_set['T'], test_set['O'], test_set['A'] = sa.calc_at_risk(X[:fold_size], T[:fold_size], O[:fold_size]);
        
        
        finetune_config = {'ft_lr':0.0001, 'ft_epochs':100}
        pretrain_config = {'pt_lr':0.01, 'pt_epochs':50, 'pt_batchsize':None,'corruption_level':.0}
       #pretrain_config = None         #No pre-training 
        n_layers = 3
        n_hidden = 100
        do_rate = 0
        non_lin = theano.tensor.nnet.relu

        if BayesOpt == True:
            maxval, bo_params, err = Bayesian_Optimization.tune(i, non_lin)
            finetune_config = {'ft_lr':bo_params[3], 'ft_epochs':100}
            pretrain_config = {'pt_lr':bo_params[2], 'pt_epochs':50, 'pt_batchsize':None,'corruption_level':.0}
           #pretrain_config = None         #No pre-training 
            n_layers = bo_params[0]
            n_hidden = bo_params[1]
            do_rate = bo_params[4]

        train_cost_list, cindex_train, test_cost_list, cindex_test = train(pretrain_set, train_set, test_set,
                 pretrain_config, finetune_config, n_layers, n_hidden, coxphfit=False,
                 dropout_rate=do_rate, non_lin = non_lin)
        

        #Save results in the desired folder         
        expID = 'nl' + str(n_layers) + '-' + 'hs' + str(n_hidden) + '-' + \
        'dor'+ str(do_rate) + '-id' + str(i)       
    
        print expID
         
        ## write output to file
        outputFileName = os.path.join(resultPath, expID  + 'ci_tst')
        f = file(outputFileName, 'wb')
        cPickle.dump(cindex_test, f, protocol=cPickle.HIGHEST_PROTOCOL)
        f.close()
        
        outputFileName = os.path.join(resultPath, expID  + 'ci_trn')
        f = file(outputFileName, 'wb')
        cPickle.dump(cindex_train, f, protocol=cPickle.HIGHEST_PROTOCOL)
        f.close()
        
        outputFileName = os.path.join(resultPath , expID  + 'lpl_trn')
        f = file(outputFileName, 'wb')
        cPickle.dump(train_cost_list, f, protocol=cPickle.HIGHEST_PROTOCOL)
        f.close()
    
        outputFileName = os.path.join(resultPath, expID  + 'lpl_tst')
        f = file(outputFileName, 'wb')
        cPickle.dump(test_cost_list, f, protocol=cPickle.HIGHEST_PROTOCOL)
        f.close()
if __name__ == '__main__':
    Run()
