# -*- coding: utf-8 -*-
"""
Created on Sat Apr  2 17:59:16 2016

@author: Safoora Yousefi
"""
import os
import scipy.io as sio
from SurvivalAnalysis import SurvivalAnalysis
import Bayesian_Optimization
import cPickle
import numpy as np
from train import train
import theano
import shutil
def pickSubType(subtypesVec, subtype):
    inds = [i for i in range(len(subtypesVec)) if (subtypesVec[i] == subtype)]
    return inds
def Run():      
    #where c-index and cost function values are saved 
    resultPath = os.path.join(os.getcwd(), 'results/Brain_P_results/relu/BO_disc/GDLS')
    if os.path.exists(resultPath):
        shutil.rmtree(resultPath)
        os.makedirs(resultPath)
    else:
        os.makedirs(resultPath)

    #where the data (possibly multiple cross validation sets) are stored
    #we use 10 permutations of the data and consequently 10 different training 
    #and testing splits to produce the results in the paper
    pin = os.path.join(os.getcwd(), 'data/Glioma/shuffles/')
    numberOfShuffles = len([name for name in os.listdir(pin)])        
    c = os.path.join(os.getcwd(), 'data/Glioma/Brain_C.mat')
    p = os.path.join(os.getcwd(), 'data/Glioma/Brain_P.mat')

    Brain_C = sio.loadmat(c)
    Brain_P = sio.loadmat(p)

    T = np.asarray([t[0] for t in Brain_C['Survival']])
    O = 1 - np.asarray([c[0] for c in Brain_C['Censored']])
    X = Brain_P['Expression']

    #concat = np.zeros((len(X), 2))
    #inds = pickSubType(Brain_C['Subtype'], 'IDHmut-non-codel')
    #concat[inds] = [1,0]
    #inds = pickSubType(Brain_C['Subtype'], 'IDHmut-codel')
    #concat[inds] = [1,1]
    #X = np.concatenate((concat, X), 1)
    
    # Use Bayesian Optimization for model selection, 
    #if false ,manually set parameters will be used
    BayesOpt = True
    opt = 'GDLS'    
    finetune_config = {'ft_lr':0.001, 'ft_epochs':40}
    #pretrain_config = {'pt_lr':0.01, 'pt_epochs':100, 'pt_batchsize':None,'corruption_level':.0}
    pretrain_config = None         #No pre-training 
    n_layers = 2
    n_hidden = 33
    do_rate = 0.5
    non_lin = theano.tensor.nnet.relu

    if BayesOpt == True:
        maxval, bo_params, err = Bayesian_Optimization.tune(non_lin)
        n_layers = bo_params[0]
        n_hidden = bo_params[1]
        do_rate = bo_params[2]
    print "***selected model***"  
    expID = 'nl' + str(n_layers) + '-' + 'hs' + str(n_hidden) + '-' + \
    'dor'+ str(do_rate)       
    print expID
         
    for i in range(numberOfShuffles): 
        expID = 'nl' + str(n_layers) + '-' + 'hs' + str(n_hidden) + '-' + \
        'dor'+ str(do_rate) + '-id' + str(i)       
        print expID
        #file names: shuffle0.mat, etc.
        order = pin + 'shuffle' + str(i) + '.mat'            
        order = sio.loadmat(order)
        order = order['order']
        order = np.asarray([e[0] for e in order.transpose()])

        X = X[order]

        #C is censoring status. 0 means alive patient. We change it to O 
        #for comatibility with lifelines package        
        O = O[order]
        T = T[order]
        
        #Use the entire dataset for pretraining
        pretrain_set = X
                
        #foldsize denotes th amount of data used for testing. The same amount 
        #of data is used for model selection. The rest is used for training.
        fold_size = int(20 * len(X) / 100)
        
        train_set = {}
        test_set = {}

        #caclulate the risk group for every patient i: patients who die after i
        sa = SurvivalAnalysis()    
        train_set['X'], train_set['T'], train_set['O'], train_set['A'] = sa.calc_at_risk(X[fold_size * 2:], T[fold_size * 2:], O[fold_size * 2:]);
        test_set['X'], test_set['T'], test_set['O'], test_set['A'] = sa.calc_at_risk(X[:fold_size], T[:fold_size], O[:fold_size]);
        
        
        train_cost_list, cindex_train, test_cost_list, cindex_test, model = train(pretrain_set, train_set, test_set,
                 pretrain_config, finetune_config, n_layers, n_hidden, coxphfit=False,
                 dropout_rate=do_rate, non_lin = non_lin, optim = opt)
        

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
        
        outputFileName = os.path.join(resultPath, expID  + 'final_model')
        f = file(outputFileName, 'wb')
        cPickle.dump(model, f, protocol=cPickle.HIGHEST_PROTOCOL)
        f.close()
if __name__ == '__main__':
    Run()
