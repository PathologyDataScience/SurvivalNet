# -*- coding: utf-8 -*-
"""
Created on Sat Apr  2 17:59:16 2016

@author: Ayine
"""
import os
from sklearn.preprocessing import scale
import scipy.io as sio
from SurvivalAnalysis import SurvivalAnalysis
import cPickle
import numpy as np
from train import train
import theano
def Run():      
    #where c-index and cost function values are saved 
    resultPath = os.path.join(os.getcwd(), 'results/Brain_P_results/relu/Apr4/')
    
    #where the data (possibly multiple cross validation sets) are stored
    #we use 10 permutations of the data and consequently 10 different training 
    #and testing splits to produce the results in the paper
    pin = os.path.join(os.getcwd(), '../survivalnet/data/Brain_P/')
    numberOfShuffles = len([name for name in os.listdir(pin)])        
    for i in range(numberOfShuffles): 
        #file names: shuffle0.mat, etc.
        p = pin + 'shuffle' + str(i) + '.mat'            
        mat = sio.loadmat(p)
        X = mat['X']
        X = scale(X.astype('float64'))

        #C is censoring status. 0 means alive patient. We change it to O 
        #for comatibility with lifelines package        
        C = mat['C']
        T = mat['T']
        T = np.asarray([t[0] for t in T.transpose()])
        O = 1 - np.asarray([c[0] for c in C.transpose()])
        
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
        train_cost_list, cindex_train, test_cost_list, cindex_test = train(pretrain_set, train_set, test_set,
                 pretrain_config, finetune_config, n_layers, n_hidden, coxphfit=False,
                 dropout_rate=do_rate, non_lin=theano.tensor.nnet.relu)
        

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