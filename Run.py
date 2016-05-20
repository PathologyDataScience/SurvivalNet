"""
Created on Tue Oct 20 12:53:54 2015
@author: syouse3
"""
import sys
import os
sys.path.insert(0, '..')
from train import train
import numpy as np
import scipy.io as sio
from SurvivalAnalysis import SurvivalAnalysis
import theano
def bayesopt_costfunc(params):
    #where the data (possibly multiple cross validation sets) are stored
    #we use 10 permutations of the data and consequently 10 different training 
    #and testing splits to produce the results in the paper
    pin = os.path.join(os.getcwd(), 'data/Glioma/shuffles/')
    numberOfShuffles = len(os.listdir(pin))        
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
    avg_cost = 0.0 
    for i in range(numberOfShuffles): 
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
        train_set['X'], train_set['T'], train_set['O'], train_set['A'] = sa.calc_at_risk(X[fold_size*2:], T[fold_size*2:], O[fold_size*2:]);
        test_set['X'], test_set['T'], test_set['O'], test_set['A'] = sa.calc_at_risk(X[fold_size:fold_size*2], T[fold_size:fold_size*2], O[fold_size:fold_size*2]);
        
        
        
    	## PARSET 
    	finetune_config = {'ft_lr':0.01, 'ft_epochs':50}
    	#pretrain_config = {'pt_lr':0.01, 'pt_epochs':50, 'pt_batchsize':None,'corruption_level':.0}

    	n_layers = int(params[0])
    	n_hidden = int(params[1])
    	do_rate = params[2]
 	pretrain_config = None         #No pre-training 
        non_lin = theano.tensor.nnet.relu
	#non_lin = theano.tensor.tanh

        #Print experiment identifier         
        expID = 'nl' + str(n_layers) + '-' + 'hs' + str(n_hidden) + '-' + 'dor' + str(do_rate) + '-id' + str(i)       
        print expID
 
   	_, _, _, cindex_test,_ = train(pretrain_set, train_set, test_set,
             pretrain_config, finetune_config, n_layers, n_hidden, coxphfit=False,
             dropout_rate=do_rate, non_lin=theano.tensor.nnet.relu, optim = "GDLS", disp = True)
    
    	avg_cost += cindex_test[-1]
    return (1 - avg_cost/numberOfShuffles)

if __name__ == '__main__':
    bayesopt_costfunc([1, 20, 3])
