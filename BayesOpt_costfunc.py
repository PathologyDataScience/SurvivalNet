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
    ## PARSET
    p = os.path.join(os.getcwd(), 'data/Brain_P/shuffle' + str(int(np.floor(params[5]))) + '.mat' )           
    mat = sio.loadmat(p)
    X = mat['X']
    X = X.astype('float64')
    
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
    valid_set = {}
    
    sa = SurvivalAnalysis()    

    train_set['X'], train_set['T'], train_set['O'], train_set['A'] = \
    sa.calc_at_risk(X[fold_size * 2:], T[fold_size * 2:], O[fold_size * 2:]);
    
    valid_set['X'], valid_set['T'], valid_set['O'], valid_set['A'] = \
    sa.calc_at_risk(X[fold_size:2*fold_size], T[fold_size:2*fold_size], O[fold_size:2*fold_size]);

    ## PARSET 
    finetune_config = {'ft_lr':params[3], 'ft_epochs':50}
    pretrain_config = {'pt_lr':params[2], 'pt_epochs':50, 'pt_batchsize':None,'corruption_level':.0}
    n_layers = int(params[0])
    n_hidden = int(params[1])
    do_rate = params[4]
    _, _, _, cindex_test = train(pretrain_set, train_set, valid_set,
             pretrain_config, finetune_config, n_layers, n_hidden, coxphfit=False,
             dropout_rate=do_rate, non_lin=theano.tensor.nnet.relu)
    
    cost = cindex_test[-1]
    return (1 - cost)

if __name__ == '__main__':
    bayesopt_costfunc([3, 100, .01, .0001, 0, 0])
