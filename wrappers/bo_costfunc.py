"""
Created on Tue Oct 20 12:53:54 2015
@author: syouse3
"""
import sys
import os
sys.path.insert(0, '..')
sys.path.insert(0, '../data')
from train import test_SdA
import finalExperiment 
import numpy as np
import scipy.io as sio
from nonLinearities import ReLU, LeakyReLU, Sigmoid

def bo_costfunc(params):
    ## PARSET
    p = os.path.join(os.getcwd(), '../data/Brain_P/shuffle' + str(int(np.floor(params[5]))) + '.mat' )           
    mat = sio.loadmat(p)
    X = mat['X']
    X = X.astype('float64')   
    C = mat['C'].transpose()
    T = mat['T'].transpose()
        
    T = np.asarray([t[0] for t in T])
    O = 1 - np.asarray([c[0] for c in C], dtype='int32')
    fold_size = int(15 * len(X) / 100)  
      
    X_val = X[fold_size:2*fold_size]
    T_val = T[fold_size:2*fold_size]
    O_val = O[fold_size:2*fold_size]
    
    X_train = X[fold_size * 2:]       
    T_train = T[fold_size * 2:]
    O_train = O[fold_size * 2:]
      
    n_layer = int(params[0])
    hSize = int(params[1])
    ptlr = params[2] 
    ftlr = params[3]
    do_rate = params[4]
    
    ## PARSET 
    ptepochs=50
    tepochs=50
    bs = 1
    dropout = True
    pretrain = True
    
    x_train, t_train, o_train, at_risk_train = finalExperiment.calc_at_risk(X_train, T_train, O_train);
    x_test, t_test, o_test, at_risk_test = finalExperiment.calc_at_risk(X_val, T_val, O_val);
    
   
    cost_list, tst_cost_list, c = test_SdA(train_observed = o_train, train_X = x_train, train_y = t_train, at_risk_train = at_risk_train, \
     at_risk_test=at_risk_test, test_observed = o_test, test_X = x_test, test_y = t_test,\
     finetune_lr=ftlr, pretrain=pretrain, pretraining_epochs = ptepochs, n_layers=n_layer, n_hidden = hSize,\
     pretrain_lr=ptlr, training_epochs = tepochs , batch_size=bs, drop_out = dropout, dropout_rate= do_rate, \
     non_lin=Sigmoid, alpha=5.5)

    cost = c[-1]
    return cost
    
if __name__ == '__main__':
    bo_costfunc([2, 10, .1, .1, .3, 1.61585888e-01])