# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 11:35:38 2015
@author: syouse3
"""
import os
from train import test_SdA
from loadMatData import read_pickle
def wrapper():
    pathout = '/home/syouse3/git/Survivalnet/SAE/results/LUAD-Oct26/Layers/'
    
    #test_SdA(finetune_lr=0.01, pretrain=True, pretraining_epochs=50, n_layers=3, n_hidden=120, coxphfit=True,
    #         pretrain_lr=0.5, training_epochs=600, pretrain_mini_batch=False, batch_size=100, augment=False,
    #         drop_out=True, pretrain_dropout=False, dropout_rate=0.5, grad_check=False, plot=False):
    if not os.path.exists(pathout):
        os.makedirs(pathout)
        
    layers = [2, 6, 8, 10, 12, 14, 16]
    hSizes = [60]
#    do_rates = [.7, .5, .3, .1, 0]
    do_rates = [0]
    pt_ops = [True]
    
    observed, X, survival_time, at_risk_X = read_pickle(name='LUAD_P.pickle')
    K = 5;
    m = len(X)

    foldnum = 1
    
    for hSize in hSizes:
        for n_layer in layers:
            for pretrain in pt_ops:
                for do_rate in do_rates:
                    test_SdA(observed = observed, X = X, survival_time = survival_time, at_risk_X = at_risk_X, foldnum = foldnum, K = K, finetune_lr=0.1, pretrain=pretrain, pretraining_epochs=400, n_layers=n_layer,\
                    n_hidden = hSize, coxphfit=True, pretrain_lr=2.0, training_epochs=400, pretrain_mini_batch=True, batch_size=8, \
                    augment = False, drop_out = False, pretrain_dropout=False, dropout_rate= do_rate,grad_check=False, \
                    plot=False, resultPath = pathout)            
if __name__ == '__main__':
    wrapper()
