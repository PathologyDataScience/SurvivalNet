# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 11:35:38 2015
@author: syouse3
"""
import os
from train import test_SdA
from loadMatData import load_data
import math
import numpy as np
def wrapper():
    pathout = '/home/syouse3/git/Survivalnet/SAE/results/LUAD-Oct28/DropOutFractionsKFCV/'
    
    #test_SdA(finetune_lr=0.01, pretrain=True, pretraining_epochs=50, n_layers=3, n_hidden=120, coxphfit=True,
    #         pretrain_lr=0.5, training_epochs=600, pretrain_mini_batch=False, batch_size=100, augment=False,
    #         drop_out=True, pretrain_dropout=False, dropout_rate=0.5, grad_check=False, plot=False):
    if not os.path.exists(pathout):
        os.makedirs(pathout)
        
    layers = [22]
    hSizes = [200]
    do_rates = [0, .1, .3, .5, .7]
    pt_ops = [True]    
    
    observed, X, survival_time, at_risk_X = load_data()
    K = 5;
    m = len(X)
    F = math.floor(m/K)
    cursor = -1
    foldnum = 1
    while cursor < F * K:
        starti = cursor + 1
        if m - cursor + 1 < K:
            break
        else:
            endi = cursor + F
        print starti
        print endi
   
 #       if foldnum == 1:
#            X = np.concatenate((X[starti:endi+ 1], X[endi + 1:]), axis=0)
#            observed = np.concatenate((observed[starti:endi + 1], observed[endi + 1:]), axis=0)
#            survival_time = np.concatenate((survival_time[starti:endi + 1], survival_time[endi + 1:]), axis=0)            
#            at_risk_X = np.concatenate((at_risk_X[starti:endi + 1], at_risk_X[endi + 1:]), axis=0)            

        if foldnum > 1:
            X = np.concatenate((X[starti:endi + 1], X[:starti], X[endi + 1:]), axis=0)
            print X.shape
            observed = np.concatenate((observed[starti:endi + 1], observed[:starti],observed[endi + 1:]), axis=0)   
            survival_time = np.concatenate((survival_time[starti:endi + 1], survival_time[:starti], survival_time[endi + 1:]), axis=0)            
            at_risk_X = np.concatenate((at_risk_X[starti:endi + 1], at_risk_X[:starti], at_risk_X[endi + 1:]), axis=0)            

        for hSize in hSizes:
            for n_layer in layers:
                for pretrain in pt_ops:
                    for do_rate in do_rates:
                        test_SdA(observed = observed, X = X, survival_time = survival_time, at_risk_X = at_risk_X, foldnum = foldnum, K = K, finetune_lr=0.01, pretrain=pretrain, pretraining_epochs=400, n_layers=n_layer,\
                        n_hidden = hSize, coxphfit=False, pretrain_lr=2.0, training_epochs=400, pretrain_mini_batch=True, batch_size=8, \
                        augment = False, drop_out = True, pretrain_dropout=False, dropout_rate= do_rate,grad_check=False, \
                        plot=False, resultPath = pathout) 
        cursor = cursor + F
        foldnum = foldnum + 1
if __name__ == '__main__':
    wrapper()
