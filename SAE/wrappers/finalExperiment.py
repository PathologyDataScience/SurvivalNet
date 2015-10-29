
"""
Created on Tue Oct 20 12:53:54 2015

@author: syouse3
"""
import os
from train import test_SdA
import math
from loadMatData import load_data
import numpy as np

def wrapper():
    pathout = '/home/syouse3/git/Survivalnet/SAE/results/LUAD-Oct28/FinalExperiment6l100hs/'
    
    #test_SdA(finetune_lr=0.01, pretrain=True, pretraining_epochs=50, n_layers=3, n_hidden=120, coxphfit=True,
    #         pretrain_lr=0.5, training_epochs=600, pretrain_mini_batch=False, batch_size=100, augment=False,
    #         drop_out=True, pretrain_dropout=False, dropout_rate=0.5, grad_check=False, plot=False):
    if not os.path.exists(pathout):
        os.makedirs(pathout)
        
    layers = [6]
    hSizes = [10]
    do_rates = [.1]
    pt_ops = [True]
    oo, xx, ss, aa = load_data(p='data/LUAD_P.mat')
    k = 5;
    m = len(xx)
    F = math.floor(m/k)
    print F
    cursor = -1
    foldn = 1
    print xx.shape
    print oo.shape
    print aa.shape
    while cursor < F * k:
        #print cursor
        #print F*k
        starti = cursor + 1
        if m - cursor <= k:
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
 
        if foldn > 1:
            x = np.concatenate((xx[starti:endi + 1], xx[:starti], xx[endi + 1:]), axis=0)
            print x.shape
            o = np.concatenate((oo[starti:endi + 1], oo[:starti],oo[endi + 1:]), axis=0)   
            #print o.shape             
            s = np.concatenate((ss[starti:endi + 1], ss[:starti], ss[endi + 1:]), axis=0)            
            #print s.shape                        
            a = np.concatenate((aa[starti:endi + 1], aa[:starti], aa[endi + 1:]), axis=0)
            print a.shape            
        else:
            x = xx
            o = oo
            a = aa
            s = ss
        for hSize in hSizes:
            for n_layer in layers:
                for pretrain in pt_ops:
                    for do_rate in do_rates:
                        test_SdA(observed = o, X = x, survival_time = s, at_risk_X = a, K = k, foldnum = foldn, finetune_lr=0.01, pretrain=pretrain, pretraining_epochs=400, n_layers=n_layer,\
                        n_hidden = hSize, coxphfit=False, pretrain_lr=2.0, training_epochs=400, pretrain_mini_batch=True, batch_size=8, \
                        augment = False, drop_out = True, pretrain_dropout=False, dropout_rate= do_rate,grad_check=False, \
                        plot=False, resultPath = pathout) 
        cursor = cursor + F
        foldn = foldn + 1
                    
if __name__ == '__main__':
    wrapper()
