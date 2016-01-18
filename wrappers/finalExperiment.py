
"""
Created on Tue Oct 20 12:53:54 2015
@author: syouse3
"""


import sys
import os
sys.path.insert(0, '../')
from train import test_SdA
import bayesopt_wrapper
import numpy as np
import scipy.io as sio
from nonLinearities import ReLU, LeakyReLU, Sigmoid
import cPickle


def calc_at_risk(X, T, O):
    tmp = list(T)
    T = np.asarray(tmp).astype('float64')
    order = np.argsort(T)
    sorted_T = T[order]
    at_risk = np.asarray([list(sorted_T).index(x)+1 for x in sorted_T]).astype('int32')
    T = np.asarray(sorted_T)
    O = O[order]
    X = X[order]
    return X, T, O, at_risk - 1
    
def wrapper(i, path2data, path2output, validation = True):
    p = path2data + 'shuffle' + str(i) + '.mat'    
    print '*** shuffle #%d *** \n' % i                       
    mat = sio.loadmat(p)
    X = mat['X']
    X = X.astype('float64')       
    O = mat['C']
    T = mat['T']
    T = np.asarray([t[0] for t in T.transpose()])
    O = 1 - np.asarray([o[0] for o in O.transpose()])
    fold_size = int(15 * len(X) / 100)       
    
    X_test = X[:fold_size]
    T_test = T[:fold_size]
    O_test = O[:fold_size]
    
    X_train = X[fold_size * 2:]       
    T_train = T[fold_size * 2:]
    O_train = O[fold_size * 2:]
    if validation == True:
#        bb = 0
        maxval, params, err = bayesopt_wrapper.bayesopt_tuning(i)
        cost_func(O_train, X_train, T_train, O_test, X_test, T_test, i, resultPath = path2output, params = params)
        
    else: 
        cost_func(O_train, X_train, T_train, O_test, X_test, T_test, i, resultPath = path2output, params = None)


def cost_func(O_train, X_train, T_train, O_test, X_test, T_test, shuffle_id, resultPath, params = None):
        
    print '*** Model assesment using selected params ***'   
    #outputFileName = os.path.join(resultPath, 'modelSelect' + str(shuffle_id) + '.txt')
    #f = file(outputFileName, 'rb')        
    #cPickle.dump(params, f, protocol=cPickle.HIGHEST_PROTOCOL)
    #params = cPickle.load(f);   
    #f.close()
#    brainparams_sig = np.array([[2, 300, 0.05, 0.01, 0.1],\
#    [4,	275,	0.09,	0.01,	0.1],\
#    [3,	172,	0.1,	0.02,	0.1],\
#    [5,	96,	0.1,	0.04,	0],\
#    [3,	180,	0.01,	0.01,	0],\
#    [4,	299,	0.09,	0.001,	0],\
#    [3,	299,	0.09,	0.007,	0.1],\
#    [2,	299,	0.07,	0.001,	0],\
#    [4,	299,	0.03,	0.01,	0.1],\
#    [2,	195,	0.01,	0.03,	0.3]])
#    
#    params = brainparams_sig[shuffle_id]
#    
    if not os.path.exists(resultPath):
        os.makedirs(resultPath)
    if params == None:
        ## PARSET
        n_layer = 4
        hSize = 275
        do_rate = 0.1
        ftlr = .001
        ptlr = .001
    else:
        print params
        n_layer = params[0]
        hSize = params[1]
        ptlr = params[2] 
        ftlr = params[3]
        do_rate = params[4]
    ## PARSET    
    ptepochs=500
    tepochs=500
    bs = 1
    dropout = True
    pretrain = True
    nonlinearity = Sigmoid
    
    
    x_train, t_train, o_train, at_risk_train = calc_at_risk(X_train, T_train, O_train);
    x_test, t_test, o_test, at_risk_test = calc_at_risk(X_test, T_test, O_test);
    
   
    cost_list, tst_cost_list, c = test_SdA(train_observed = o_train, train_X = x_train, train_y = t_train, at_risk_train = at_risk_train, \
     at_risk_test=at_risk_test, test_observed = o_test, test_X = x_test, test_y = t_test,\
     finetune_lr=ftlr, pretrain=pretrain, pretraining_epochs = ptepochs, n_layers=n_layer, n_hidden = hSize,\
     pretrain_lr=ptlr, training_epochs = tepochs , batch_size=bs, drop_out = dropout, dropout_rate= do_rate, \
     non_lin=nonlinearity, alpha=5.5)
    
    expID = 'pt' + str(pretrain) + 'ftlr' + str(ftlr) + '-' + 'pt' + str(ptepochs) + '-' + \
    'nl' + str(n_layer) + '-' + 'hs' + str(hSize) + '-' + \
    'ptlr' + str(ptlr) + '-' + 'ft' + str(tepochs) + '-' + 'bs' + str(bs) + '-' +  \
    'dor'+ str(do_rate) + '-' + 'do'+ str(dropout) + '-id' + str(shuffle_id)       

    print expID
     
    ## write output to file
    outputFileName = os.path.join(resultPath, expID  + 'ci')
    f = file(outputFileName, 'wb')
    cPickle.dump(c, f, protocol=cPickle.HIGHEST_PROTOCOL)
    f.close()
    
    outputFileName = os.path.join(resultPath , expID  + 'lpl')
    f = file(outputFileName, 'wb')
    cPickle.dump(cost_list, f, protocol=cPickle.HIGHEST_PROTOCOL)
    f.close()

    outputFileName = os.path.join(resultPath, expID  + 'lpl_tst')
    f = file(outputFileName, 'wb')
    cPickle.dump(tst_cost_list, f, protocol=cPickle.HIGHEST_PROTOCOL)
    f.close()


    return 
if __name__ == '__main__':
    ## PARSET
    pout = os.path.join(os.getcwd(), '../results/Brain_P_results/sigmoid/Dec/iter500/')
    pin = os.path.join(os.getcwd(), '../data/Brain_P/')
    maxShuffIter = 10
    for i in range(maxShuffIter):
    	wrapper(i, pin, pout, False)



                        
