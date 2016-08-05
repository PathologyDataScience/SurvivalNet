"""
Created on Tue Oct 20 12:53:54 2015
@author: syouse3
"""
import sys
import os
sys.path.append('./../')
from train import train
import numpy as np
import cPickle
import scipy.io as sio
from .SurvivalAnalysis import SurvivalAnalysis
import theano
import shutil
def pickSubType(subtypesVec, subtype):
    inds = [i for i in range(len(subtypesVec)) if (subtypesVec[i] == subtype)]
    return inds
def panorg_cost_func(params):
    ## PARSET 
    path_br = '/home/syouse3/OVBRCA_impute/imputed/BRCA_Integ.mat'
    path_ov = '/home/syouse3/OVBRCA_impute/imputed/OV_Integ.mat'
    opt = 'GDLS'    
    pretrain_config = None         #No pre-training 
    pretrain_set = None         #No pre-training 
    avg_cost = 0.0 
    D_br = sio.loadmat(path_br)
    D_ov = sio.loadmat(path_ov)
    T_ov = np.asarray([t[0] for t in D_ov['Integ_Survival']])
    O_ov = 1 - np.asarray([c[0] for c in D_ov['Integ_Censored']])
    ## PARSET 
    X_ov = D_ov['Integ_X']
    i = int(params[4])
    T_br = np.asarray([t[0] for t in D_br['Integ_Survival']])
    O_br = 1 - np.asarray([c[0] for c in D_br['Integ_Censored']])
    ## PARSET 
    X_br = D_br['Integ_X']
    prng = np.random.RandomState(i)
    order = prng.permutation(np.arange(len(X_br)))
    X_br = X_br[order]
    O_br = O_br[order]
    T_br = T_br[order]
            
    #foldsize denotes th amount of data used for testing. The same amount 
    #of data is used for model selection. The rest is used for training.
    fold_size = int(20 * len(X_br) / 100)
    
    train_set = {}
    test_set = {}

    #caclulate the risk group for every patient i: patients who die after i
    sa = SurvivalAnalysis()
    X_train = np.concatenate((X_ov, X_br[2*fold_size:]), axis = 0)
    O_train = np.concatenate((O_ov, O_br[2*fold_size:]), axis = 0)
    T_train = np.concatenate((T_ov, T_br[2*fold_size:]), axis = 0)
    #X_train = X_br[2*fold_size:]
    #O_train = O_br[2*fold_size:]
    #T_train = T_br[2*fold_size:]
    train_set['X'], train_set['T'], train_set['O'], train_set['A'] = sa.calc_at_risk(X_train, T_train, O_train);
    test_set['X'], test_set['T'], test_set['O'], test_set['A'] = sa.calc_at_risk(X_br[fold_size:fold_size*2], T_br[fold_size:fold_size*2], O_br[fold_size:fold_size*2]);
    n_layers = int(params[0])
    n_hidden = int(params[1])
    do_rate = params[2]
    #reg1 = params[3]
    #reg2 = params[4]
    if params[3] > .5:
        nonlin = theano.tensor.nnet.relu
    else: nonlin = np.tanh
    finetune_config = {'ft_lr':0.001, 'ft_epochs':40}
    #Print experiment identifier         
    expID = 'nl' + str(n_layers) + '-' + 'hs' + str(n_hidden) + '-' + 'dor' + str(do_rate) + str(nonlin) +'reg' + str(params[3]) +'-id' + str(int(i))       
    #print expID 
    _, _, test_cost_list, cindex_test,model, maxIter = train(pretrain_set, train_set, test_set,
         pretrain_config, finetune_config, n_layers, n_hidden, coxphfit=False,
           dropout_rate=do_rate, non_lin=nonlin, optim = "GDLS", disp = False, earlystp = False)
    if np.isnan(test_cost_list[-1]):
         print 'Skipping due to NAN'
         return 1 
    print expID, cindex_test[maxIter], 'at iter:', maxIter
    return (1 - cindex_test[maxIter])

def cost_func(params):
    ## PARSET 
    path = os.path.join(os.getcwd(), '../data/LUSC_Gene.mat')
    pretrain_config = None         #No pre-training 
    i = params[4]
    X = sio.loadmat(path)
    T = np.asarray([t[0] for t in X['Survival']])
    O = 1 - np.asarray([c[0] for c in X['Censored']])
    ## PARSET 
    X = X['Gene_X']
    prng = np.random.RandomState(int(i))
    order = prng.permutation(np.arange(len(X)))
    X = X[order]
    O = O[order]
    T = T[order]
                    
    #foldsize denotes th amount of data used for testing. The same amount 
    #of data is used for model selection. The rest is used for training.
    fold_size = int(20 * len(X) / 100)
    
    train_set = {}
    test_set = {}

    #caclulate the risk group for every patient i: patients who die after i
    sa = SurvivalAnalysis()    
    train_set['X'], train_set['T'], train_set['O'], train_set['A'] = sa.calc_at_risk(X[2*fold_size:], T[2*fold_size:], O[2*fold_size:]);
    test_set['X'], test_set['T'], test_set['O'], test_set['A'] = sa.calc_at_risk(X[fold_size:fold_size*2], T[fold_size:fold_size*2], O[fold_size:fold_size*2]);
    #test_set['X'], test_set['T'], test_set['O'], test_set['A'] = sa.calc_at_risk(X[:fold_size], T[:fold_size], O[:fold_size]);
    
    n_layers = int(params[0])
    n_hidden = int(params[1])
    do_rate = params[2]
    #reg1 = params[3]
    #reg2 = params[4]
    pretrain_set = X
    pretrain_config = None         #No pre-training 
    if params[3] > .5:
        nonlin = theano.tensor.nnet.relu
    else: nonlin = np.tanh
    finetune_config = {'ft_lr':0.001, 'ft_epochs':40}
        #Print experiment identifier         
    expID = 'nl' + str(n_layers) + '-' + 'hs' + str(n_hidden) + '-' + 'dor' + str(do_rate) + str(nonlin) +'-id' + str(int(i))       
    print expID 
    _, _, test_cost_list, cindex_test,_, maxIter = train(pretrain_set, train_set, test_set,
         pretrain_config, finetune_config, n_layers, n_hidden, coxphfit=False,
           dropout_rate=do_rate, non_lin=nonlin, optim = "GDLS", disp = False, earlystp = False)
    if np.isnan(test_cost_list[-1]):
        print 'Skipping due to NAN'
        return 1 
    #print expID, cindex_test[maxIter], 'at iter:', maxIter
    return (1 - cindex_test[maxIter])

def aggr_st_cost_func(params):
    #where the data (possibly multiple cross validation sets) are stored
    #we use 10 permutations of the data and consequently 10 different training 
    #and testing splits to produce the results in the paper
    numberOfShuffles = 20
    path = os.path.join(os.getcwd(), '../data/Brain_Sub_Data.mat')
    resultPath = os.path.join(os.getcwd(), '../results/Astro_AggSub_Integ/')
    final = False
    if final:
	if os.path.exists(resultPath):
            shutil.rmtree(resultPath)
            os.makedirs(resultPath)
        else:
            os.makedirs(resultPath)

    D = sio.loadmat(path)
    T = np.asarray([t[0] for t in D['Sub_Survival']])
    O = 1 - np.asarray([c[0] for c in D['Sub_Censored']])
    X = D['Sub_Integ_X']
    ST = D['Sub_Types']

    inds = pickSubType(ST, 'IDHwt')
    X_sub = X[inds,:]
    T_sub = T[inds]
    O_sub = O[inds]

    rinds = np.ones(len(X), np.bool)
    rinds[inds] = 0
    X = X[rinds,:]
    T = T[rinds]
    O = O[rinds]
    
    
    
    shuffleResults = []
    avg_cost = 0.0 
    ft = np.multiply(np.ones((numberOfShuffles, 1)), 40)
    i = 0
    while i < numberOfShuffles: 
        prng = np.random.RandomState(i)
   	order = prng.permutation(np.arange(len(X_sub)))
        X_sub = X_sub[order]
        O_sub = O_sub[order]
        T_sub = T_sub[order]
        #Use the entire dataset for pretraining
        pretrain_set = X
                
        #foldsize denotes th amount of data used for testing. The same amount 
        #of data is used for model selection. The rest is used for training.
        fold_size = int(40 * len(X_sub) / 100)
        
	X_train = np.concatenate((X, X_sub[2*fold_size:]), axis = 0)
	T_train = np.concatenate((T, T_sub[2*fold_size:]), axis = 0)
	O_train = np.concatenate((O, O_sub[2*fold_size:]), axis = 0)

        train_set = {}
        test_set = {}
	
        sa = SurvivalAnalysis()    
        train_set['X'], train_set['T'], train_set['O'], train_set['A'] = sa.calc_at_risk(X_train, T_train, O_train);
        test_set['X'] = X_sub[fold_size:fold_size*2]
        test_set['T'] = T_sub[fold_size:fold_size*2]
        test_set['O'] = O_sub[fold_size:fold_size*2]
        #test_set['X'] = X[:fold_size]
        #test_set['T'] = T[:fold_size]
        #test_set['O'] = O[:fold_size]
       
        print train_set['X'].shape, test_set['X'].shape

	test_set['X'], test_set['T'], test_set['O'], test_set['A'] = sa.calc_at_risk(test_set['X'], test_set['T'], test_set['O']);
        
    	## PARSET 
    	finetune_config = {'ft_lr':0.01, 'ft_epochs':ft[i]}
    	#pretrain_config = {'pt_lr':0.01, 'pt_epochs':50, 'pt_batchsize':None,'corruption_level':.0}

    	n_layers = int(params[0])
    	n_hidden = int(params[1])
    	do_rate = params[2]
 	pretrain_config = None         #No pre-training 
        #non_lin = theano.tensor.nnet.relu
	non_lin = theano.tensor.tanh

        #Print experiment identifier         
        expID = 'nl' + str(n_layers) + '-' + 'hs' + str(n_hidden) + '-' + 'dor' + str(do_rate) + '-id' + str(i)       
        print expID
 
   	train_cost_list, cindex_train, test_cost_list, cindex_test, model, maxIter = train(pretrain_set, train_set, test_set,
             pretrain_config, finetune_config, n_layers, n_hidden, coxphfit=False,
             dropout_rate=do_rate, non_lin=theano.tensor.nnet.relu, optim = "GDLS", disp = False, earlystp = False)
	#if np.isnan(test_cost_list[-1]):
	#     print 'Skipping due to NAN'
	#     return 1
	jj = np.max(np.where(~np.isnan(cindex_test)), axis=1)
    	avg_cost += cindex_test[jj]
	shuffleResults.append(cindex_test[jj])
         ## write output to file
        #outputFileName = os.path.join(resultPath, expID  + 'ci_tst')
        #f = file(outputFileName, 'wb')
        #cPickle.dump(cindex_test, f, protocol=cPickle.HIGHEST_PROTOCOL)
        #f.close()
        
        #outputFileName = os.path.join(resultPath, expID  + 'ci_trn')
        #f = file(outputFileName, 'wb')
        #cPickle.dump(cindex_train, f, protocol=cPickle.HIGHEST_PROTOCOL)
        #f.close()
        
        #outputFileName = os.path.join(resultPath , expID  + 'lpl_trn')
        #f = file(outputFileName, 'wb')
        #cPickle.dump(train_cost_list, f, protocol=cPickle.HIGHEST_PROTOCOL)
        #f.close()
    
        #outputFileName = os.path.join(resultPath, expID  + 'lpl_tst')
        #f = file(outputFileName, 'wb')
        #cPickle.dump(test_cost_list, f, protocol=cPickle.HIGHEST_PROTOCOL)
        #f.close()
        
        #outputFileName = os.path.join(resultPath, expID  + 'final_model')
        #f = file(outputFileName, 'wb')
        #cPickle.dump(model, f, protocol=cPickle.HIGHEST_PROTOCOL)
        #f.close()
	i = i + 1
 

    outputFileName = resultPath  + 'shuffle_cis'
    if final: sio.savemat(outputFileName, {'cis':shuffleResults})#, f, protocol=cPickle.HIGHEST_PROTOCOL)
    print np.mean(shuffleResults), np.std(shuffleResults)
    return (1 - avg_cost/numberOfShuffles)

def st_cost_func(params):
    #where the data (possibly multiple cross validation sets) are stored
    #we use 10 permutations of the data and consequently 10 different training 
    #and testing splits to produce the results in the paper
    numberOfShuffles = 1
    path = os.path.join(os.getcwd(), '../data/Brain_Sub_Data.mat')
    resultPath = os.path.join(os.getcwd(), '../results/final/models/IDHwt_Integ/')
    final = True
    if final is True:
	if os.path.exists(resultPath):
            shutil.rmtree(resultPath)
            os.makedirs(resultPath)
        else:
            os.makedirs(resultPath)

    D = sio.loadmat(path)
    avg_cost = 0.0 
    i = 0
    shuffleResults = []
    while i < numberOfShuffles: 
        inds = pickSubType(D['Sub_Types'], 'IDHwt')
    	prng = np.random.RandomState(i)
        T = np.asarray([t[0] for t in D['Sub_Survival'][inds]])
        O = 1 - np.asarray([c[0] for c in D['Sub_Censored'][inds]])
        X = D['Sub_Integ_X'][inds]
    	order = prng.permutation(np.arange(len(X)))
        X = X[order]
        #C is censoring status. 0 means alive patient. We change it to O 
        #for comatibility with lifelines package        
        O = O[order]
        T = T[order]
        #Use the entire dataset for pretraining
        pretrain_set = X
                
        #foldsize denotes th amount of data used for testing. The same amount 
        #of data is used for model selection. The rest is used for training.
        fold_size = int(25 * len(X) / 100)
        
        train_set = {}
        test_set = {}

        #caclulate the risk group for every patient i: patients who die after i
        sa = SurvivalAnalysis()
       # train_set['X'], train_set['T'], train_set['O'], train_set['A'] = sa.calc_at_risk(X[2*fold_size:], T[2*fold_size:], O[2*fold_size:]);
        test_set['X'], test_set['T'], test_set['O'], test_set['A'] = sa.calc_at_risk(X[fold_size:2*fold_size], T[fold_size:2*fold_size], O[fold_size:2*fold_size]);
        train_set['X'], train_set['T'], train_set['O'], train_set['A'] = sa.calc_at_risk(X, T, O);
       # if final: train_set['X'], train_set['T'], train_set['O'], train_set['A'] = sa.calc_at_risk(X[1*fold_size:], T[1*fold_size:], O[1*fold_size:]);
       # if final: test_set['X'], test_set['T'], test_set['O'], test_set['A'] = sa.calc_at_risk(X[:1*fold_size], T[:1*fold_size], O[:1*fold_size]);
        
        print train_set['X'].shape, test_set['X'].shape
        
    	## PARSET 
    	finetune_config = {'ft_lr':0.001, 'ft_epochs':40}
    	#pretrain_config = {'pt_lr':0.01, 'pt_epochs':50, 'pt_batchsize':None,'corruption_level':.0}

    	n_layers = int(params[0])
    	n_hidden = int(params[1])
    	do_rate = params[2]
 	pretrain_config = None         #No pre-training 
        non_lin = theano.tensor.nnet.relu
	#non_lin = theano.tensor.tanh

        #Print experiment identifier         
        expID = 'nl' + str(n_layers) + '-' + 'hs' + str(n_hidden) + '-' + 'dor' + str(do_rate) + '-id' + str(i)       
       
 
   	train_cost_list, cindex_train, test_cost_list, cindex_test, model, _ = train(pretrain_set, train_set, test_set,
             pretrain_config, finetune_config, n_layers, n_hidden, coxphfit=False,
             dropout_rate=do_rate, non_lin=theano.tensor.nnet.relu, optim = "GDLS", disp = True, earlystp = False)
    	if np.isnan(test_cost_list[-1]):
	     print 'Skipping due to NAN'
	     return 1 
        avg_cost += cindex_test[-1]
        print expID, cindex_test[-1]
	shuffleResults.append(cindex_test[-1])
    	i = i + 1
         ## write output to file
    #    outputFileName = os.path.join(resultPath, expID  + 'ci_tst')
    #    f = file(outputFileName, 'wb')
    #    cPickle.dump(cindex_test, f, protocol=cPickle.HIGHEST_PROTOCOL)
    #    f.close()
        
    #    outputFileName = os.path.join(resultPath, expID  + 'ci_trn')
    #    f = file(outputFileName, 'wb')
    #    cPickle.dump(cindex_train, f, protocol=cPickle.HIGHEST_PROTOCOL)
    #    f.close()
        
    #    outputFileName = os.path.join(resultPath , expID  + 'lpl_trn')
    #    f = file(outputFileName, 'wb')
    #    cPickle.dump(train_cost_list, f, protocol=cPickle.HIGHEST_PROTOCOL)
    #    f.close()
    
    #    outputFileName = os.path.join(resultPath, expID  + 'lpl_tst')
    #    f = file(outputFileName, 'wb')
    #    cPickle.dump(test_cost_list, f, protocol=cPickle.HIGHEST_PROTOCOL)
    #    f.close()
        
        outputFileName = os.path.join(resultPath, expID  + 'final_model')
        f = file(outputFileName, 'wb')
        cPickle.dump(model, f, protocol=cPickle.HIGHEST_PROTOCOL)
        f.close()
    #outputFileName = resultPath  + 'shuffle_cis'
    #if final is True: sio.savemat(outputFileName, {'cis':shuffleResults})#, f, protocol=cPickle.HIGHEST_PROTOCOL)
    print np.mean(shuffleResults), np.std(shuffleResults)
    return (1 - avg_cost/numberOfShuffles)
if __name__ == '__main__':
        res = panorg_cost_func([1,10,.9,1,0])
	print res
