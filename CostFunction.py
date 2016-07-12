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
from SurvivalAnalysis import SurvivalAnalysis
import theano
import shutil
def pickSubType(subtypesVec, subtype):
    inds = [i for i in range(len(subtypesVec)) if (subtypesVec[i] == subtype)]
    return inds

def cost_func(params):
    #where the data (possibly multiple cross validation sets) are stored
    #we use 10 permutations of the data and consequently 10 different training 
    #and testing splits to produce the results in the paper
    numberOfShuffles = 20
    ## PARSET 
    path = os.path.join(os.getcwd(), '../data/BRCA_Integ.mat')
#    X = (X - np.mean(X, axis = 0)) / np.std(X, axis = 0)
    # Use Bayesian Optimization for model selection, 
    #if false ,manually set parameters will be used
    BayesOpt = False
    opt = 'GDLS'    
    #pretrain_config = {'pt_lr':0.01, 'pt_epochs':100, 'pt_batchsize':None,'corruption_level':.0}
    #ft = [15,15,15,15,67,43,15,15,25,51,15,15,31,27,17,15,22,26,45]
    pretrain_config = None         #No pre-training 
    non_lin = theano.tensor.nnet.relu
    avg_cost = 0.0 

    for i in range(numberOfShuffles): 
	X = sio.loadmat(path)
    	T = np.asarray([t[0] for t in X['Survival']])
    	O = 1 - np.asarray([c[0] for c in X['Censored']])
    	## PARSET 
	X = X['Integ_X']
	prng = np.random.RandomState()
    	order = prng.permutation(np.arange(len(X)))
        X = X[order]
        #C is censoring status. 0 means alive patient. We change it to O 
        #for comatibility with lifelines package        
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

        finetune_config = {'ft_lr':0.001, 'ft_epochs':100}
        #Print experiment identifier         
        expID = 'nl' + str(n_layers) + '-' + 'hs' + str(n_hidden) + '-' + 'dor' + str(do_rate) + str(nonlin) +'reg' + str(params[3]) +'-id' + str(i)       
	print expID 
   	_, _, test_cost_list, cindex_test,_, maxIter = train(pretrain_set, train_set, test_set,
             pretrain_config, finetune_config, n_layers, n_hidden, coxphfit=False,
       	       dropout_rate=do_rate, non_lin=nonlin, optim = "GDLS", disp = False, reg1 = params[4], reg2 = params[5])
        if np.isnan(test_cost_list[-1]):
	     print 'Skipping due to NAN'
	     return 1 
        #print expID, cindex_test[maxIter], 'at iter:', maxIter
    	avg_cost += cindex_test[maxIter]
    return (1 - avg_cost/numberOfShuffles)
def aggr_st_cost_func(params):
    #where the data (possibly multiple cross validation sets) are stored
    #we use 10 permutations of the data and consequently 10 different training 
    #and testing splits to produce the results in the paper
    numberOfShuffles = 100
    path = os.path.join(os.getcwd(), '../data/Brain_Sub_Data.mat')
    resultPath = os.path.join(os.getcwd(), '../results/Oligo_aggr_sub_Integ/')
    if os.path.exists(resultPath):
        shutil.rmtree(resultPath)
        os.makedirs(resultPath)
    else:
        os.makedirs(resultPath)

    D = sio.loadmat(path)
    avg_cost = 0.0 
    ft = np.multiply(np.ones((numberOfShuffles, 1)), 40)
    i = 0
    prng = np.random.RandomState()
    while i < numberOfShuffles: 
        T = np.asarray([t[0] for t in D['Sub_Survival']])
        O = 1 - np.asarray([c[0] for c in D['Sub_Censored']])
        X = D['Sub_Integ_X']
        X = (X - np.min(X, axis = 0))/(np.max(X, axis = 0) - np.min(X, axis = 0))
    	order = prng.permutation(np.arange(len(X)))
	ST = D['Sub_Types'][order]
        X = X[order]
        #C is censoring status. 0 means alive patient. We change it to O 
        #for comatibility with lifelines package        
        O = O[order]
        T = T[order]
        #Use the entire dataset for pretraining
        pretrain_set = X
                
        #foldsize denotes th amount of data used for testing. The same amount 
        #of data is used for model selection. The rest is used for training.
        fold_size = int(40 * len(X) / 100)
        
        train_set = {}
        test_set = {}

        #caclulate the risk group for every patient i: patients who die after i
        sa = SurvivalAnalysis()    
	#sizeST = len(pickSubType(ST[2*fold_size:], 'IDHmut-non-codel'))
	#print sizeST
        train_set['X'], train_set['T'], train_set['O'], train_set['A'] = sa.calc_at_risk(X[2*fold_size:], T[2*fold_size:], O[2*fold_size:]);
        inds = pickSubType(ST[:fold_size], 'IDHmut-codel')
	print len(inds)
	inds = pickSubType(ST[fold_size:fold_size*2], 'IDHmut-codel')
	#inds = inds[:28]
        test_set['X'] = X[fold_size:fold_size*2]
        test_set['T'] = T[fold_size:fold_size*2]
        test_set['O'] = O[fold_size:fold_size*2]
        #test_set['X'] = X[:fold_size]
        #test_set['T'] = T[:fold_size]
        #test_set['O'] = O[:fold_size]
        test_set['X'] = test_set['X'][inds]
        test_set['T'] = test_set['T'][inds]
        test_set['O'] = test_set['O'][inds]
       

	test_set['X'], test_set['T'], test_set['O'], test_set['A'] = sa.calc_at_risk(test_set['X'], test_set['T'], test_set['O']);
        
        print 'Observed:', sum(test_set['O']), sum(train_set['O']) 
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
 
   	train_cost_list, cindex_train, test_cost_list, cindex_test, model, _ = train(pretrain_set, train_set, test_set,
             pretrain_config, finetune_config, n_layers, n_hidden, coxphfit=False,
             dropout_rate=do_rate, non_lin=theano.tensor.nnet.relu, optim = "GDLS", disp = False, earlystp = False)
	if np.isnan(test_cost_list[-1]):
	     print 'Skipping due to NAN'
	     return 1
    	avg_cost += cindex_test[maxIter]
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
 

    return (1 - avg_cost/numberOfShuffles)

def st_cost_func(params):
    #where the data (possibly multiple cross validation sets) are stored
    #we use 10 permutations of the data and consequently 10 different training 
    #and testing splits to produce the results in the paper
    numberOfShuffles = 100
    path = os.path.join(os.getcwd(), '../data/Brain_Sub_Data.mat')
    path = os.path.join(os.getcwd(), '../data/Brain_Sub_Data.mat')
    resultPath = os.path.join(os.getcwd(), '../results/Oligo_sub_sub_integ/')
    if os.path.exists(resultPath):
        shutil.rmtree(resultPath)
        os.makedirs(resultPath)
    else:
        os.makedirs(resultPath)

    D = sio.loadmat(path)
    avg_cost = 0.0 
    prng = np.random.RandomState()
    i = 0
    while i < numberOfShuffles: 
        inds = pickSubType(D['Sub_Types'], 'IDHmut-codel')
        T = np.asarray([t[0] for t in D['Sub_Survival'][inds]])
        O = 1 - np.asarray([c[0] for c in D['Sub_Censored'][inds]])
        X = D['Sub_Integ_X'][inds]
    	order = prng.permutation(np.arange(len(X)))
        X = X[order]
	print X.shape
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
        train_set['X'], train_set['T'], train_set['O'], train_set['A'] = sa.calc_at_risk(X[2*fold_size:], T[2*fold_size:], O[2*fold_size:]);
        #train_set['X'], train_set['T'], train_set['O'], train_set['A'] = sa.calc_at_risk(X, T, O);
        test_set['X'], test_set['T'], test_set['O'], test_set['A'] = sa.calc_at_risk(X[:fold_size], T[:fold_size], O[:fold_size]);
        
    	print train_set['X'].shape    
        
    	## PARSET 
    	finetune_config = {'ft_lr':0.001, 'ft_epochs':ft[i]}
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
 
   	train_cost_list, cindex_train, test_cost_list, cindex_test, model, _ = train(pretrain_set, train_set, test_set,
             pretrain_config, finetune_config, n_layers, n_hidden, coxphfit=False,
             dropout_rate=do_rate, non_lin=theano.tensor.nnet.relu, optim = "GDLS", disp = True, earlystp = False)
    	if np.isnan(test_cost_list[-1]):
	    i = i - 1
	    print 'i = ', i
	    continue
        avg_cost += cindex_test[-1]
	print cindex_test[-1]
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
	i = i + 1
    return (1 - avg_cost/numberOfShuffles)
if __name__ == '__main__':
        res = cost_func([1.83,50,0.891067,0,0.1, .1])
	print res
