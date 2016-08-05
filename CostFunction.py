import os
from survivalnet.train import train
import numpy as np
import scipy.io as sio
from survivalnet.optimization import SurvivalAnalysis
import theano
import shutil
def panorg_cost_func(params):
    panorgan = True
    n_layers = int(params[0])
    n_hidden = int(params[1])
    do_rate = params[2]
    if params[3] > .5:
        nonlin = theano.tensor.nnet.relu
    else: nonlin = np.tanh
    i = int(params[4])
    finetune_config = {'ft_lr':0.001, 'ft_epochs':40}
    #Print experiment identifier         
    expID = 'nl' + str(n_layers) + '-' + 'hs' + str(n_hidden) + '-' + 'dor' + str(do_rate) + str(nonlin) +'-id' + str(int(i))       
    parstr = 'nl' + str(params[0]) + '-' + 'hs' + str(params[1]) + '-' + 'dor' + str(params[2]) + str(params[3]) +'-id' + str(params[4])       
    print parstr 
   ## PARSET 
    path_br = 'data/panorgan/BRCA_Integ.mat'
    path_ov = 'data/panorgan/OV_Integ.mat'
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
    if panorgan:
        X_train = np.concatenate((X_ov, X_br[2*fold_size:]), axis = 0)
        O_train = np.concatenate((O_ov, O_br[2*fold_size:]), axis = 0)
        T_train = np.concatenate((T_ov, T_br[2*fold_size:]), axis = 0)
    else:
        X_train = X_br[2*fold_size:]
        O_train = O_br[2*fold_size:]
        T_train = T_br[2*fold_size:]

    train_set['X'], train_set['T'], train_set['O'], train_set['A'] = sa.calc_at_risk(X_train, T_train, O_train);
    test_set['X'], test_set['T'], test_set['O'], test_set['A'] = sa.calc_at_risk(X_br[fold_size:fold_size*2], T_br[fold_size:fold_size*2], O_br[fold_size:fold_size*2]);
    _, _, test_cost_list, cindex_test,model, maxIter = train(pretrain_set, train_set, test_set,
         pretrain_config, finetune_config, n_layers, n_hidden, coxphfit=False,
           dropout_rate=do_rate, non_lin=nonlin, optim = "GDLS", disp = False, earlystp = False)
    if not test_cost_list or np.isnan(test_cost_list[-1]):
         print 'Skipping due to NAN'
         return 1 
    return (1 - cindex_test[maxIter])

def cost_func(params):
    n_layers = int(params[0])
    n_hidden = int(params[1])
    do_rate = params[2]
    if params[3] > .5:
        nonlin = theano.tensor.nnet.relu
    else: nonlin = np.tanh
    i = int(params[4])
    ## PARSET 
    path = os.path.join(os.getcwd(), 'data/KIPAN_Gene.mat')
    pretrain_config = None         #No pre-training 
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
    pretrain_set = X
    pretrain_config = None         #No pre-training 
    finetune_config = {'ft_lr':0.001, 'ft_epochs':40}
        #Print experiment identifier         
    expID = 'nl' + str(n_layers) + '-' + 'hs' + str(n_hidden) + '-' + 'dor' + str(do_rate) + str(nonlin) +'-id' + str(int(i))       
    print expID 
    _, _, test_cost_list, cindex_test,_, maxIter = train(pretrain_set, train_set, test_set,
         pretrain_config, finetune_config, n_layers, n_hidden, coxphfit=False,
           dropout_rate=do_rate, non_lin=nonlin, optim = "GDLS", disp = False, earlystp = False)
    if not test_cost_list or np.isnan(test_cost_list[-1]):
        print 'Skipping due to NAN'
        return 1 
    return (1 - cindex_test[maxIter])

#def aggr_st_cost_func(params):

#def st_cost_func(params):

if __name__ == '__main__':
        res = cost_func([1.0,10.0,0.0,0.0,0.34])
	print res
