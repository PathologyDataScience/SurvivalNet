import survivalnet as sn
import sys
sys.path.append('./..')
import os
import scipy.io as sio
from time import clock
from survivalnet.optimization import SurvivalAnalysis
import Bayesian_Optimization as BayesOpt
import cPickle
import numpy as np
from survivalnet.train import train
import theano
import shutil

def Run():      
  panorg = False
  #where c-index and cost function values are saved 
  resultPath = os.path.join(os.getcwd(), 'results/final/BRCA_Gene/')
  if os.path.exists(resultPath):
      shutil.rmtree(resultPath)
      os.makedirs(resultPath)
  else:
      os.makedirs(resultPath)
  path_br = 'data/panorgan/BRCA_Gene.mat'
  path_ov = 'data/panorgan/OV_Gene.mat'
  opt = 'GDLS'    
  pretrain_config = None         #No pre-training 
  pretrain_set = None         #No pre-training 
  avg_cost = 0.0 
  D_br = sio.loadmat(path_br)
  D_ov = sio.loadmat(path_ov)
  T_ov = np.asarray([t[0] for t in D_ov['Gene_Survival']])
  O_ov = 1 - np.asarray([c[0] for c in D_ov['Gene_Censored']])
  ## PARSET 
  X_ov = D_ov['Gene_X']
  T_br = np.asarray([t[0] for t in D_br['Gene_Survival']])
  O_br = 1 - np.asarray([c[0] for c in D_br['Gene_Censored']])
  ## PARSET 
  X_br = D_br['Gene_X']
  
  # Use Bayesian Optimization for model selection, 
  #if false ,manually set parameters will be used
  doBayesOpt = True
  opt = 'GDLS'    
  #pretrain_config = {'pt_lr':0.01, 'pt_epochs':1000, 'pt_batchsize':None,'corruption_level':.3}
  pretrain_config = None         #No pre-training 
  numberOfShuffles = 20
  ft = np.multiply(np.ones((numberOfShuffles, 1)), 40)
  shuffleResults =[]
  avg_cost = 0
  i = 0 
  while i < numberOfShuffles: 
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
    if panorg:
      X_train = np.concatenate((X_ov, X_br[1*fold_size:]), axis = 0)
      O_train = np.concatenate((O_ov, O_br[1*fold_size:]), axis = 0)
      T_train = np.concatenate((T_ov, T_br[1*fold_size:]), axis = 0)
    else:
      X_train = X_br[1*fold_size:]
      O_train = O_br[1*fold_size:]
      T_train = T_br[1*fold_size:]
    train_set['X'], train_set['T'], train_set['O'], train_set['A'] = sa.calc_at_risk(X_train, T_train, O_train);
    test_set['X'], test_set['T'], test_set['O'], test_set['A'] = sa.calc_at_risk(X_br[:fold_size], T_br[:fold_size], O_br[:fold_size]);
 
    if doBayesOpt == True:
      print '***Model Selection with BayesOpt for shuffle', str(i), '***'
      maxval, bo_params, err = BayesOpt.tune(i)
      n_layers = bo_params[0]
      n_hidden = bo_params[1]
      do_rate = bo_params[2]
      nonlin = theano.tensor.nnet.relu if bo_params[3]>.5 else np.tanh
    else:
      n_layers = 1
      n_hidden = 67
      do_rate = .35
      #nonlin = theano.tensor.nnet.relu
      nonlin = np.tanh 

    expID = 'nl' + str(n_layers) + '-' + 'hs' + str(n_hidden) + '-' + \
            'dor'+ str(do_rate) + '-id' + str(i)       

    finetune_config = {'ft_lr':0.01, 'ft_epochs':ft[i]}
    print '***Model Assesment***'
    train_cost_list, cindex_train, test_cost_list, cindex_test, model, _ = train(pretrain_set, train_set, test_set,
    pretrain_config, finetune_config, n_layers, n_hidden, coxphfit=False,
    dropout_rate=do_rate, non_lin = nonlin, optim = opt, disp = True, earlystp = False )
    i = i + 1
    shuffleResults.append(cindex_test[-1])
    avg_cost += cindex_test[-1]
    print expID , ' ',   cindex_test[-1],  'average = ',avg_cost/i
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
  outputFileName = resultPath  + 'shuffle_cis'
  sio.savemat(outputFileName, {'cis':shuffleResults})#, f, protocol=cPickle.HIGHEST_PROTOCOL)
  print np.mean(shuffleResults), np.std(shuffleResults)
if __name__ == '__main__':
  Run()
