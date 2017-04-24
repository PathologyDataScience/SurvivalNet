import os
from survivalnet.train import train
import numpy as np
import scipy.io as sio
from survivalnet.optimization import SurvivalAnalysis
import theano
import shutil
import cPickle

def cost_func(params):
	n_layers = int(params[0])
	n_hidden = int(params[1])
	do_rate = params[2]
	if params[3] > .5:
		nonlin = theano.tensor.nnet.relu
	else: 
		nonlin = np.tanh
	lambda1 = params[4]
	lambda2 = params[5]


	#load data sets
	f = open('train_set', 'rb')
	train_set = cPickle.load(f)
	f.close()
	f = open('val_set', 'rb')
	val_set = cPickle.load(f)
	f.close()

	pretrain_config = None         #No pre-training 
	pretrain_set = None

	finetune_config = {'ft_lr':0.001, 'ft_epochs':40}
	#Print experiment identifier         
	expID = 'nl' + str(n_layers) + '-' + 'hs' + str(n_hidden) + '-' + 'dor' + str(do_rate) + str(nonlin) 
	print expID 
	_, _, val_cost_list, cindex_val, _, _, _, maxIter = train(pretrain_set, train_set, val_set,
			pretrain_config, finetune_config,
			n_layers, n_hidden, dropout_rate=do_rate, lambda1= lambda1, lambda2 = lambda2,  non_lin=nonlin, 
			optim = "GDLS", verbose = False, earlystp = False)
	if not val_cost_list or np.isnan(val_cost_list[-1]):
		print 'Skipping due to NAN'
		return 1 
	return (1 - cindex_val[maxIter])

if __name__ == '__main__':
	res = cost_func([1.0,38.0,0,0.1, 0.1, 0])
	print res
