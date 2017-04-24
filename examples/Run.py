import survivalnet as sn
import argparse
import sys
sys.path.append('./..')
import Bayesian_Optimization as BayesOpt
import os
import scipy.io as sio
from survivalnet.optimization import SurvivalAnalysis
import numpy as np
from survivalnet.train import train
import theano
import shutil
import cPickle

def Run(input_path, output_path):      
	if not os.path.exists(output_path):
		os.makedirs(output_path)

	D = sio.loadmat(input_path)
	T = np.asarray([t[0] for t in D['Survival']]).astype('float32')
	#C is censoring status where 1 means incomplete folow-up. We change it to Observed status where 1 means death. 
	O = 1 - np.asarray([c[0] for c in D['Censored']]).astype('int32')
	X = D['Integ_X']
	X = X.astype('float32')

	# Use Bayesian Optimization for model selection, 
	# if false, manually set parameters will be used
	doBayesOpt = True
	# Optimization algorithm.
	opt = 'GD'    
	#Pretraining settings
	#pretrain_config = {'pt_lr':0.01, 'pt_epochs':1000, 'pt_batchsize':None,'corruption_level':.3}
	pretrain_config = None         #No pre-training 

	#for the results in the paper we used 20 randomizations of training/validation/testing 
	numberOfShuffles = 20
	epochs = 40
	cindex_results =[]
	avg_cost = 0
	i = 0 
	while i < numberOfShuffles: 
		#set random generator seed for reproducibility
		prng = np.random.RandomState(i)
		order = prng.permutation(np.arange(len(X)))
		X = X[order]
		O = O[order]
		T = T[order]

		#Use the entire dataset for pretraining
		pretrain_set = X

		#foldsize denotes th amount of data used for testing. The same amount 
		#of data is used for model selection. The rest is used for training.
		fold_size = int(20 * len(X) / 100)

		train_set = {}
		test_set = {}
		val_set = {}
		#caclulate the risk group for every patient i: patients who die after i
		sa = SurvivalAnalysis()    
		finetune_config = {'ft_lr':0.0001, 'ft_epochs':epochs}
		train_set['X'], train_set['T'], train_set['O'], train_set['A'] = sa.calc_at_risk(X[2*fold_size:],
				T[2*fold_size:], 
				O[2*fold_size:]);
		test_set['X'], test_set['T'], test_set['O'], test_set['A'] = sa.calc_at_risk(X[:fold_size],
				T[:fold_size],
				O[:fold_size]);
		val_set['X'], val_set['T'], val_set['O'], val_set['A'] = sa.calc_at_risk(X[fold_size:2*fold_size],
				T[fold_size:2*fold_size],
				O[fold_size:2*fold_size]);
		#Write data sets for bayesopt
		f = file('train_set', 'wb')
		cPickle.dump(train_set, f, protocol=cPickle.HIGHEST_PROTOCOL)
		f.close()
		f = file('val_set', 'wb')
		cPickle.dump(val_set, f, protocol=cPickle.HIGHEST_PROTOCOL)
		f.close()

		if doBayesOpt == True:
			print '***Model Selection with BayesOpt for shuffle', str(i), '***'
			maxval, bo_params, err = BayesOpt.tune()
			n_layers = bo_params[0]
			n_hidden = bo_params[1]
			do_rate = bo_params[2]
			nonlin = theano.tensor.nnet.relu if bo_params[3]>.5 else np.tanh
			lambda1 = bo_params[4]
			lambda2 = bo_params[5]
		else:
			n_layers = 1
			n_hidden = 100
			do_rate = 0.5
			lambda1 = 0
			lambda2 = 0
			nonlin = np.tanh #nonlin = theano.tensor.nnet.relu

		expID = 'nl' + str(n_layers) + '-' + 'hs' + str(n_hidden) + '-' + \
				'dor'+ str(do_rate) + '-id' + str(i)       


		print '***Model Assesment***'
		train_cost_list, cindex_train, test_cost_list, cindex_test, model, _ = train(pretrain_set, train_set, test_set, val_set,
				pretrain_config, finetune_config,
				n_layers, n_hidden, dropout_rate=do_rate, lambda1=lambda1, lambda2=lambda2, non_lin = nonlin,
				optim = opt, verbose = True, earlystp = False)
		i = i + 1
		cindex_results.append(cindex_test[-1])
		avg_cost += cindex_test[-1]
		print expID , ' ',   cindex_test[-1],  'average = ',avg_cost/i
	outputFileName = output_path  + 'cis.mat'
	sio.savemat(outputFileName, {'cis':cindex_results})
	print np.mean(cindex_results), np.std(cindex_results)
if __name__ == '__main__':
	parser = argparse.ArgumentParser(prog='Run',
			formatter_class=argparse.ArgumentDefaultsHelpFormatter,
			description = 'Script to train survival net')
	parser.add_argument('-ip', '--input_path', dest='input_path',
			default='./data/Brain_Integ.mat',
			help='Path specifying location of dataset.')
	parser.add_argument('-sp', '--output_path', dest='output_path',
			default='./results',
			help='Path specifying where to save output files.')
	args = parser.parse_args()
	Run(args.input_path, args.output_path)
