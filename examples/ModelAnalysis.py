import pickle
import scipy.io as sio 
import survivalnet as sn
import os

# #############################################################################
# Integrated models ###########################################################
# #############################################################################
# define model/dataset pairs
ModelPaths = ['./models/']
Models = ['nl1-hs10-dor0.62-id0final_model']
Data = ['./data/Brain_Integ.mat']

# load datasets and perform feature analysis
for i, Path in enumerate(ModelPaths):

	# load normalized data
	X = scipy.io.loadmat(Data[i])

	# extract relevant values
	Samples = X['Patients']
	Normalized = X['Integ_X']
	Raw = X['Integ_X_raw']
	Symbols = X['Integ_Symbs']
	Survival = X['Survival']
	Censored = X['Censored']

	# load model
	f = open(Path + Models[i], 'rb')
	Model = pickle.load(f)
	f.close()

	sn.analysis.FeatureAnalysis(Model, Normalized, Raw, Symbols,
								Survival, Censored,
								NBox=11, NScatter=11, NKM=11,
								NCluster=Raw.shape[1],
								Tau=5e-2, Path=Path)
