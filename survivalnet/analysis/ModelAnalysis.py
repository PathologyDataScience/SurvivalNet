import pickle
import scipy.io as sio 
import survivalnet as sn
import os

# analysis of transfer learning gene expression (BRCA, BRCA+OV, BRCA+OV+UCEC)
# regular gene expression models (GBMLGG) and integrated (GBMLGG, BRCA) models

# #############################################################################
# transfer learning experiments ###############################################
# #############################################################################
"""
# define model/dataset pairs
ModelPaths = ['/Users/lcoop22/Drive/Work/Papers/SurvivalNet/Models/BRCA/BRCA_Gene/',
		 '/Users/lcoop22/Drive/Work/Papers/SurvivalNet/Models/BRCA+OV_Gene/',
		 '/Users/lcoop22/Drive/Work/Papers/SurvivalNet/Models/BRCA+OV+UCEC_Gene/']
Models = ['nl1-hs1000-dor0.4-id0final_model',
		  'nl2-hs1000-dor8e-05-id0final_model',
		  'nl3-hs999-dor0.0-id0final_model']
Data = ['/Users/lcoop22/Drive/Work/Papers/SurvivalNet/Data/SingleCancerDatasets/BRCA/BRCA_Gene.mat',
		'/Users/lcoop22/Drive/Work/Papers/SurvivalNet/Data/PanCancerDatasets/BRCA+OV/OVBRCA_Gene.mat',
		'/Users/lcoop22/Drive/Work/Papers/SurvivalNet/Data/PanCancerDatasets/BRCA+OV+UCEC/OVBRCAUCEC_Gene.mat']

# load BRCA data
X = scipy.io.loadmat(Data[0])

# extract relevant values
Samples = X['Gene_Patients']
Normalized = X['Gene_X']
Symbols = X['Gene_Symbs']
Survival = X['Gene_Survival']
Censored = X['Gene_Censored']

# get number of samples
N = Normalized.shape[0]

# load model and analyze
f = open(ModelPaths[0] + Models[0], 'rb')
Model = pickle.load(f)
f.close()
sn.analysis.FeatureAnalysis(Model, Normalized, Normalized, Symbols,
							Survival, Censored,
							NBox=10, NScatter=10, NKM=10, NCluster=50,
							Tau=5e-2, Path=ModelPaths[0])

# load BRCA + OV data
X = scipy.io.loadmat(Data[1])

# extract relevant values and BRCA samples
Samples = X['Patients'][-N:]
Normalized = X['X'][-N:,]
Symbols = X['Symbs']
Survival = X['Survival'][-N:]
Censored = X['Censored'][-N:]

# load model and analyze
f = open(ModelPaths[1] + Models[1], 'rb')
Model = pickle.load(f)
f.close()
sn.analysis.FeatureAnalysis(Model, Normalized, Normalized, Symbols,
							Survival, Censored,
							NBox=10, NScatter=10, NKM=10, NCluster=50,
							Tau=5e-2, Path=ModelPaths[1])

# load BRCA + OV + UCEC data
X = scipy.io.loadmat(Data[2])

# extract relevant values and BRCA samples
Samples = X['Patients'][-N:]
Normalized = X['X'][-N:,]
Symbols = X['Symbs']
Survival = X['Survival'][-N:]
Censored = X['Censored'][-N:]

# load model and analyze
f = open(ModelPaths[2] + Models[2], 'rb')
Model = pickle.load(f)
f.close()
sn.analysis.FeatureAnalysis(Model, Normalized, Normalized, Symbols,
							Survival, Censored,
							NBox=10, NScatter=10, NKM=10, NCluster=50,
							Tau=5e-2, Path=ModelPaths[2])
"""
# #############################################################################
# Glioma gene expression experiment ###########################################
# #############################################################################


# define model/dataset pairs
ModelPaths = ['../../']
Models = ['final_gene50']
Data = ['Brain_Gene.mat']

# load datasets and perform feature analysis
for i, Path in enumerate(ModelPaths):

	# load normalized data
	X = sio.loadmat(Data[i])

	# extract relevant values
	Samples = X['Patients']
	Normalized = X['Gene_X']
	Symbols = X['Gene_Symbs']
	Survival = X['Survival']
	Censored = X['Censored']

	# load model
	f = open(os.path.join(os.getcwd(),Path + Models[i]), 'rb')
	Model = pickle.load(f)
	f.close()
	resultPath = Path+Models[i]
	sn.analysis.FeatureAnalysis(Model, Normalized, Normalized, Symbols,
			Survival, Censored,
			NBox=10, NScatter=10, NKM=10, NCluster=50,
			Tau=5e-2, Path=resultPath)

	# #############################################################################
# Integrated models ###########################################################
# #############################################################################
"""
# define model/dataset pairs
ModelPaths = ['/Users/lcoop22/Drive/Work/Papers/SurvivalNet/Models/GBMLGG/Integ/',
			  '/Users/lcoop22/Drive/Work/Papers/SurvivalNet/Models/BRCA/BRCA_Integ/']
Models = ['nl1-hs10-dor0.62-id0final_model',
		  'nl1-hs10-dor0-id0final_model']
Data = ['/Users/lcoop22/Drive/Work/Papers/SurvivalNet/Data/SingleCancerDatasets/GBMLGG/Brain_Integ.mat',
		'/Users/lcoop22/Drive/Work/Papers/SurvivalNet/Data/SingleCancerDatasets/BRCA/BRCA_Integ.mat']

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

# #############################################################################
# Subtype integrated models ###########################################################
# #############################################################################

# define model/dataset pairs
ModelPaths = ['/Users/lcoop22/Drive/Work/Papers/SurvivalNet/Models/GBMLGG/IDHmut-non-codel-Integ/']
Models = ['nl1-hs500-dor0.9-id5final_model_IDHmut_noncodel_proteinexpr']
Data = ['/Users/lcoop22/Drive/Work/Papers/SurvivalNet/Data/SingleCancerDatasets/GBMLGG/Brain_Integ_Subtypes.mat']

# load datasets and perform feature analysis
for i, Path in enumerate(ModelPaths):

	# load normalized data
	X = scipy.io.loadmat(Data[i])

	# extract relevant values
	Normalized = X['IDHmut_non_codel_Integ_X']
	Raw = X['IDHmut_non_codel_Integ_X_Raw']
	Symbols = X['Integ_Symbs']
	Survival = X['IDHmut_non_codel_Survival']
	Censored = X['IDHmut_non_codel_Censored']

	# load model
	f = open(ModelPaths[i] + Models[i], 'rb')
	Model = pickle.load(f)
	f.close()

	sn.analysis.FeatureAnalysis(Model, Normalized, Raw, Symbols,
								Survival, Censored,
								NBox=10, NScatter=10, NKM=20, NCluster=100,
								Tau=5e-2, Path=ModelPaths[i])
								"""
