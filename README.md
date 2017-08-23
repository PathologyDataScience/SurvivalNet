# SurvivalNet
Survival net is an automated pipeline for survival analysis using deep learning. It is implemented in python using Theano to be compatible with current GPUs. It features the following functionalities:

* Training deep fully connected networks with Cox partial likelihood for survival analysis.
* Layer-wise unsupervised pre-training of the network
* Automatic hyper-parameter tuning with Bayesian Optimization
* Interpretation of the trained neural net based on partial derivatives

A short paper descibing this software and its performance compared to Cox+ElasticNet and Random Survival Forests was presented in ICLR 2016 and is available [here](https://arxiv.org/pdf/1609.08663.pdf).

In the **examples** folder you can find scripts to:

* Train a neural network on your dataset using Bayesian Optimization (Run.py)
* Set parameters for Bayesian Optimizaiton (BayesianOptimization.py)
* Define a cost function for use by Bayesian Optimization (CostFunction.py)
* Interpret a trained model and analyse feature importance (ModelAnalysis.py)

The example scripts provided assume the data is a .mat file containinig, 'Survival', 'Censored', and either 'Integ\_X' or 'Gene\_X' depending on what feature set we are using. But the train module takes the following numpy arrays packed in a dictionary as input:

X: input data of size (number of patients, number of features). Patients must be sorted with respect to T.
T: sorted survival labels; either time of event or time to last follow-up. size: (number of patients, ).
O: observed status. Array of 1s and 0s. 1 means event is observed, 0 means sample is censored. size: (number of patients, ). Also sorted with respect to T.
A: for patient _i_, the corresponding element in A is the index of the first patient in the at risk group of _i_. Look at Run.py for an example of how to calculate this vector using the provided functions.

These vectors are packed into a dictionty D and passed to train (found in train.py module) as demonstrated in Run.py.
