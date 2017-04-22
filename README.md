# SurvivalNet
Survival net is an automated pipeline for survival analysis using deep learning, implemented in python using Theano to be compatible with current GPUs. It features the following functionalities:

* Training deep fully connected networks with Cox partial likelihood for survival analysis.
* Layer-wise unsupervised pre-training of the network
* Automatic hyper-parameter tuning with Bayesian Optimization
* Interpretation of the trained neural net based on partial derivatives

A short paper descibing this software and its performance compared to Cox+ElasticNet and Random Survival Forests was presented in ICLR 2016 and is available [here](https://arxiv.org/pdf/1609.08663.pdf).


