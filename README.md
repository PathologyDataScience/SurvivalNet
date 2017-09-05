# SurvivalNet
SurvivalNet is a package for building survival analysis models using deep learning. The SurvivalNet package has the following features:

* Training deep networks for time-to-event data using Cox partial likelihood
* Automatic tuning of network architecture and learning hyper-parameters with Bayesian Optimization
* Interpretation of trained networks using partial derivatives
* Layer-wise unsupervised pre-training

A short paper descibing our approach of using Cox partial likelihood was presented in ICLR 2016 and is available [here](https://arxiv.org/pdf/1609.08663.pdf). A [longer paper](https://doi.org/10.1101/131367) was later published describing the package and showing applications in BioRxiv.

# Getting Started
The **examples** folder provides scripts to:

* Train a neural network on your dataset using Bayesian Optimization (Run.py)
* Set parameters for Bayesian Optimizaiton (BayesianOptimization.py)
* Define a cost function for use by Bayesian Optimization (CostFunction.py)
* Interpret a trained model and analyze feature importance (ModelAnalysis.py)

The Run.py training module takes as input a python dictionary containing the following numpy arrays:
* X: input data of size (number of patients, number of features). Patients must be sorted with respect to event or censoring times 'T'.
* T: Time of event or time to last follow-up, appearing in increasing order and corresponding to the rows of 'X'. size: (number of patients, ).
* O: Right-censoring status. A value of 1 means the event is observed (i.e. deceased or disease progression), a 0 value indicates that the sample is censored. size:(number of patients, ).
* A: An index array encoding the at-risk set for each sample. For sample _i_, the _i_ th element in A is the index of the next sample in the at risk group of _i_ (see Run.py for an example of how to calculate this vector using the provided functions) size:(number of patients, ).

These vectors are packed into a dictionty D and passed to train (found in train.py module) as demonstrated in Run.py.

The provided example scripts assume the data is a .mat file containinig, 'Survival', 'Censored', and either 'Integ\_X' or 'Gene\_X' depending on what feature set we are using.

## Installation Guide for Docker Image

A Docker image for SurvivalNet is provided for those who prefer not to build from source. This image contains an installation of SurvivalNet on a bare Ubuntu operating system along with sample data used in our paper published in *Scientific Reports*. This helps users avoid installation of the */bayesopt/* package and other dependencies required by SurvivalNet.

The SurvivalNet Docker Image can either be downloaded [here](https://hub.docker.com/r/cramraj8/survivalnet2.0/), or can be pulled from Docker hub using the following command:
    
    sudo docker pull cramraj8/survivalnet2.0

Running this image on your local machine with the command
    
    sudo docker run -it cramraj8/survivalnet2.0 /bin/bash

launches a terminal within the image where users have access to the package installation. 

Example python scripts can be found in the folder 
    
    cd /SurvivalNet/examples/ 

where users can see how to train and validate deep survival models. The main script
    
    python Run.py
    
will perform Bayesian optimization to identify the optimal deep survival model configuation and will update the terminal with the step by step updates of the learning process.

The sample data file - ***Brain_Integ.mat*** is located inside the */SurvivalNet/data/* folder. By default, ***Run.py*** uses this data for learning.


### Using your own data to train networks

You can train a network using your own data by mounting a folder within the SurvivalNet Docker image. The command

    sudo docker run -v /<hostmachine_data_path>/:/<container_data_path>/ -it cramraj8/survivalnet2.0 /bin/bash
    
will pull the Docker Image, run the container, and mount the host machine data directory with container data path.
The container Data Path is ***/SurvivalNet/data/<data_file_name>***. 
  
