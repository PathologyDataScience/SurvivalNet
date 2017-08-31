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

This project was build on Docker Platform for the easy use and the platform independency. The link for Docker Image is found [here](https://hub.docker.com/r/cramraj8/survivalnet2.0/).
You can pull the Docker Image using this command on terminal.
    
    sudo docker pull cramraj8/survivalnet2.0


Docker Image (***cramraj8/survivalnet2.0***) was built on top of Ubuntu-Docker Image. All the dependencies and libraries was added into the Docker Image. The *Bayesian Optimization* Python package was already installed inside the Docker Image. This ***Bayesian Optimization(BayesOpt package)*** can be located by */bayesopt/* folder.

The survivalNet python package will be found inside the *Ubuntu-Docker* along with *BayesOpt* folder. 


(Download and) Run the Docker Image (from Docker Hub) on local machine
    
    sudo docker run -it cramraj8/survivalnet2.0 /bin/bash


This command will look for the **survivalnet2.0** Docker Image locally and if not found, then the Docker Engine will look at the Docker Hub.
Once the Download is completed, a Docker Container will be created, and the terminal will turn into bash mode.



Now you are inside the Docker Container.
The project package is located inside the */SurvivalNet/* folder. 

If you want to run the example python script, you can navigate into
    
    cd /SurvivalNet/examples/ 
folder.
There you can find some python scripts. But *Run.py* is the proper python script that can tell you about the deep learning progress.
    
    python Run.py
will execute the python-script run command, and you will see the network’s learning process step by step.



The data - ***Brain_Integ.mat*** is located inside the */SurvivalNet/data/* folder.
By default, this data will be considered into the network learning process.



Once you done with exploration of SurvivalNet package, 
    
    exit
to exit the Docker Container.



Now you can check the Docker Image existence by,
    
    sudo docker images 


### For using your data to train the network from your local machine

For using the SurvivalNet Package with Docker, there is no need to write Dockerfile to pull the Docker Image.
    
    sudo docker run -v /<hostmachine_data_path>/:/<container_data_path>/ -it cramraj8/survivalnet2.0 /bin/bash
is enough to pull the Docker Image, run the container, and mount the host machine data directory with container data path.
The container Data Path is usually be
	***/SurvivalNet/data/<data_file_name>***
  
  
