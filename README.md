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

* X: input data of size (number of patients, number of features). Patients must be sorted with respect to T.
* T: sorted survival labels; either time of event or time to last follow-up. size: (number of patients, ).
* O: observed status. Array of 1s and 0s. 1 means event is observed, 0 means sample is censored. size:(number of patients, ). 
Also sorted with respect to T.
* A: for patient _i_, the corresponding element in A is the index of the first patient in the at risk group of _i_. Look at Run.py for an example of how to calculate this vector using the provided functions. size:(number of patients, ).


These vectors are packed into a dictionty D and passed to train (found in train.py module) as demonstrated in Run.py.










## Installation Guide for Docker Image


This project was build on Docker Platform for the easy use and the platform independency. The link for Docker Image is found [here](https://hub.docker.com/r/cancerdatascience/snet/).
You can pull the Docker Image using this command on terminal.
    
    sudo docker pull cancerdatascience/snet:version1


Docker Image (***cancerdatascience/snet:version1***) was built on top of Ubuntu-Docker Image. All the dependencies and libraries was added into the Docker Image. The *Bayesian Optimization* Python package was already installed inside the Docker Image. This ***Bayesian Optimization(BayesOpt package)*** can be located by */bayesopt/* folder.

The survivalNet python package will be found inside the *Ubuntu-Docker* along with *BayesOpt* folder. 


(Download and) Run the Docker Image (from Docker Hub) on local machine
    
    sudo docker run -it cancerdatascience/snet:version1 /bin/bash


This command will look for the **snet:version1** Docker Image locally and if not found, then the Docker Engine will look at the Docker Hub.
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
    
    sudo docker run -v /<hostmachine_data_path>/:/<container_data_path>/ -it cancerdatascience/snet:version1 /bin/bash
is enough to pull the Docker Image, run the container, and mount the host machine data directory with container data path.
The container Data Path is usually be
	***/SurvivalNet/data/<data_file_name>***
  
  
