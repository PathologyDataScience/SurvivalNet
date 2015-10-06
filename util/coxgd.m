%script to analyze AML data

close all; clear all; clc;

%load data
load AML.mat;

%define model parameters
K = 10;
Iterations = 10000;
StepSize = 1e-5;

%get dimensions
N = length(T);
D = size(X, 2);

%select random subset of features
Indices = ceil(D*rand(K,1));

%z-score normalization
X = (X - ones(N,1)*mean(X,1)) ./ (ones(N,1)*std(X,[],1));
X = X(:,Indices);

%initialize model
Beta = 0.1*rand(K,1);

%solution according to coxphfit.m
[mBeta, mL] = coxphfit(X, T, 'censoring', C, 'init', Beta);

%calculate initial log-partial likelihood
L0 = LogPartialL(X, T, C, Beta);

%simple gradient optimization
L = zeros(1,Iterations);
GradMag = 1;
for i = 1:Iterations
    L(i) = LogPartialL(X, T, C, Beta);
    dL = dLogPartialL(X, T, C, Beta).';
    Beta = Beta + StepSize * dL;
end

%display results - show Matlab's best answer
figure; plot(L); hold on;
plot([1 Iterations], [mL mL], 'r');