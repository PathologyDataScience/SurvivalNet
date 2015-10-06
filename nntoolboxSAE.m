clear all
prepareData

K = 5;

%% configure secondary network
% Create an empty network
autoencHid1 = network;

% Set the number of inputs and layers
autoencHid1.numInputs = 1;
autoencHid1.numlayers = 1;

% Connect the 1st (and only) layer to the 1st input, and also connect the
% 1st layer to the output
autoencHid1.inputConnect(1,1) = 1;
autoencHid1.outputConnect = 1;

% Add a connection for a bias term to the first layer
autoencHid1.biasConnect = 1;

%% configure autoencoder
%hiddenSize1 = 100;
saeC = zeros(80,1);
for hiddenSize = 30:2:30
autoenc1 = feedforwardnet(hiddenSize);
autoenc1.trainFcn = 'trainscg';
autoenc1.trainParam.epochs = 200;

% Do not use process functions at the input or output
autoenc1.inputs{1}.processFcns = {};
autoenc1.outputs{2}.processFcns = {};

% Set the transfer function for both layers to the logistic sigmoid
autoenc1.layers{1}.transferFcn = 'logsig';
autoenc1.layers{2}.transferFcn = 'logsig';

% Divide samples into three sets randomly
autoenc1.divideFcn = 'dividerand';

autoenc1.performFcn = 'mse';
autoenc1.performParam.normalization = 'percent';

%autoenc1.performParam.L2WeightRegularization = 0.004;
%autoenc1.performParam.sparsityRegularization = 4;
%autoenc1.performParam.sparsity = 0.15;

%% Train the autoencoder 5-fold cross validation

    m = size(X, 1);
    F = floor(m / K);
    cursor = 0;
    genErrSum = 0;
    while (cursor < F * K)
        starti = cursor + 1;

        if (m - cursor < K)
            endi = m;
        else
            endi = cursor + F;
        end

        Xfold = X(starti:endi, :);
        yfold = T(starti:endi);
        cfold = C(starti:endi);
        Xtfold = X([1:starti - 1 endi + 1:m], :);
        ytfold = T([1:starti - 1 endi + 1:m]);
        ctfold = C([1:starti - 1 endi + 1:m]);

        autoenc1 = train(autoenc1, Xtfold', Xtfold');
        W1 = autoenc1.IW{1};
        %% REMOVE LAST LAYER
        % Set the size of the input and the 1st layer
        inputSize = size(Xtfold, 2);
        autoencHid1.inputs{1}.size = inputSize;
        autoencHid1.layers{1}.size = hiddenSize;

        % Use the logistic sigmoid transfer function for the first layer
        autoencHid1.layers{1}.transferFcn = 'logsig';

        % Copy the weights and biases from the first layer of the trained
        % autoencoder to this network
        autoencHid1.IW{1,1} = autoenc1.IW{1,1};
        autoencHid1.b{1,1} = autoenc1.b{1,1};

        feat1 = autoencHid1(Xtfold');
        feat1 = feat1';
        
        %% independent final layer training
        % Create an empty network
% finalCox = network;
% 
% % Set the number of inputs and layers
% finalCox.numInputs = 1;
% finalCox.numLayers = 1;
% 
% % Connect the 1st (and only) layer to the 1st input, and connect the 1st
% % layer to the output
% finalCox.inputConnect(1,1) = 1;
% finalCox.outputConnect = 1;
% 
% % Add a connection for a bias term to the first layer
% finalCox.biasConnect = 1;
% 
% % Set the size of the input and the 1st layer
% finalCox.inputs{1}.size = hiddenSize;
% finalCox.layers{1}.size = 1;
% 
% % Use the sigmoid transfer function for the first layer
% finalCox.layers{1}.transferFcn = 'logsig';
% 
% % Use all of the data for training
% finalCox.divideFcn = 'dividetrain';
% 
% % Use the cross-entropy performance function
% finalCox.performFcn = 'mse';
% 
% % You can experiment by the number of training epochs and the training
% % function
% finalCox.trainFcn = 'trainscg';
% finalCox.trainParam.epochs = 40;
% 
% finalCox = train(finalCox, feat1', ytfold');
        %% SUPERVISED AUTOENCODER CONFIGURATION
        % Create an empty network
        finalNetwork = network;

        % Specify one input and three layers
        finalNetwork.numInputs = 1;
        finalNetwork.numLayers = 2;

        % Connect the 1st layer to the input
        finalNetwork.inputConnect(1,1) = 1;

        % Connect the 2nd layer to the 1st layer
        finalNetwork.layerConnect(2,1) = 1;

        % Connect the output to the 2rd layer
        finalNetwork.outputConnect(2) = 1;

        % Add a connection for a bias term for each layer
        finalNetwork.biasConnect = [1; 1];

        % Set the size of the input
        finalNetwork.inputs{1}.size = inputSize;

        % Set the size of the first layer to the same as the layer in autoencHid1
        finalNetwork.layers{1}.size = hiddenSize;

        % Set the size of the second layer to the same as the layer in finalCox
        finalNetwork.layers{2}.size = 1;

        % Set the transfer function for the first layer to the same as in
        % autoencHid1
        finalNetwork.layers{1}.transferFcn = 'logsig';

        % Set the transfer function for the second layer to the same as in
        % autoencHid2
        finalNetwork.layers{2}.transferFcn = 'logsig';

        % Use all of the data for training
        finalNetwork.divideFcn = 'dividetrain';
        [beta,logl,H,stats] = coxphfit(feat1,ytfold);
        % Copy the weights and biases from the three networks that have already
        % been trained
        finalNetwork.IW{1,1} = autoencHid1.IW{1,1};
        finalNetwork.b{1} = autoencHid1.b{1,1};
        finalNetwork.LW{2,1} = beta';%finalCox.IW{1,1};
        finalNetwork.b{2} = 0;%finalCox.b{1,1};

        % Use the cross-entropy performance function
        finalNetwork.performFcn = 'mse';
        finalNetwork.performParam.normalization = 'percent';
        % You can experiment by changing the number of training epochs and the
        % training function
        finalNetwork.trainFcn = 'trainscg';
        finalNetwork.trainParam.epochs = 200;
        %view(finalNetwork);
        finalNetwork = train(finalNetwork, Xtfold', ytfold');
        predy = finalNetwork(Xtfold');
        cursor = cursor + F; 
    end
    saeC(hiddenSize) = saeC(hiddenSize) / K;

%perf = mse(autoenc1, autoenc1(D'), D', 'normalization', 'percent')
end