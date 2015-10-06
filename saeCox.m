clear all;
prepareData

%X = X(randperm(size(X, 1), :));
%% train SAE here
%  Setup and train a stacked denoising autoencoder (SDAE)
rng(0, 'v5uniform')
% [coeff, X] = pca(X, 'NumComponents', floor(size(X, 2)/10));
sae = saesetup([size(X, 2) 20]);
sae.ae{1}.activation_function       = 'sigm';
sae.ae{1}.learningRate              = 1;
sae.ae{1}.inputZeroMaskedFraction   = 0;
%sae.ae{2}.activation_function       = 'sigm';
%sae.ae{2}.learningRate              = 1;
%sae.ae{2}.inputZeroMaskedFraction   = 0;
opts.numepochs =   100;
opts.batchsize = 191;
sae = saetrain(sae, X, opts);
 
% %% obtain dimension reduced data
sae.ae{1} = nnff(sae.ae{1}, X, X);
Xred = sae.ae{1}.a{sae.ae{1}.n - 1};
Xred = Xred(:, 2:end);

%% Train the autoencoder 5-fold cross validation
K = 3;
m = size(X, 1);
F = floor(m / K);
cursor = 0;
genErrSum = 0;
cindex_train = 0;
cindex_test = 0;
while (cursor < F * K)
    starti = cursor + 1;

    if (m - cursor < K)
        endi = m;
    else
        endi = cursor + F;
    end
    X_test = Xred(starti:endi, :);
    y_test = T(starti:endi);
    c_test = C(starti:endi);
    X_train = Xred([1:starti - 1 endi + 1:m], :);
    y_train = T([1:starti - 1 endi + 1:m]);
    c_train = C([1:starti - 1 endi + 1:m]);

    %% cox coefficients
    [b2, logl, H, stats] = coxphfit(X_train, y_train);


    cindex_train  = cindex_train  + cIndex(b2, X_train, y_train, c_train);
    cindex_test  = cindex_test  + cIndex(b2, X_test, y_test, c_test);
    cursor = cursor + F; 
%perf = mse(autoenc1, autoenc1(D'), D', 'normalization', 'percent')
end
cindex_test = cindex_test / K;
cindex_train = cindex_train / K;
% net = configure(net, data_tr, data_tr);
% net.trainFcn = 'trainlm';
% net.performFcn = 'mse';
% net.performParam.normalization = 'percent';
% net.trainParam.epochs = 5;
% 
% net = train(net, data_tr, data_tr, 'useParallel', 'yes');
%view(net)
%y = zeros(size(outcome));
%y = net(data_tr);
% 
% 
% perf = mse(net,y,data_tr, 'normalization', 'percent')
% 
% net = configure(net, data_tr, outcome);
% net2 = train(net, data_tr, outcome);
% y2 = net2(data_tr);
% 
% perf2 = mse(net2,y2,outcome, 'normalization', 'percent')