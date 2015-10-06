clear all;
prepareData;
augmentData;

JobGen(0, 1, 6, 0.000100, 0.000005, 300, 0.000000, 0.000000, 3, 0, '/home/syouse3/git/survivalnet/survivalnet/NNSA-master/results/0.0001-do0-au0-ae1/');
%X = X(randperm(size(X, 1), :));
%% train SAE here
%  Setup and train a stacked denoising autoencoder (SDAE)
rng(0, 'v5uniform')
% [coeff, X] = pca(X, 'NumComponents', floor(size(X, 2)/10));
% sae = saesetup([size(X, 2) 20]);
% sae.ae{1}.activation_function       = 'sigm';
% sae.ae{1}.learningRate              = 1;
% sae.ae{1}.inputZeroMaskedFraction   = 0;
% %sae.ae{2}.activation_function       = 'sigm';
% %sae.ae{2}.learningRate              = 1;
% %sae.ae{2}.inputZeroMaskedFraction   = 0;
% opts.numepochs =   1;
% opts.batchsize = 191;
% sae = saetrain(sae, X, opts);
%  
% % %% obtain dimension reduced data
% sae.ae{1} = nnff(sae.ae{1}, X, X);
% Xred = sae.ae{1}.a{sae.ae{1}.n - 1};
% Xred = Xred(:, 2:end);

%% Use the SDAE to initialize a FFNN  
% nn = mynnsetup([size(X, 2) 1]);
% nn.activation_function              = 'sigm';
% nn.learningRate                     = 1;
% nn.inputZeroMaskedFraction          = 0;
% nn.W{1} = rand(size(X,2), 1);
%nn.W{2} = sae.ae{2}.W{1}';

%% initialize cox coefficients
%[b, logl, H, stats] = coxphfit(X, T);
%nn.W{nn.n - 1} = [1;b];

%% feed forward pass
%nn = mynnff(nn, X, T, C);

%% calculate lpl and cindex without fine-tuning
% Xred = nn.a{nn.n - 1};
% Xred = Xred(:, 2:end);
% LogPartialL(Xred, T, C, b)
% cIndex(b, Xred, T, C)

%% back prop
K = 5;
m = size(X, 1);
F = floor(m / K);
cursor = 0;
c = 0;
while (cursor < F * K)
    starti = cursor + 1;
    if (m - cursor < K)
        endi = m;
    else
        endi = cursor + F;
    end
    %x_test = reducedX(starti:endi, :);
    x_test = X(starti:endi, :);
    y_test = T(starti:endi);
    c_test = C(starti:endi);
    %x_train = reducedX([1:starti - 1 endi + 1:m], :);
    x_train = X([1:starti - 1 endi + 1:m], :);
    y_train = T([1:starti - 1 endi + 1:m]);
    c_train = C([1:starti - 1 endi + 1:m]);

    nn = mynnsetup([size(x_train, 2) 1]);
    nn.activation_function              = 'sigm';
    nn.learningRate                     = 1;
    nn.inputZeroMaskedFraction          = 0;
    nn.W{1} = [ones(1, 1); rand(size(x_train, 2), 1)];%sae.ae{1}.W{1}';
    
    nn = mynnff(nn, x_train, y_train, c_train);
    Xred_train = nn.a{nn.n - 1};
    Xred_train = Xred_train(:, 2:end);
    

    
    %% Train
    maxiter = 2000;
    lpl_train = zeros(maxiter, 1);
    lpl_test = zeros(maxiter, 1);
    cindex_train = zeros(maxiter, 1);
    cindex_test = zeros(maxiter, 1);
    for iter = 1:1:maxiter
            [diff, grads] = gradCheck(nn, y_train, c_train);

            for j = 1: nn.n - 1
                nn.W{j} = nn.W{j} + StepSize .* grads{j};
            end
            b2 = nn.W{nn.n - 1};
            b2 = b2(2:end, :);
            %nn = mynnff(nn, x_train, y_train, c_train);
            %Xred = nn.a{end - 1};
            %Xred = Xred(:, 2:end);
            %% Test
            iter
            lpl_train(iter) = lpl_train(iter) + LogPartialL(Xred_train, y_train, c_train, b2);
            cindex_train (iter) = cindex_train (iter) + cIndex(b2, Xred_train, y_train, c_train);
            %% Test
            nn_test = mynnff(nn, x_test, y_test, c_test);
            Xred_test = nn_test.a{end - 1};
            Xred_test = Xred_test(:, 2:end);            
            cindex_test (iter) = cindex_test (iter) + cIndex(b2, Xred_test, y_test, c_test);
            lpl_test(iter) = lpl_test(iter) + LogPartialL(Xred_test, y_test, c_test, b2);
    end
end
cindex_test = cindex_test / K;
cindex_train = cindex_train / K;
lpl_test = lpl_test / K;
lpl_train = lpl_train / K;
 
% K = 45;
% m = size(X, 1);
% F = floor(m / K);
% cursor = 0;
% c = 0;
% while (cursor < F * K)
%     starti = cursor + 1;
%     if (m - cursor < K)
%         endi = m;
%     else
%         endi = cursor + F;
%     end
%     %x_test = reducedX(starti:endi, :);
%     x_test = X(starti:endi, :);
%     y_test = T(starti:endi);
%     c_test = C(starti:endi);
%     %x_train = reducedX([1:starti - 1 endi + 1:m], :);
%     x_train = X([1:starti - 1 endi + 1:m], :);
%     y_train = T([1:starti - 1 endi + 1:m]);
%     c_train = C([1:starti - 1 endi + 1:m]);
%     %[b, logl, H, stats] = coxphfit(x_train(:, 2:end), y_train);
%     %% fine tune FFNN here
%     opts.numepochs =   1;
%     opts.batchsize = size(x_train, 1);
%     nn = nntrain(nn, x_train, y_train, opts);
%     [er, labels] = nntest(nn, x_test, y_test);
%     [labels y_test];
%     %assert(er < 0.16, 'Too big error');
%     c = c + cIndexMod(labels, y_test, c_test);
%     %c = c + cIndex(nn.W{end}', x_test(:, 2:end), y_test, c_test);
%     cursor = cursor + F;
% end
% c = c / K;

