%clear all;
function superDeepSAE (id, Xall, X_LABELED, T, C, hiddenSize, stepSizeMax, stepSizeMin, maxiter, ...
    dropoutFraction, inputZeroMaskedFraction, unsupervisedLearningRate, augment, K, testApproach, path, randLastLayer, randAllLayers, removeBiasInCindex)

X = Xall;
%% train SAE here
%  Setup and train a stacked denoising autoencoder (SDAE)
rng(0, 'v5uniform')
sae = saesetup([size(X, 2) hiddenSize]);

for hl = 1:numel(sae.ae)
    sae.ae{hl}.activation_function       = 'sigm';
    sae.ae{hl}.learningRate              = unsupervisedLearningRate;
    sae.ae{hl}.inputZeroMaskedFraction   = inputZeroMaskedFraction;
    sae.ae{hl}.dropoutFraction           = dropoutFraction;
end

opts.numepochs = 200;
opts.batchsize = 8;
sae = saetrain(sae, X, opts);
 
% nn = mynnsetup([size(x, 2) hiddenSize 1]);
% nn.activation_function              = 'sigm';
% nn.learningRate                     = 1;
% nn.inputZeroMaskedFraction          = 0;
% for hl = 1:nn.n - 2
%     nn.W{hl} = sae.ae{hl}.W{1}';
% end

%% initialize cox coefficients
%b = coxphfit(x, T, 'censoring', C);
% nn.W{nn.n - 1} = [1;b];
% %% feed forward pass
% nn = mynnff(nn, X, T, C);
%% calculate lpl and cindex without fine-tuning on all data
% Xout = nn.a{nn.n - 1};
% Xout = Xout(:, 2:end);
% LogPartialL(Xout, T, C, b);
% cIndex(b, Xout, T, C)

%% back prop with K-fold cross validation
m = size(X_LABELED, 1);
if (augment == 1)
    m = m / 2;
end
F = floor(m / K);
cursor = 0;
lpl_train = zeros(maxiter, 1);
lpl_test = zeros(maxiter, 1);
cindex_train = zeros(maxiter, 1);
cindex_test = zeros(maxiter, 1);

while (cursor < F * K)
    starti = cursor + 1;
    if (m - cursor < K)
        endi = m;
    else
        endi = cursor + F;
    end
 
    x_test = X_LABELED(starti:endi, :);
    y_test = T(starti:endi);
    c_test = C(starti:endi);

    x_train = X_LABELED([1:starti - 1 endi + 1:m], :);
    y_train = T([1:starti - 1 endi + 1:m]);
    c_train = C([1:starti - 1 endi + 1:m]);

    if (augment == 1)
        x_test = [x_test; X_LABELED(starti + m:endi + m, :)]; 
        y_test = [y_test; T(starti + m:endi + m)];
        c_test = [c_test; C(starti + m:endi + m)];

        x_train = [x_train; X_LABELED([1+m:starti-1+m endi+1+m:m+m], :)];
        y_train = [y_train; T([1+m:starti-1+m endi+1+m:m+m])];
        c_train = [c_train; C([1+m:starti+m-1 endi+1+m:m+m])];
        
    end
    
    %% Use the SDAE to initialize a FFNN  
    nn = mynnsetup([size(x_train, 2) hiddenSize 1]);
    nn.activation_function       = 'sigm';
    nn.inputZeroMaskedFraction   = 0;
    nn.learningRate              = 1;
    nn.dropoutFraction           = 0;

    if (randAllLayers == 0)
        for hl = 1:nn.n - 2
             nn.W{hl} = sae.ae{hl}.W{1}';
        end
    end

%     %% initialize cox coefficients for last layer weights
    if (randLastLayer == 0)
        x = x_train;
        for s = 1 : numel(sae.ae)
            t = nnff(sae.ae{s}, x, x);
            x = t.a{2};
            %remove bias term
            x = x(:,2:end);
        end

        b = coxphfit(x, y_train, 'censoring', c_train);
        nn.W{nn.n - 1} = [1;b];
    end
    
    nn = mynnff(nn, x_train);
    Xred_test = nn.a{nn.n - 1};
    b = nn.W{nn.n - 1};
    
    if (removeBiasInCindex == 1)
        Xred_test = Xred_test(:, 2:end);
        b = b(2:end);
    end
    
    %% calculate lpl and cindex without fine-tuning on training data
    cIndex(b, Xred_test, y_test, c_test)
    LogPartialL(Xred_test, y_test, c_test, b)
    %% Train w. bp
    for iter = 1:1:maxiter
            %% change stepsize with iterations
            StepSize = (stepSizeMax) * ((maxiter) - iter) / maxiter + (stepSizeMin)* iter/maxiter;
            %%  differentiation
            nn = calcGradient(nn, y_train, c_train, b);
            
            %% gradient checking
%            [diff, grads] = gradCheck(nn, y_train, c_train, b);
            
            %% update weights
            for j = 1: nn.n - 1
                nn.W{j} = nn.W{j} + StepSize .* nn.deltaW{j};
            end
            
            %% get performance with updated weights
            % apply updated parameters to train data
            
            nn = mynnff(nn, x_train);
            Xred_train = nn.a{end - 1};
            b2 = nn.W{nn.n - 1};
            
            if (removeBiasInCindex == 1)
                b2 = b2(2:end, :);
                Xred_train = Xred_train(:, 2:end);
            end
            
            lpl_train_show = LogPartialL(Xred_train, y_train, c_train, b2);
            lpl_train(iter) = lpl_train(iter) + lpl_train_show;
            cindex_train_show = cIndex(b2, Xred_train, y_train, c_train);
            cindex_train (iter) = cindex_train (iter) + cindex_train_show;
            
            %% Test
            % apply updated parameters to test data
            nn_test = nn;
            nn_test.testing = 1;
            nn_test = mynnff(nn, x_test);
            Xred_test = nn_test.a{end - 1};
            if (removeBiasInCindex == 1)
                Xred_test = Xred_test(:, 2:end);
            end
            
            cindex_test_show = cIndex(b2, Xred_test, y_test, c_test);
            cindex_test (iter) = cindex_test (iter) + cindex_test_show;
            lpl_test_show = LogPartialL(Xred_test, y_test, c_test, b2); 
            lpl_test(iter) = lpl_test(iter) + lpl_test_show;
            
            if (mod(iter, 10) == 1)
                %iter;
                save(strcat(path, 'sae-', num2str(id), '-lpl-trn', '.mat'), 'lpl_train' );
                save(strcat(path, 'sae-', num2str(id), '-lpl-tst', '.mat'), 'lpl_test' );
                save(strcat(path, 'sae-', num2str(id), '-ci-trn', '.mat'), 'cindex_train' );
                save(strcat(path, 'sae-', num2str(id), '-ci-tst', '.mat'), 'cindex_test' );
            end
    end
    if (testApproach == 0)
        break;
    elseif (testApproach == 1)
        cursor = cursor + F;
    end
end
if (testApproach == 1)
    cindex_test = cindex_test / K;
    cindex_train = cindex_train / K;
    lpl_test = lpl_test / K;
    lpl_train = lpl_train / K;
end


save(strcat(path, 'sae-', num2str(id), '-lpl-trn', '.mat'), 'lpl_train' );
save(strcat(path, 'sae-', num2str(id), '-lpl-tst', '.mat'), 'lpl_test' );
save(strcat(path, 'sae-', num2str(id), '-ci-trn', '.mat'), 'cindex_train' );
save(strcat(path, 'sae-', num2str(id), '-ci-tst', '.mat'), 'cindex_test' );
save(strcat(path, 'sae-', num2str(id), 'nn', '.mat'), 'nn' );
% plot(1:iter, cindex_train(1: iter));
% hold on
% plot(1:iter, cindex_test(1: iter));
% figure
% plot(1:iter, lpl_train(1: iter));
% figure
% plot(1:iter, lpl_test(1: iter));

