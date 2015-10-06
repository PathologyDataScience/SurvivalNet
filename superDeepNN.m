%clear all;
function superDeepNN (id, Xall, X_LABELED, T, C, hiddenSize, stepSizeMax, stepSizeMin, maxiter, ...
    dropoutFraction, inputZeroMaskedFraction, learningRate, augment, K, testApproach, path, randLastLayer, randAllLayers, removeBiasInCindex)

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
    nn.learningRate              = learningRate;
    nn.dropoutFraction           = dropoutFraction;
%     %% initialize cox coefficients for last layer weights

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
            if (mod(iter, 10) == 0)
                StepSize
            end
            %%  differentiation
            nn = calcGradient(nn, y_train, c_train, b);
            
            %% gradient checking
%             [diff, grads] = gradCheck(nn, y_train, c_train, b);
            
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
            iter
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

% plot(1:iter, cindex_train(1: iter));
% hold on
% plot(1:iter, cindex_test(1: iter));
% figure
% plot(1:iter, lpl_train(1: iter));
% figure
% plot(1:iter, lpl_test(1: iter));
