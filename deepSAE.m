prepareData

%% train SAE here
%  Setup and train a stacked denoising autoencoder (SDAE)
rng(0, 'v5uniform')
sae = saesetup([271 100 100]);
sae.ae{1}.activation_function       = 'sigm';
sae.ae{1}.learningRate              = 1;
sae.ae{1}.inputZeroMaskedFraction   = 0.5;
sae.ae{2}.activation_function       = 'sigm';
sae.ae{2}.learningRate              = 1;
sae.ae{2}.inputZeroMaskedFraction   = 0.5;
opts.numepochs =   1;
opts.batchsize = 191;
sae = saetrain(sae, X, opts);
%visualize(sae.ae{1}.W{1}(:,2:end)')

%% remove last layer and obtain dimension reduced data
%reducedX = sae.ae{1}.a{end - 1};

%% Use the SDAE to initialize a FFNN
nn = nnsetup([271 100 100]);
nn.activation_function              = 'sigm';
nn.learningRate                     = 1;
nn.W{1} = sae.ae{1}.W{1};
nn.W{2} = sae.ae{2}.W{1};
nn.W{end} = zeros();

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
    %[b, logl, H, stats] = coxphfit(x_train(:, 2:end), y_train);
    %% fine tune FFNN here
    opts.numepochs =   1;
    opts.batchsize = size(x_train, 1);
    nn = nntrain(nn, x_train, y_train, opts);
    [er, bad] = nntest(nn, x_test, y_test)
    %assert(er < 0.16, 'Too big error');
    c = c + cIndexMod(nn.a{end}(2:end)', y_test, c_test);
    %c = c + cIndex(b, x_test(:, 2:end), y_test, c_test);
    cursor = cursor + F;
end
c = c / K;

