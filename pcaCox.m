clear all;
prepareData
%% constants 
iter = 1;
K = 3;

c = zeros(iter, 1);

m = size(X, 1);
F = floor(m / K);
cursor = 0;
cindex_train = 0;
cindex_test = 0;
while (cursor < F * K)
    starti = cursor + 1;

    if (m - cursor < K)
        endi = m;
    else
        endi = cursor + F;
    end
    
    X_test = X(starti:endi, :);
    y_test = T(starti:endi);
    c_test = C(starti:endi);
    X_train = X([1:starti - 1 endi + 1:m], :);
    y_train = T([1:starti - 1 endi + 1:m]);
    c_train = C([1:starti - 1 endi + 1:m]);

    [coeff,X_train_red] = pca(X_train, 'NumComponents', 20);
    %% cox coefficients
    [b2, logl, H, stats] = coxphfit(X_train_red, y_train);

        
    X_test_reduced = X_test * coeff;
    cindex_train  = cindex_train  + cIndex(b2, X_train_red, y_train, c_train);
    cindex_test  = cindex_test  + cIndex(b2, X_test_reduced, y_test, c_test);
    cursor = cursor + F; 
%perf = mse(autoenc1, autoenc1(D'), D', 'normalization', 'percent')
end
cindex_test = cindex_test / K;
cindex_train = cindex_train / K;

