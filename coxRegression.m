clear all;
close all;
load LUSC_P

K = 10;
shuffles = 10;
elasticnet = [1, .7, .5, .3, .1, 0];
c = zeros(length(elasticnet), K, shuffles);
stds = zeros(length(elasticnet), 1);
m = size(X, 1);
F = floor(m / K);
for sh= 1:shuffles
randind = randperm (m);
X = X(randind, :);
T = T(randind);
C = C(randind);
for i=1:length(elasticnet)
cursor = 0;
k = 1;
while (cursor < F * K)
    starti = cursor + 1;
    if (m - cursor < K)
        endi = m;
    else
        endi = cursor + F;
    end
    X_test = X(starti:endi, :);
    T_test = T(starti:endi);
    C_test = C(starti:endi);
    X_train = X([1:starti - 1 endi + 1:m], :);
    T_train = T([1:starti - 1 endi + 1:m]);
    C_train = C([1:starti - 1 endi + 1:m]);
    Y_train = [T_train, -C_train+1];
    testvalSize = size(X_test, 1);
    valsize = floor(testvalSize / 4);
    
    X_CV = X_test(1:valsize, :);
    T_CV = T_test(1:valsize);
    C_CV = C_test(1:valsize);
    Y_CV = [T_CV, -C_CV+1];
    
    X_test = X_test(valsize + 1:end, :);
    T_test = T_test(valsize + 1:end);
    C_test = C_test(valsize + 1:end);
    Y_test = [T_test, -C_test+1];
    %[b,logl,H,stats] = coxphfit(X_train, T_train, 'censoring', C_train, 'l2regrate', l2r, 'l1regrate', l1r);
    opts = glmnetSet;
    opts.alpha = elasticnet(i);
    opts.intr = false;
%    opts.maxit = 1000;
    options = glmnetSet(opts); 
    fit=glmnet(X_train, Y_train, 'cox', options);
    %glmnetPlot(fit);
    tmp = 0;
    for b = 1:(size(fit.beta, 2))
        if (tmp < cIndex(fit.beta(:, b), X_CV, T_CV, C_CV))
            chosen = b;
            tmp = cIndex(fit.beta(:, b), X_CV, T_CV, C_CV);
        end
    end
    c(i, k, sh) = c(i, k, sh) + cIndex(fit.beta(:, chosen), X_test, T_test, C_test);
    cursor = cursor + F ;
    k = k + 1
end
i
end
sh
end
