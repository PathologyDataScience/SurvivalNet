% dataManip
% tstDataManip

% load AML.mat
% load AML.tst.mat
% X = X(1:190, :);
% C = C(1:190);
% T = T(1:190);
% Xall = [X; Xtst];
% N = size(Xall, 1);
% 



% % randrows = randperm(size(X, 1));
% % T = T(randrows);
% % C = C(randrows);
% % X = X(randrows, :);

%% get rid of protein data
%Xall = Xall(:, 1:20);

% pca on protein data
% Xprot = Xall(:, 41:end);
% [coeff, Xprot] = pca(Xprot, 'NumComponents', 150);
% 
% Xall = [Xall(:, 1:40), Xprot];
% X = Xall(1:190, :);
% %% randomize data
%  X = rand(191, 271);
%  T = rand(191, 1);
%  C = (rand(191, 1) > .5);

% load Xall_normalized_pcareduced.mat
% 
% load X_normalized_pcareduced.mat

load LUSC_P.mat
%load LUAD_P.mat
%load Brain_P.mat

% [coeff, X] = pca(X, 'NumComponents', 150);
% 
N = size(X, 1);
for j = 1:size(X, 2)
    if(length(unique(X(:, j))) > 3)
        X(:, j) = (X(:, j) - ones(N,1)*mean(X(:, j),1)) ./ (ones(N,1)*std(X(:, j),[],1));
    end
end
