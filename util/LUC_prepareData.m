% dataManip
% tstDataManip

load AML.mat
load AML.tst.mat
% % load('Ximputed.mat');
% % % load('Ximputed_t.mat');
% % %training data, survival time and censoring
% % D = Ximputed;
% % X = [D(:,2:18), D(:,23:end)];
% % %X = normalizeClns(X, [2,3,18:271]);
% % C = D(:, 20);
% % %C = normalizeClns(C, 1);
% % T = D(:, 21);
%  % %T = normalizeClns(T, 1);
X = X(1:190, :);
C = C(1:190);
T = T(1:190);
Xall = [X; Xtst];
N = size(Xall, 1);

for j = 1:size(Xall, 2)
    if (length(unique(Xall(:, j))) > 3)
        Xall(:, j) = (Xall(:, j) - ones(N,1)*mean(Xall(:, j),1)) ./ (ones(N,1)*std(Xall(:, j),[],1));
    end
end

% N = size(X, 1);
% for j = 1:size(X, 2)
%     if (length(unique(X(:, j))) > 3)
%         X(:, j) = (X(:, j) - ones(N,1)*mean(X(:, j),1)) ./ (ones(N,1)*std(X(:, j),[],1));
%     end
% end
% % randrows = randperm(size(X, 1));
% % T = T(randrows);
% % C = C(randrows);
% % X = X(randrows, :);
% 
%% get rid of protein data
%Xall = Xall(:, 1:20);

% pca on protein data
Xprot = Xall(:, 41:end);
[coeff, Xprot] = pca(Xprot, 'NumComponents', 150);

Xall = [Xall(:, 1:40), Xprot];
X = Xall(1:190, :);
% %% randomize data
%  X = rand(191, 271);
%  T = rand(191, 1);
%  C = (rand(191, 1) > .5);

% load Xall_normalized_pcareduced.mat
% 
% load X_normalized_pcareduced.mat
