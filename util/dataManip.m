X = text2cell('trainingData-release.csv', '\t');
X = X(2:end,:);

%user ID
X(:,1) = mat2cell((1:size(X, 1))', ones(size(X, 1), 1));

%sex F=1, M=0
X(:,2) = mat2cell(double(strcmp(X(:,2), 'F')), double(ones(size(X, 1), 1)));
% %age 
X(:,3) = mat2cell(str2double(X(:,3)), ones(size(X, 1), 1), ones(1, 1));

%21:end 
 %X(:,21:end) = mat2cell(str2double(X(:,21:end)), ones(size(X, 1), 1), ones(size(X(:,21:end), 2), 1));

%AHD
X(:,4) = mat2cell(str2double(X(:,4)), ones(size(X, 1), 1), ones(1, 1));

%5:7 YES=1, NO=0
X(:,5:7) = mat2cell(double(strcmp(X(:,5:7), 'YES')), ones(size(X, 1), 1), ones(3, 1));

%8 Yes=1, No=0
X(:,8) = mat2cell(double(strcmp(X(:,8), 'Yes')), ones(size(X, 1), 1), ones(1, 1));

%CytoCat diploid=0,
X(:,9) = mat2cell(double(~strcmp(X(:,9), 'diploid')), ones(size(X, 1), 1), ones(1, 1));

%resp.simple CR=1, Resis=0
X(:,14) = mat2cell(double(strcmp(X(:,14), 'CR')), ones(size(X, 1), 1), ones(1, 1));
% surv alive=1, dead=0
X(:,16) = mat2cell(double(strcmp(X(:,16), 'A')), ones(size(X, 1), 1), ones(1, 1));    
%binarize chemo.simplest
header = unique(X(:,13));
chemo = zeros(size(X, 1), size(header, 1));
for i=1:size(X, 1)
    chemo(i,:) = double(strcmp(header, X(i,13))');
end
chemo = mat2cell(chemo, ones(size(X, 1), 1), ones(5, 1));
X = [X(:,1:12) chemo X(:, 14:end)];
    
%prepare data for knnimpute
X(strcmp(X(:,10), 'ND'), 10) = {double(NaN)};
X(strcmp(X(:,11), 'ND'), 11) = {double(NaN)};
X(strcmp(X(:,12), 'NotDone'), 12) = {double(NaN)};
X(strcmp(X, 'NA')) = {double(NaN)};

X(strcmp(X, 'POS')) = {double(1)};
X(strcmp(X, 'NEG')) = {double(0)};
X(strcmp(X, 'No')) = {double(0)};
X(strcmp(X, 'Yes')) = {double(1)};

X(:,21:end) = mat2cell(str2double(X(:, 21:end)), ones(size(X, 1), 1), ones(size(X(:,21:end),2), 1));
X = cell2mat(X);
Ximputed = knnimpute(X');
Ximputed = Ximputed';
save('Ximputed', 'Ximputed');
