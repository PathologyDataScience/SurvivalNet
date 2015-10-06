function L = LogPartialL(X, Y, Censored, Beta)
%Calculates the log-partial likelihood for a cox model given input features
%X, event and followup times Y, right-censoring, and model weights.
%inputs:
%X - D x N matrix of subject features. Each column is a subject, each row a
%    feature.
%Y - N-length vector of event and followup times.
%Censored - N-length binary vector of right-censoring. A value of 1
%            indicates that the subject was alive at the last followup.
%Beta - D-length vector of model weights.
%outputs:
%L - log partial likelihood of X, Y, Censoring given model defined by Beta.

%sort samples by outcome
[Y, Order] = sort(Y);
Censored = Censored(Order);
X = X(Order,:);

%calculate XB
Prediction = X*Beta;

%calculate log partial sums
PartialSums = cumsum(exp(Prediction), 'reverse');

%determine index of who is at risk at each unique time (to deal with
%tied-times)
[~, AtRisk] = ismember(Y, Y);

%sum over non-censored subjects
L = (1-Censored.') * (Prediction - log(PartialSums(AtRisk)));