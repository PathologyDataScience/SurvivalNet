function dL = dLogPartialL(X, Y, Censored, Beta)
%Calculates the derivative log-partial likelihood for a cox model given 
%input features X, event and followup times Y, right-censoring, and model 
%weights.
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

%determine index of who is at risk at each unique time (to deal with
%tied-times)
[~, AtRisk] = ismember(Y, Y);

%calculate risk
Risk = exp(X*Beta);

%calculate partial sums
Numerator = cumsum(X .* (Risk * ones(1,size(X,2))), 'reverse');
Numerator = Numerator(AtRisk,:);
Denominator = cumsum(Risk, 'reverse');
Denominator = Denominator(AtRisk,:) * ones(1,size(X,2));

%sum over non-censored subjects
dL = (1-Censored.') * (X - (Numerator ./ Denominator));


%     obsfreq = freq .* ~cens;
%     Xb = X*b;
%     r = exp(Xb);
%     risksum = cumsum(freq.*r, 'reverse'); 
%     risksum = risksum(atrisk);
%     L = obsfreq'*(Xb - log(risksum));

%     if nargout>=2
%         % Compute first derivative dL/db
%         [n,p] = size(X);
%         Xr = X .* repmat(r.*freq,1,p);
%         Xrsum = cumsum(Xr, 'reverse'); 
%         Xrsum = Xrsum(atrisk,:);
%         A = Xrsum ./ repmat(risksum,1,p);
%         dl = obsfreq' * (X-A);
%         if nargout>=5
%             % Set the mlflag (monotone likelihood flag) to indicate if the
%             % likelihood appears to be monotone, not at an optimum.  This
%             % can happen if, at each of the sorted failure times, the
%             % specified linear combination of the X's is larger than that
%             % of all other observations at risk.
%             if n>2
%                 mlflag = all(cens(1:end-1) | (diff(Xb)<0 & ~tied(1:end-1)));
%             else
%                 mlflag = true;
%             end
%         end
%     end
%     if nargout>=3
%         % Compute second derivative d2L/db2
%         t1 = repmat(1:p,1,p);
%         t2 = sort(t1);
%         XXr = X(:,t1) .* X(:,t2) .* repmat(r.*freq,1,p^2);
%         XXrsum = cumsum(XXr, 1, 'reverse');
%         XXrsum = XXrsum(atrisk,:,:) ./ repmat(risksum,[1,p^2]);
%         ddl = reshape(-obsfreq'*XXrsum, [p,p]);
%         ddl = ddl + A'*(A.*repmat(obsfreq,1,p));
%     end