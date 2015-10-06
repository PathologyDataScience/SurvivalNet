function [ dLPLdX ] = dLogPartialLdX( X, Y, C, b )
% given n samples in n rows of X, computes the gradient of the log partial likelihood of a cox model
% with respect to n samples and returns the result in dLPLdX
    [Ysorted, Order] = sort(Y);
    %Censorted = Censored(Order);
    Xsorted = X(Order,:);
    Csorted = C(Order,:);
    [~, atRiskBegin] = ismember(Ysorted, Ysorted);
   
    [m, n] = size(X);
    
    dLPLdXSort = repmat(b', m , 1);% .* repmat(-(C - 1), 1, length(b));
    dLPLdXSort((Csorted == 1), :) = 0;
    Risk = exp(Xsorted * b);
    denom = cumsum(Risk, 'reverse');
    denom = denom(atRiskBegin);
    
    for i = 1:m % iterate to calc derivative wrt to all samples
        for j = 1:m % the sum in the ll expression for each sample
            if (~Csorted(j) && (Ysorted(j) <= Ysorted(i)))
                dLPLdXSort(i, :) = dLPLdXSort(i, :) - ...
                 (Risk(i) .* b') ./ denom(j);         
            end
        end
%        dLPLdX(i, :) = rehsape(dLPLdX, [1, length(b)]);
    end
    dLPLdX(Order, :) = dLPLdXSort;
end

