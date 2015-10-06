function c = cIndex(Beta, X, Survival, Censoring)
%Compute concordance-index of survival times given survival, censoring, 
%model coefficients and predictor values. See Frank Harrell's "Regression 
%Modeling Strategies".
%inputs:
%Beta - P x 1 vector of model coefficients.
%X - N x P matrix of predictor values - predictors in columns and samples
%    in rows.
%Survival - N x 1 vector of followup or survival times.
%Censoring - N x 1 vector of [0, 1], where a 1 indicates sample was right-
%            censored.
%outputs:
%c - concordance index measuring agreement between actual and predicted
%    ordering of useable sample pairs. Useable pairs are those where both
%    samples are uncensored and have unequal failure times, and pairs where
%    one sample is censored but the last followup exceeds the failure time
%    of the uncensored sample in the pair.

%generate matrix of comparable samples
Comparable = false(length(Survival));
for i = 1:length(Survival)
    for j = 1:length(Survival)
        if(Censoring(i) == 0 && Censoring(j) == 0)
            if(Survival(i) ~= Survival(j))
                Comparable(i,j) = true;
            end
        elseif(Censoring(i) == 1 && Censoring(j) == 1)
            Comparable(i,j) = false;
        else %one sample is censored and the other is not
            if(Censoring(i) == 1)
                if(Survival(i) >= Survival(j))
                    Comparable(i,j) = true;
                end
            else
                if(Survival(j) >= Survival(i))
                    Comparable(i,j) = true;
                end
            end
        end
    end
end

%get list of comparable pairs
[p1, p2] = find(Comparable);

%make predictions on data
Y = - X * Beta;

%judge accuracy of predictions
c = 0;
for i = 1:length(p1)
    if(Censoring(p1(i)) == 0 && Censoring(p2(i)) == 0)
        if(Y(p1(i)) == Y(p2(i)))
            c = c + 0.5;
        elseif((Y(p1(i)) > Y(p2(i))) && (Survival(p1(i)) > Survival(p2(i))))
            c = c + 1;
        elseif((Y(p2(i)) > Y(p1(i))) && (Survival(p2(i)) > Survival(p1(i))))
            c = c + 1;
        end
    elseif(Censoring(p1(i)) == 1 && Censoring(p2(i)) == 1)
        %do nothing - samples cannot be ordered
    else %one sample is censored and the other is not
        if(Censoring(p1(i)) == 1)
            if((Survival(p1(i)) > Survival(p2(i))) & (Y(p1(i)) > Y(p2(i))))
                c = c + 1;
            elseif(Y(p1(i)) == Y(p2(i)))
                c = c + 0.5;
            end
        else
            if((Survival(p2(i)) > Survival(p1(i))) & (Y(p2(i)) > Y(p1(i))))
                c = c + 1;
            elseif(Y(p1(i)) == Y(p2(i)))
                c = c + 0.5;
            end
        end
    end
end

%normalize c
c = c / length(p1);