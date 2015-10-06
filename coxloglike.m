function L = coxloglike(P, B, C, T)
    L = 0;
    denom = 0;
    for i = 1:numel(C)
        if (C == 0)
            for j = 1:numel(C)
                if (T(j) >= T(i))
                    denom = denom + exp(P(:, j) * B);
                end
            end
            L = L + (P(:, i) * B) - log(denom);   
        end
    end
end