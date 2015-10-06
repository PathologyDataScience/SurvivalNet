function Xn = normalizeClns(X, Clns)
Xn = X;
    for c = Clns
        [a, b, m] = zscore(Xn(:, c));
        Xn(:, c) = normalize(a, b, m);
        % = (Xn(:, c) - mean(Xn(:, c)))/std(Xn(:, c));
    end
end