function [ y ] = dSigm( x )
    t = sigm(x);
    y = t .* (ones(size(x)) - t);
end

