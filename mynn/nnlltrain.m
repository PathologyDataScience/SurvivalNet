function [ nn ] = nnlltrain( nn, X, y, C )
%NNLLTRAIN Summary of this function goes here
%   Detailed explanation goes here
    b = rand(size(X, 2), 1);
    J = nn.n; % number of layers
    m = size(X, 1);
    
    % a feed forward pass with all samples
    dLPLdX = dLogPartialLdX(nn.a{J}(:, :), Y, C, b);
    for j = J:1
        [Q P] = size(nn.W{j})
        deltaW{j} = zeros(Q, P);
        for q = Q:1
            for p = P:1
                for i=1:m
                    
                    % calculate derivative
                    deltaW{j}(q, p) = deltaW{j}(q, p) + dLPLdX(i, :) * dXndW(j, p, q);
                end
            end
        end
    end

end