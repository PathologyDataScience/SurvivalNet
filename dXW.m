function [ tmp ] = dXW(nn, dXX, i, l1, p, q)
%DXW calculates dXred/dw{l1}(p, q)      
%    tmp = dXX{l1+1}(:, q + 1) * dSigm(nn.a{l1}(i, :) * nn.W{l1}(:, q)) * nn.a{l1}(i, p);
    tmp = dXX{l1+1}(:, q + 1) * nn.d_act{l1 + 1}(i, q + 1) * nn.a{l1}(i, p);
end