function [ nn ] = calcGradient( nn, Y, C, b )
% calculate derivative of L wrt to all w in every layer, and
% store the results in nn.deltaW{j} for layer j.
    Xred = nn.a{nn.n - 1};
    J = nn.n; % number of layers
    [m n] = size(Xred);
    
%% calculate the m * n matrix of dLdX
    dLdX = dLogPartialLdX(Xred, Y, C, b);
        

%% Numerical calculation of dLdX
%     e = 1e-4;
%     dLdXapprox = zeros(m, n);
%     for i = 1:m
%         for j = 1:n
%             XredMinus = Xred;
%             XredMinus(i , j) = Xred(i , j) - e;
%             gradMinus = LogPartialL(XredMinus, Y, C, b);
%             
%             XredPlus = Xred;
%             XredPlus(i , j) = Xred(i, j) + e;
%             gradPlus = LogPartialL(XredPlus, Y, C, b);   
%             dLdXapprox(i , j) = (gradPlus - gradMinus) / (2 * e);
%         end
%     end
%     
%% Last layer
    nn.deltaW{J - 1} = dLogPartialL(Xred, Y, C, b)';
   
%% Exact computation of dXX
    dxx = cell(1,m);
    for i = 1:m
        dxx{i} = dXX(nn, i);
    end
    
    
%% Sigmoid layers
    nn.d_act = cell(size(nn.a));
    for j = (J - 1):-1:1
        switch nn.activation_function 
            case 'sigm'
                nn.d_act{j} = nn.a{j} .* (1 - nn.a{j});
            case 'tanh_opt'
                nn.d_act{j} = 1.7159 * 2/3 * (1 - 1/(1.7159)^2 * nn.a{j}.^2);
        end
    end
    for j = (J - 2):-1:1
%        tic;
        [P, Q] = size(nn.W{j});
        nn.deltaW{j} = zeros(P, Q);
%        nn.deltaWapprox{j} = zeros(P, Q);
%        nn.dXdwapprox{j} = zeros([P, Q, size(Xred)]);
        for p = P:-1:1
            for q = Q:-1:1
%% Numerical calcucation of dXdW
% 
%                 nnPlus = nn;
%                 nnMinus = nn;
%                 
%                 nnPlus.W{j}(p , q) = nnPlus.W{j}(p , q) + e;
%                 nnMinus.W{j}(p , q) = nnMinus.W{j}(p , q) - e;
%                 
%                 nnPlus = mynnff(nnPlus, nnPlus.a{1}(:, 2:end) );
%                 nnMinus = mynnff(nnMinus, nnMinus.a{1}(:, 2:end) );
%                 
%                 XredMinus = nnMinus.a{nn.n - 1};
%                 XredPlus = nnPlus.a{nn.n - 1};
%                 
%                 nn.dXdwapprox{j}(p,q, :,:) = (XredPlus - XredMinus)/(2 * e);

%% Back Propagation
                for i = 1:1:m    
                % for each w, dL/dw, note that L is the result of summation
                % over Xi
                   tmp = dXW(nn, dxx{i}, i, j, p, q);
%                    tmp2 = reshape(nn.dXdwapprox{j}(p, q, i, :), 1, n);
%                    diff = norm(tmp' - tmp2)
                   %nn.deltaWapprox{j}(p, q) = nn.deltaWapprox{j}(p, q) + reshape(nn.dXdwapprox{j}(p, q, i, :), 1, n) * ...
                   %dLdXapprox(i, :)';
                   nn.deltaW{j}(p, q) = nn.deltaW{j}(p, q) + tmp' * dLdX(i, :)';
                end
            end
        end
%        toc
    end
end