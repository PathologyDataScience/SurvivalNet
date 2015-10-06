function [ diff, grads ] = gradCheck(nn, T, C, b)
%GRADCHECK gradient cheking wrt w_j in layer l
    e = 1e-4;
    L = nn.n;
    b = b;
    diff{L - 1} = zeros(size(b));
    grads{L - 1} = zeros(size(b));
    Xred = nn.a{L - 1};
    for j = 1:numel(b)
        gradPlus = LogPartialL(Xred, T, C, [b(1:j - 1); b(j) + e; b(j+1:end)]);
        gradMinus = LogPartialL(Xred, T, C, [b(1:j - 1); b(j) - e; b(j+1:end)]);
        approxGrad = (gradPlus -  gradMinus) / (2 * e);
        diff{L - 1}(j) = nn.deltaW{L - 1}(j)-approxGrad;
        grads{L - 1}(j) = approxGrad;
    end
    for l = 1: L-2
        diff{l} = zeros(size(nn.W{l}));
        grads{l} = zeros(size(nn.W{l}));
        for p = 1:size(nn.W{l}, 1)
            for q = 1:size(nn.W{l}, 2)
                
                nnPlus = nn;
                nnPlus.W{l}(p, q) = nnPlus.W{l}(p, q) + e;
                nnPlus = mynnff(nnPlus, nnPlus.a{1}(:, 2:end));
                XredPlus = nnPlus.a{L - 1};
                gradPlus = LogPartialL(XredPlus, T, C, b);
                
                nnMinus = nn;
                nnMinus.W{l}(p, q) = nnMinus.W{l}(p, q) - e;
                nnMinus = mynnff(nnMinus, nnMinus.a{1}(:, 2:end));
                XredMinus = nnMinus.a{L - 1};
                gradMinus = LogPartialL(XredMinus, T, C, b);
                
                approxGrad = (gradPlus -  gradMinus) / (2 * e);
                diff{l}(p, q) = nn.deltaW{l}(p, q) - approxGrad;
                grads{l}(p, q) = approxGrad;
            end
        end
    end
end

