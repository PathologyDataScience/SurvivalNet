function out = dXX(nn, i)
    J = nn.n - 1;
    out = cell(J, 1);    
    out{J} = eye(size(nn.a{J},2));
    
    for j = (J - 1):-1:1
        B = dXXadj(nn, j, i);
        out{j} = out{j + 1} * B;
    end
end

function B = dXXadj(nn, l1, i)
    B = bsxfun(@times, nn.W{l1}, dSigm(nn.a{l1}(i, :) * nn.W{l1}))';
    B = [zeros(1, size(nn.a{l1}, 2)); B];  
end