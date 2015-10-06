function JobGen(pretrain, hSize, augment, id, stepSizeMax, stepSizeMin, maxiter, learningRate, dropoutFraction, inputZeroMaskedFraction, K, testApproach, randLastLayer, randAllLayers, removeBiasInCindex, path )

    prepareData;
    if (augment == 0)
    else
        augmentData;
    end
    
    archs = cell(40, 1);

    %% initialize archs
    archs{1} = hSize;% .* ones(1, 20);
    for i = 2:size(archs, 1)
        archs{i} = [archs{i - 1}, hSize];
    end
    hiddenSize = archs{id};
    if(pretrain == 1)
        superDeepSAE (id, Xall, X, T, C, hiddenSize, stepSizeMax, stepSizeMin, maxiter, ...
                dropoutFraction, inputZeroMaskedFraction, learningRate, augment, K, testApproach, path, randLastLayer, randAllLayers, removeBiasInCindex)
    else
        superDeepNN (id, Xall, X, T, C, hiddenSize, stepSizeMax, stepSizeMin, maxiter, ...
                dropoutFraction, inputZeroMaskedFraction, unsupervisedLearningRate, augment, K, testApproach, path, randLastLayer, randAllLayers, removeBiasInCindex)
   end
end

