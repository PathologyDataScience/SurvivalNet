close all;
%fclose all;
clear all;
cancerType = 'LUSC';
OutputFolder = ['/home/lcoop22/TorqueOut/' cancerType '/'];
Mem = '512'; %free memory
Prefix = 'Torque.';

%% TODO: SET EXPERIMENT PARAMETERS
stepSizeMin = 5e-3;
stepSizeMax = 1e-2;
maxiter = 300;
pretrainV = [1];
augment = 1;
dropoutFractionV = [0 0.5];
learningRateV = [1];
maxLayersInExperiment = 10;
minLayersInExperiment = 2;
hSizeV = [20 40 60 80 100 120 140];
jumpLayers = 1;
inputZeroMaskedFraction = 0;
K = 3;
testApproach = 0; % do not do k-fold cvls ls 
randLastLayerV = [0];
randAllLayersV = [0];
removeBiasInCindex = 0;

for id = minLayersInExperiment:jumpLayers:maxLayersInExperiment
    for pretrain = pretrainV
        for hSize = hSizeV
            for dropoutFraction = dropoutFractionV
                for learningRate = learningRateV
                    for randLastLayer = randLastLayerV
                        for randAllLayers = randAllLayersV
                            idstr = ['hs' num2str(hSize) '-do' num2str(dropoutFraction) '-i' num2str(maxiter) '-ss' num2str(stepSizeMax) '-pretrain' num2str(pretrain)];
                            mkdir(['/home/syouse3/git/survivalnet/survivalnet/NNSA-master/' cancerType '_results/' idstr]);    
                            path = ['/home/syouse3/git/survivalnet/survivalnet/NNSA-master/' cancerType '_results/' idstr '/'];


                            %generate job string            
%                            JobGen(pretrain, hSize, augment, id, stepSizeMax, stepSizeMin, maxiter, learningRate, dropoutFraction, inputZeroMaskedFraction, K, testApproach, randLastLayer, randAllLayers, removeBiasInCindex, path);
                            Job = sprintf('matlab -nojvm -nodesktop -nosplash -logfile "%s" -r "addpath(genpath(''/home/syouse3/git/survivalnet/survivalnet/NNSA-master/'')); JobGen(%d, %d, %d, %d, %f, %f, %d, %f, %f, %f, %d, %d, %d, %d, %d, ''%s''); exit;"',...
                            [OutputFolder Prefix idstr '-' num2str(id) '.txt'],...
                            pretrain, hSize, augment, id, stepSizeMax, stepSizeMin, maxiter, learningRate, dropoutFraction, inputZeroMaskedFraction, K, testApproach, randLastLayer, randAllLayers, removeBiasInCindex, path);

%                            submit job
                           [status, result] = SubmitTorqueJob(Job, [Prefix idstr '-' num2str(id)], Mem);

                           %update console
                           fprintf('job %d, folder: %s, status: %s.', id, num2str(id), result);
                        end
                    end
                end
            end
        end
    end
end
