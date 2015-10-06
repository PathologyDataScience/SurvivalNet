clear all;
close all;


archs = [1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20];
colors = {[51/255 0 0], [102 51 0]./255, [153 153 0]./255, [102 204 0] ./ 255, [51 255 153] ./ 255, [255 0 0] ./ 255, [0 0 204] ./ 255, [255 51 153] ./ 255, [153 51 255] ./ 255, [0 255 0] ./255, ...
    [255 128 0]./255, [255 255 0]./255, [153 153 0]./255, [128 255 0] ./ 255, [0 255 255] ./ 255, [0 128 255] ./ 255, [128 128 128] ./ 255, [102 0 51] ./ 255, [0 0 0] ./ 255, [0 51 51] ./255};
%colors = {colors, colors * 2};
path =  '/home/syouse3/git/survivalnet/survivalnet/NNSA-master/LUSC_results/hs20-do0.5-i250-ss0.01/';
measure = '-ci-tst.mat';
for i = 2:10
    load([path 'sae-' num2str(archs(i)) measure]);
    [cindex_test(1) max(cindex_test) cindex_test(250)]
    hlines(i - 1) = plot(1:250, cindex_test(1:250), '-o');
    set(hlines(i - 1), 'color', colors{i +2});
    hold on
end
 l = legend([hlines], '2', '3', '4', '5', '6', '7', '8', '9', ...
     '10', '11', '12', '13', '14' ,'15', '16', '17', '18', '19', '20');
% cd '/home/syouse3/git/survivalnet/survivalnet/NNSA-master/results/0.0001-do0-au1-ae1/'
% measure = '-ci-tst.mat';
% for i = 1:13
%     load(['sae-' num2str(archs(i)) measure]);
%     hlines2(i) = plot(1:600, cindex_test, '--^');
%     set(hlines2(i), 'color', colors{i});
%     hold on
% end
% 

% l.Location = 'northwest';
% load('0.0001-do0-au0-ae0/sae-1-ci-tst.mat');
% hlines(1) = plot(1:300, cindex_test, '-d');
% 
% load('0.0001-do0-au0-ae1/sae-1-ci-tst.mat');
% hlines2(1) = plot(1:300, cindex_test, '-o');
% 
% set(hlines(1), 'color', [51/255 0 0]);
% set(hlines2(1), 'color', [51/255 0 0]);
% set(hlines(2), 'color', [102/255 51/255 0]);
% hold on
% 
% %load('dropout/sae-3-0.001-ci-tst.mat');
% % load('dropout/sae-3-0.001-ci-trn.mat');
%  load('dropout/sae-3-0.0001-ci-tst.mat');
% % load('dropout/sae-3-0.0001-ci-trn.mat');
% hlines(3) = plot(1:300, cindex_test);
% set(hlines(3), 'color', [153/255 153/255 0]);
% hold on
% 
% %load('dropout/sae-4-0.001-ci-tst.mat');
% % load('dropout/sae-4-0.001-ci-trn.mat');
%  load('dropout/sae-4-0.0001-ci-tst.mat');
% % load('dropout/sae-4-0.0001-ci-trn.mat');
% hlines(4) = plot(1:300, cindex_test);
% set(hlines(4), 'color', [102 204 0] ./ 255);
% hold on
% 
% %load('dropout/sae-5-0.001-ci-tst.mat');
% % load('dropout/sae-5-0.001-ci-trn.mat');
%  load('dropout/sae-5-0.0001-ci-tst.mat');
% % load('dropout/sae-5-0.0001-ci-trn.mat');
% hlines(5) = plot(1:300, cindex_test);
% set(hlines(5), 'color', [51 255 153] ./ 255);
% hold on
% 
% %load('dropout/sae-6-0.001-ci-tst.mat');
% % load('dropout/sae-6-0.001-ci-trn.mat');
%  load('dropout/sae-6-0.0001-ci-tst.mat');
% % load('dropout/sae-6-0.0001-ci-trn.mat');
% hlines(6) = plot(1:300, cindex_test);
% set(hlines(6), 'color', [102 255 255] ./ 255);
% hold on
% 
% %load('dropout/sae-7-0.001-ci-tst.mat');
% % load('dropout/sae-7-0.001-ci-trn.mat');
%  load('dropout/sae-7-0.0001-ci-tst.mat');
% % load('dropout/sae-7-0.0001-ci-trn.mat');
% hlines(7) = plot(1:300, cindex_test);
% set(hlines(7), 'color', [255 0 0] ./ 255);
% hold on
% 
% %load('dropout/sae-8-0.001-ci-tst.mat');
% % load('dropout/sae-8-0.001-ci-trn.mat');
%  load('dropout/sae-8-0.0001-ci-tst.mat');
% % load('dropout/sae-8-0.0001-ci-trn.mat');
% hlines(8) = plot(1:300, cindex_test);
% set(hlines(8), 'color', [0 0 204] ./ 255);
% hold on
% 
% %load('dropout/sae-9-0.001-ci-tst.mat');
% % load('dropout/sae-9-0.001-ci-trn.mat');
%  load('dropout/sae-9-0.0001-ci-tst.mat');
% % load('dropout/sae-9-0.0001-ci-trn.mat');
% hlines(9) = plot(1:300, cindex_test);
% set(hlines(9), 'color', [255 51 153] ./ 255);
% hold on
% 
% %load('dropout/sae-10-0.001-ci-tst.mat');
% % load('dropout/sae-10-0.001-ci-trn.mat');
%  load('dropout/sae-10-0.0001-ci-tst.mat');
% % load('dropout/sae-10-0.0001-ci-trn.mat');
% hlines(10) = plot(1:300, cindex_test);
% set(hlines(10), 'color', [153 51 255] ./ 255);
% 
%legend([hlines], '1', '2', '3', '4', '5', '6', '7', '8', '9', '10');