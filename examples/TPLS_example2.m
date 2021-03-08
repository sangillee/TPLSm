% TPLS_example2.m
% Written by Arthur 3/8/2021
% this script shows how one can use T-PLS to assess cross-validation performance
% to see how to use T-PLS to build a predictor, see example 1
% be sure to run line-by-line to follow along
clear
clc

%%%%%%%%%%%%% data explanation %%%%%%%%%%%%%%
% be sure to run this script in the examples directory where there's example data
% also make sure to have added the TPLSm folder to your path
load('TPLSdat.mat')

% this should load X, Y, run, subj, mask
% mask is the 3d brain image mask 
% see how many voxels there are inside the mask
% this should say that there are 3714 voxels
sum(mask(:))

% X is the single trial betas. It has 3714 columns, each of which corresponds to a voxel
% Y is binary variable to be predicted. In this case, the Y was whether the participant chose left or right button
% hopefully, when we create whole-brain predictor, we should be able to see left and right motor areas

% subj is a numerical variable that tells us the subject number that each observation belongs to.
% In this dataset, there are only 3 datasets.
% run is a numerical variable that tells us the scanner run that each observation belongs to.
% In this dataset, each of the 3 subjects had 8 scan runs.



%%%%%%%%%%%%%%%%%%% Cross Validation %%%%%%%%%%%%%%%%%%%%%
% There are only 3 subjects in this dataset, so we will do 3-fold CV
% This entails repeating the following step 3 times
% 1. Divide the data into training and testing. In this case, 2 subjects in training and 1 subject in testing.
% 2. Using just the training data (i.e., 2 subjects), do secondary cross-validation to choose best tuning parameter
% 3. Based on the best tuning parameter, fit a whole-brain predictor using all training data (2 subjects).
% 4. Assess how well the left out subject is predicted
% 5. Repeat 1~4 

AUCstorage = nan(3,1);
ACCstorage = nan(3,1);
for i = 1:3 % primary cross-validatio fold
    test = subj==i;
    train = ~test;
    
    % perform Cross-validation within training data
    cvmdl = TPLS_cv(X(train,:),Y(train),subj(train));
    cvstats = evalTuningParam(cvmdl,'AUC',X(train,:),Y(train),1:25,0:.05:1,run(train));
    
    % fit T-PLS model using all training data based on best tuning parameter
    mdl = TPLS(X(train,:),Y(train));
    
    % predict the testing subject
    score = predict(mdl,cvstats.compval_best,cvstats.threshval_best,X(test,:));
    
    % assess performance of prediction
    [~,~,~,AUCstorage(i)] = perfcurve(Y(test),score,1);
    
    % alternative way to predict is to extract the predictor and use them instead of using the 'predict' function (whichever is more convenient)
    [betamap,bias] = makePredictor(mdl,cvstats.compval_best,cvstats.threshval_best);
    score = bias + X(test,:)*betamap;
    
    % let's assess accuracy this time
    prediction = score > 0.5; % anything above 0.5 in predicted probability is a hit(1)
    ACCstorage(i) = mean(prediction==Y(test));
end


% mean AUC
mean(AUCstorage) % average 73% AUC

% mean ACC
mean(ACCstorage) % average 66.6% accuracy
